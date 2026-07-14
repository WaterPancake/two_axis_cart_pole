"""Gymnasium wrapper for the two-axis cart-pole MuJoCo backend.

Install the optional RL dependencies before importing this module:

    pip install -e ".[rl]"

The wrapper keeps :class:`envs.mujoco_interface.TwoAxisInvertedPendulum` as the
single simulation backend and adds the Gymnasium API needed by RL libraries:

    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

For learned swing-up policies, the default observation uses ``sin``/``cos``
angle features to avoid a discontinuity at +/-pi.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import mujoco.viewer
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only without optional dep
    raise ImportError(
        "envs.gymnasium_wrapper requires Gymnasium. Install it with "
        "`pip install -e '.[rl]'`."
    ) from exc

from envs.mujoco_interface import MODEL_PATH, TwoAxisInvertedPendulum

ObservationMode = Literal["trig", "raw"]


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the default dense swing-up reward.

    Reward is ``alive_bonus - weighted_cost``.  The angle term uses
    ``1 - cos(theta)`` so hanging downward costs more than small upright errors
    without needing angle wrap special-cases.
    """

    alive_bonus: float = 1.0
    cart_position: float = 0.1
    pole_angle: float = 5.0
    cart_velocity: float = 0.01
    pole_velocity: float = 0.05
    action: float = 0.001


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians into [-pi, pi]."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class TwoAxisCartPoleEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium-compatible environment for the two-axis cart-pole.

    Parameters
    ----------
    observation_mode:
        ``"trig"`` returns ``[x, y, sin(theta_x), cos(theta_x),
        sin(theta_y), cos(theta_y), x_dot, y_dot, theta_x_dot, theta_y_dot]``.
        ``"raw"`` returns the 8-state vector with wrapped physical angles.
    physical_y_axis:
        The MuJoCo ``theta_y`` sign is opposite the physical y lean convention
        used in the writeup/controllers.  Keep this ``True`` for policy inputs
        in physical coordinates.
    randomize_reset:
        If true, reset samples from the ranges below.  Per-episode ``options``
        can override this with explicit ``qpos``/``qvel``/``state``.  ``state``
        uses this wrapper's physical y-axis convention; ``qpos``/``qvel`` are
        direct MuJoCo coordinates.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        xml_path: Path = MODEL_PATH,
        render_mode: Literal["human", "rgb_array"] | None = None,
        observation_mode: ObservationMode = "trig",
        max_episode_steps: int = 4_000,
        frame_skip: int = 1,
        reward_weights: RewardWeights | None = None,
        action_limit: float | np.ndarray | None = None,
        terminate_on_cart_limit: bool = True,
        cart_limit_margin: float = 0.05,
        randomize_reset: bool = False,
        reset_position_range: tuple[float, float] = (-0.5, 0.5),
        reset_angle_range: tuple[float, float] = (-np.pi, np.pi),
        reset_cart_velocity_range: tuple[float, float] = (-0.5, 0.5),
        reset_pole_velocity_range: tuple[float, float] = (-1.0, 1.0),
        physical_y_axis: bool = True,
        width: int = 480,
        height: int = 360,
    ):
        if observation_mode not in {"trig", "raw"}:
            raise ValueError("observation_mode must be 'trig' or 'raw'")
        if render_mode not in self.metadata["render_modes"] and render_mode is not None:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']} or None"
            )
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if cart_limit_margin < 0.0:
            raise ValueError("cart_limit_margin must be non-negative")

        self.backend = TwoAxisInvertedPendulum(xml_path=xml_path)
        self.model = self.backend.model
        self.data = self.backend.data

        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.reward_weights = reward_weights or RewardWeights()
        self.terminate_on_cart_limit = bool(terminate_on_cart_limit)
        self.cart_limit_margin = float(cart_limit_margin)
        self.randomize_reset = bool(randomize_reset)
        self.reset_position_range = reset_position_range
        self.reset_angle_range = reset_angle_range
        self.reset_cart_velocity_range = reset_cart_velocity_range
        self.reset_pole_velocity_range = reset_pole_velocity_range
        self.physical_y_axis = bool(physical_y_axis)
        self.width = int(width)
        self.height = int(height)
        self._elapsed_steps = 0
        self._viewer = None
        self._renderer: mujoco.Renderer | None = None

        self.action_space = self._make_action_space(action_limit)
        self.observation_space = self._make_observation_space()

    def _make_action_space(self, action_limit: float | np.ndarray | None) -> spaces.Box:
        if action_limit is None:
            limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
            ranges = np.asarray(self.model.actuator_ctrlrange, dtype=np.float32)
            if ranges.shape == (2, 2) and bool(np.all(limited)):
                low = ranges[:, 0]
                high = ranges[:, 1]
            else:
                low = -np.ones(2, dtype=np.float32)
                high = np.ones(2, dtype=np.float32)
        else:
            limit = np.asarray(action_limit, dtype=np.float32)
            if limit.ndim == 0:
                limit = np.full(2, float(limit), dtype=np.float32)
            low = -np.abs(limit)
            high = np.abs(limit)

        return spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

    def _cart_ranges(self) -> tuple[np.ndarray, np.ndarray]:
        ranges = np.asarray(self.model.jnt_range[:2], dtype=np.float32)
        limited = np.asarray(self.model.jnt_limited[:2], dtype=bool)

        low = np.where(limited, ranges[:, 0], -np.inf).astype(np.float32)
        high = np.where(limited, ranges[:, 1], np.inf).astype(np.float32)
        return low, high

    def _make_observation_space(self) -> spaces.Box:
        cart_low, cart_high = self._cart_ranges()

        if self.observation_mode == "raw":
            low = np.array(
                [
                    cart_low[0],
                    cart_low[1],
                    -np.pi,
                    -np.pi,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                ],
                dtype=np.float32,
            )
            high = np.array(
                [
                    cart_high[0],
                    cart_high[1],
                    np.pi,
                    np.pi,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                ],
                dtype=np.float32,
            )
        else:
            low = np.array(
                [
                    cart_low[0],
                    cart_low[1],
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    -np.inf,
                ],
                dtype=np.float32,
            )
            high = np.array(
                [
                    cart_high[0],
                    cart_high[1],
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                ],
                dtype=np.float32,
            )

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _physical_state(self) -> np.ndarray:
        """Return [x, y, theta_x, theta_y, xdot, ydot, theta_xdot, theta_ydot]."""

        state = np.asarray(self.backend.get_obs(), dtype=np.float64).copy()
        state[2:4] = wrap_to_pi(state[2:4])

        if self.physical_y_axis:  # mujoco uses the right hand rule, need to change the sign of the y-axis for the cart's hinge.
            state[3] *= -1.0
            state[7] *= -1.0

        return state

    def _get_obs(self) -> np.ndarray:
        state = self._physical_state()

        if self.observation_mode == "raw":
            obs = state
        else:
            obs = np.array(
                [
                    state[0],
                    state[1],
                    np.sin(state[2]),
                    np.cos(state[2]),
                    np.sin(state[3]),
                    np.cos(state[3]),
                    state[4],
                    state[5],
                    state[6],
                    state[7],
                ],
                dtype=np.float64,
            )

        return obs.astype(np.float32)

    def _sample_reset_state(self) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.zeros(4, dtype=np.float64)
        qvel = np.zeros(4, dtype=np.float64)

        qpos[0:2] = self.np_random.uniform(*self.reset_position_range, size=2)
        qpos[2:4] = self.np_random.uniform(*self.reset_angle_range, size=2)
        qvel[0:2] = self.np_random.uniform(*self.reset_cart_velocity_range, size=2)
        qvel[2:4] = self.np_random.uniform(*self.reset_pole_velocity_range, size=2)

        # Convert sampled physical y lean/rate back into MuJoCo hinge_y coordinates.
        if self.physical_y_axis:
            qpos[3] *= -1.0
            qvel[3] *= -1.0

        return qpos, qvel

    def _parse_reset_options(
        self, options: dict[str, Any] | None
    ) -> tuple[np.ndarray, np.ndarray]:
        options = options or {}

        if "state" in options:
            state = np.asarray(options["state"], dtype=np.float64)
            if state.shape != (8,):
                raise ValueError("reset option 'state' must have shape (8,)")
            qpos = state[:4].copy()
            qvel = state[4:].copy()
            if self.physical_y_axis:
                qpos[3] *= -1.0
                qvel[3] *= -1.0
        elif "qpos" in options or "qvel" in options:
            qpos = np.asarray(options.get("qpos", np.zeros(4)), dtype=np.float64)
            qvel = np.asarray(options.get("qvel", np.zeros(4)), dtype=np.float64)
            if qpos.shape != (4,) or qvel.shape != (4,):
                raise ValueError(
                    "reset options 'qpos' and 'qvel' must each have shape (4,)"
                )
        elif bool(options.get("randomize", self.randomize_reset)):
            qpos, qvel = self._sample_reset_state()
        else:
            qpos = np.zeros(4, dtype=np.float64)
            qvel = np.zeros(4, dtype=np.float64)

        return qpos, qvel

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        qpos, qvel = self._parse_reset_options(options)
        mujoco.mj_resetData(self.model, self.data)
        self.backend.set_state(qpos, qvel)

        self._elapsed_steps = 0
        obs = self._get_obs()
        info = self._get_info(action=np.zeros(2, dtype=np.float32))

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        for _ in range(self.frame_skip):
            self.backend.control(action)

        self._elapsed_steps += 1
        obs = self._get_obs()
        reward, cost_terms = self._reward(action)
        terminated = self._terminated()
        truncated = self._elapsed_steps >= self.max_episode_steps
        info = self._get_info(action=action, cost_terms=cost_terms)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        weights = self.reward_weights
        state = self._physical_state()

        cart_position = float(weights.cart_position * np.sum(state[0:2] ** 2))
        pole_angle = float(weights.pole_angle * np.sum(1.0 - np.cos(state[2:4])))
        cart_velocity = float(weights.cart_velocity * np.sum(state[4:6] ** 2))
        pole_velocity = float(weights.pole_velocity * np.sum(state[6:8] ** 2))
        control = float(
            weights.action * np.sum(np.asarray(action, dtype=np.float64) ** 2)
        )
        total_cost = (
            cart_position + pole_angle + cart_velocity + pole_velocity + control
        )
        reward = float(weights.alive_bonus - total_cost)

        return reward, {
            "cart_position": cart_position,
            "pole_angle": pole_angle,
            "cart_velocity": cart_velocity,
            "pole_velocity": pole_velocity,
            "control": control,
            "total": float(total_cost),
        }

    def _terminated(self) -> bool:
        state = self._physical_state()
        if not np.all(np.isfinite(state)):
            return True

        if not self.terminate_on_cart_limit:
            return False

        cart_low, cart_high = self._cart_ranges()
        lower = cart_low + self.cart_limit_margin
        upper = cart_high - self.cart_limit_margin
        finite = np.isfinite(lower) & np.isfinite(upper)
        if np.any(finite):
            below_limit = np.any(state[0:2][finite] <= lower[finite])
            above_limit = np.any(state[0:2][finite] >= upper[finite])
            return bool(below_limit or above_limit)

        return False

    def _get_info(
        self,
        action: np.ndarray,
        cost_terms: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        physical_state = self._physical_state()
        if cost_terms is None:
            _, cost_terms = self._reward(action)

        return {
            "step": self._elapsed_steps,
            "time": float(self.data.time),
            "raw_state": np.asarray(self.backend.get_obs(), dtype=np.float64).copy(),
            "physical_state": physical_state.copy(),
            "physical_angles": physical_state[2:4].copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "cost_terms": cost_terms,
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def register_env(env_id: str = "TwoAxisCartPole-v0", **kwargs: Any) -> None:
    """Register the environment with Gymnasium.

    Example
    -------
    >>> from envs.gymnasium_wrapper import register_env
    >>> register_env()
    >>> env = gymnasium.make("TwoAxisCartPole-v0", observation_mode="trig")
    """

    from gymnasium.envs.registration import register, registry

    if env_id in registry:
        return

    register(
        id=env_id,
        entry_point="envs.gymnasium_wrapper:TwoAxisCartPoleEnv",
        kwargs=kwargs,
    )
