"""Skeleton Gymnasium wrapper for the two-axis cart-pole.

This is intentionally incomplete so you can implement it yourself.
Use ``envs/gymnasium_wrapper.py`` as the completed reference implementation.

Suggested workflow:
  1. Fill in ``__init__`` spaces.
  2. Implement ``reset`` and verify the initial observation shape.
  3. Implement ``step`` with random actions.
  4. Add reward/termination once stepping works.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import mujoco.viewer
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from envs.mujoco_interface import MODEL_PATH, TwoAxisInvertedPendulum

ObservationMode = Literal["trig", "raw"]


@dataclass(frozen=True)
class RewardWeights:
    """Weights for your dense swing-up reward."""

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
    """Gymnasium-compatible wrapper around TwoAxisInvertedPendulum"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        xml_path: Path = MODEL_PATH,
        render_mode: Literal["human", "rgb_array"] | None = None,
        observation_mode: ObservationMode = "trig",  # alt is raw
        max_episode_steps: int = 4_000,
        frame_skip: int = 1,
        reward_weights: RewardWeights | None = None,
        action_limit: float | np.ndarray | None = None,
        terminate_on_cart_limit: bool = True,
        randomize_reset: bool = False,
        width: int = 480,
        height: int = 360,
        init_pos_range: tuple[float, float] = [-0.5, 0.5],
        init_vel_range: tuple[float, float] = [-1.0, 1.0],
        init_angle_pos_range: tuple[float, float] = [-0.5, 0.5],
        init_angle_vel_range: tuple[float, float] = [-1.0, 1.0],
    ):
        # TODO: validate arguments.
        self.backend = TwoAxisInvertedPendulum(xml_path=xml_path)
        self.model = self.backend.model
        self.data = self.backend.data

        # TODO: store config values on self.
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.max_episode_steps = int(max_episode_steps)
        self.frame_skip = int(frame_skip)
        self.reward_weights = reward_weights or RewardWeights()
        self._elapsed_steps = 0
        self._viewer = None
        self._renderer = None

        self.action_space = self._make_action_space(action_limit)
        self.observation_space = self._make_observation_space()

        self.init_pos_range = init_pos_range
        self.init_vel_range = init_vel_range
        self.init_angle_pos_range = init_angle_pos_range
        self.init_angle_vel_range = init_angle_vel_range

        raise NotImplementedError

    def _make_action_space(self, action_limit: float | np.ndarray | None) -> spaces.Box:
        """Return a Box matching the two MuJoCo motor controls."""
        if action_limit is None:
            limited = np.asanyarray(self.model.actuator_ctrllimited, dtype=np.float32)
            ctrl_range = np.asanyarray(self.model.actuator_ctrlrange, dtype=np.float32)
            if ctrl_range == (2, 2) and bool(np.all(limited)):
                low = ctrl_range[:, 0]
                high = ctrl_range[:, 1]
        else:
            limit = np.asarray(action_limit, dtype=np.float32)
            if limit.ndim == 0:
                np.full(2, float(limit), dtype=np.float32)

            low = -np.abs(limit)
            high = np.abs(limit)

        return spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.astype(np.float32),
        )

        raise NotImplementedError

    def _cart_range(self) -> tuple[np.ndarray, np.ndarray]:
        ranges = np.asarray(self.model.jnt_range[:2], dtype=np.float32)
        limited = np.asarray(self.model.jnt_limited[:2], dtype=bool)

        low = np.where(limited, ranges[:, 0], -np.inf)
        high = np.where(limited, ranges[:, 1], np.inf)

        return low, high

    def _make_observation_space(self) -> spaces.Box:
        """Return the Box for either raw or trig observations."""

        """raw / trig
        x_pos
        y_pos

        x_angle / sin(x_angle), cos(x_angle)
        y_angle / sin(y_angle, cos(y_angle

        x_vel
        y_vel

        X_angle_vel
        y_angle_vel
        
        """

        cart_low, cart_high = self._cart_range()

        # trig
        low = [
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
        ]
        high = [
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
        ]

        return spaces.Box(low, high, dtype=np.float)

        raise NotImplementedError

    def _physical_state(self) -> np.ndarray:
        """Return state in the sign convention you want policies to see.

        Suggested output:
            [x, y, theta_x, theta_y, xdot, ydot, theta_xdot, theta_ydot]

        Note: in mk2.xml, MuJoCo hinge_y has the opposite sign from the
        physical y lean used by the controllers/writeup.
        """

        # TODO: start from self.backend.get_obs(), wrap angles, fix y sign.
        state = np.asarray(self.backend.get_obs(), dtype=np.float32).copy()
        state[2:4] = self.warp_to_pi(state[2:4])

        # correcting hing y-axis sign
        state[3] *= -1.0
        state[7] *= -1.0

        return state

    def _get_obs(self) -> np.ndarray:
        """Convert physical state into the selected observation format."""
        # TODO: if trig, replace theta_x/theta_y with sin/cos features.

        state = self._physical_state()

        obs = np.array(
            [
                state[0],  # x pos
                state[1],  # y pos
                np.sin(state[2]),  # sin(theta_x)
                np.sin(state[3]),  # sin(theta_y)
                np.cos(state[2]),  # cos(theta_x)
                np.cos(state[3]),  # cos(theta_y)
                state[4],  # x dot
                state[5],  # y dot
                state[6],  # theta_x dot
                state[7],  # theta_y dot
            ]
        )

        return obs

    def _create_randon_state(self) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.zeros(4, dtype=np.float32)
        qvel = np.zeros(4, dtype=np.float32)

        qpos[0:2] = np.random.uniform(
            low=self.rand_pos_range[0], high=self.rand_pos_range[1], size=2
        )
        qpos[2:4] = np.random.uniform()

        qvel[0:2]
        qvel[2:4]

    def _parse_reset_options(options: dict[str, Any] | None = None):
        """
        Returns the qpos and qvel of the initial state.
        """

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Gymnasium reset API.

        ``options`` can eventually support:
          - {"qpos": [...], "qvel": [...]} for direct MuJoCo coordinates
          - {"state": [...]} for wrapper physical coordinates
          - {"randomize": True}
        """

        # TODO: call super().reset(seed=seed).
        super.reset(seed=seed)

        # TODO: reset MuJoCo data.
        mujoco.mj_resetData(self.backend.model, self.backend.data)

        # TODO: apply qpos/qvel or randomized initial state.

        # TODO: call mujoco.mj_forward.

        # TODO: reset elapsed step count.
        self._elapsed_steps = 0

        # TODO: return obs, info.
        raise NotImplementedError

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Gymnasium step API."""

        # TODO: clip action to self.action_space.
        # TODO: call self.backend.control(action) frame_skip times.
        # TODO: increment elapsed steps.
        # TODO: compute obs, reward, terminated, truncated, info.
        # TODO: return obs, reward, terminated, truncated, info.
        raise NotImplementedError

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        """Dense reward/cost for swing-up and stabilization."""

        # TODO: suggested cost terms:
        #   cart position: x^2 + y^2
        #   pole angle: (1 - cos(theta_x)) + (1 - cos(theta_y))
        #   velocities: squared cart and angular velocities
        #   control: squared action
        # reward = alive_bonus - weighted_cost

        weights = self.reward_weights

        state = self._physical_state()

        pos_reward = weights.cart_position * np.sum(state[0:2] ** 2)
        angle_reward = weights.pole_angle * np.sum(1 - np.cos(state[2:4]))
        cart_velocity_reward = weights.cart_velocity * np.sum(state[4:6] ** 2)
        pole_velocity_reward = weights.pole_velocity * np.sum(state[6:8] ** 2)
        control_reward = np.sum(action**2)

        return float(
            weights.alive_bonus
            - (
                pos_reward
                + angle_reward
                + cart_velocity_reward
                + pole_velocity_reward
                + control_reward
            )
        )

    def _terminated(self) -> bool:
        """Return True for real failure states, not time limits."""

        # TODO: terminate on non-finite state or cart hitting rail limits.
        raise NotImplementedError

    def _get_info(
        self,
        action: np.ndarray,
        cost_terms: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Extra debug information for training/evaluation."""

        # TODO: include raw_state, physical_state, action, cost_terms, etc.
        raise NotImplementedError

    def render(self) -> np.ndarray | None:
        """Render ``human`` or ``rgb_array`` depending on render_mode."""

        # TODO human: launch/sync passive viewer.
        # TODO rgb_array: create mujoco.Renderer and return rendered pixels.
        raise NotImplementedError

    def close(self) -> None:
        """Close any viewer/renderer resources."""

        # TODO: close viewer and renderer if they were created.
        raise NotImplementedError


def register_env(env_id: str = "TwoAxisCartPole-v0", **kwargs: Any) -> None:
    """Optional helper to register your env with Gymnasium."""

    # TODO: use gymnasium.envs.registration.register.
    raise NotImplementedError
