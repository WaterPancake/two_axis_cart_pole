"""Read-only adapter around the repository's MuJoCo plant."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from mc_pilco.data import TransitionDataset
from mc_pilco.state import backend_from_physical, physical_from_backend

# Importing the sibling source tree must not create bytecode outside mc-pilco.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_inserted_path = str(REPOSITORY_ROOT) not in sys.path
_previous_bytecode_setting = sys.dont_write_bytecode
sys.dont_write_bytecode = True
if _inserted_path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from envs import TwoAxisInvertedPendulum  # noqa: E402
finally:
    if _inserted_path:
        sys.path.remove(str(REPOSITORY_ROOT))
    sys.dont_write_bytecode = _previous_bytecode_setting

PolicyFunction = Callable[[np.ndarray, int], np.ndarray]

SCENARIOS = {
    "upright": np.array([0.0, 0.0, 0.05, -0.05, 0.0, 0.0, 0.0, 0.0]),
    "swing-x": np.array([0.0, 0.0, 2.8, 0.0, 0.0, 0.0, 0.5, 0.0]),
    "swing-y": np.array([0.0, 0.0, np.pi, 0.34, 0.0, 0.0, 0.0, 0.5]),
    "diagonal": np.array([0.0, 0.0, 2.7, 0.45, 0.0, 0.0, 0.2, -0.2]),
    "downward": np.array([0.0, 0.0, np.pi, 0.0, 0.0, 0.0, 0.0, 0.0]),
}


@dataclass(frozen=True)
class RolloutSummary:
    mean_cost: float
    max_abs_cart: float
    final_angle_norm: float
    final_rate_norm: float
    action_saturation: float
    terminated: bool


def numpy_swingup_cost(state: np.ndarray, action: np.ndarray) -> float:
    rail = np.maximum(np.abs(state[0:2]) - 4.8, 0.0)
    theta_x, theta_y = state[2:4]
    theta_x_dot, theta_y_dot = state[6:8]
    sin_x, cos_x = np.sin(theta_x), np.cos(theta_x)
    sin_y, cos_y = np.sin(theta_y), np.cos(theta_y)
    direction_z = cos_x * cos_y
    direction_rate = np.array(
        [
            cos_x * cos_y * theta_x_dot - sin_x * sin_y * theta_y_dot,
            cos_y * theta_y_dot,
            -sin_x * cos_y * theta_x_dot - cos_x * sin_y * theta_y_dot,
        ]
    )
    pole_error = max(1.0 - direction_z, 0.0)
    gate = np.exp(-pole_error / 0.5**2)
    return float(
        0.15 * np.sum(state[0:2] ** 2)
        + 4.0 * pole_error
        + 0.02 * np.sum(state[4:6] ** 2)
        + (0.04 + 0.35 * gate) * np.sum(direction_rate**2)
        + 0.002 * np.sum(action**2)
        + 200.0 * np.sum(rail**2)
    )


class CartPoleSimulator:
    """Physical-state simulator with held normalized controls."""

    def __init__(self, frame_skip: int = 5) -> None:
        if frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        self.backend = TwoAxisInvertedPendulum()
        self.frame_skip = int(frame_skip)
        self.physics_dt = float(self.backend.model.opt.timestep)
        self.control_dt = self.physics_dt * self.frame_skip
        self.rail_limit = float(np.min(np.abs(self.backend.model.jnt_range[:2]))) - 0.05

    def set_state(self, physical_state: np.ndarray) -> np.ndarray:
        backend_state = backend_from_physical(physical_state)
        self.backend.reset()
        self.backend.set_state(backend_state[:4], backend_state[4:])
        return self.state()

    def state(self) -> np.ndarray:
        return physical_from_backend(self.backend.get_obs())

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,):
            raise ValueError("action must have shape (2,)")
        for _ in range(self.frame_skip):
            applied = self.backend.control(action)
        state = self.state()
        terminated = not np.all(np.isfinite(state)) or bool(
            np.any(np.abs(state[0:2]) >= self.rail_limit)
        )
        return state, np.asarray(applied, dtype=np.float64), terminated

    def rollout(
        self,
        initial_state: np.ndarray,
        policy: PolicyFunction,
        steps: int,
    ) -> tuple[TransitionDataset, RolloutSummary]:
        if steps < 1:
            raise ValueError("steps must be positive")
        state = self.set_state(initial_state)
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        next_states: list[np.ndarray] = []
        costs: list[float] = []
        saturations = 0
        terminated = False
        for step in range(steps):
            requested = np.asarray(policy(state.copy(), step), dtype=np.float64)
            next_state, applied, terminated = self.step(requested)
            states.append(state.copy())
            actions.append(applied.copy())
            next_states.append(next_state.copy())
            costs.append(numpy_swingup_cost(state, applied))
            saturations += int(np.any(np.abs(applied) >= 0.999))
            state = next_state
            if terminated:
                break
        dataset = TransitionDataset(
            np.asarray(states), np.asarray(actions), np.asarray(next_states), self.control_dt
        )
        all_states = np.vstack((dataset.states, dataset.next_states[-1:]))
        summary = RolloutSummary(
            mean_cost=float(np.mean(costs)),
            max_abs_cart=float(np.max(np.abs(all_states[:, 0:2]))),
            final_angle_norm=float(np.linalg.norm(state[2:4])),
            final_rate_norm=float(np.linalg.norm(state[6:8])),
            action_saturation=saturations / len(actions),
            terminated=terminated,
        )
        return dataset, summary


def sample_initial_state(
    rng: np.random.Generator,
    *,
    cart_range: float = 0.5,
    cart_velocity_range: float = 0.5,
    pole_velocity_range: float = 1.0,
) -> np.ndarray:
    """Sample cart state and pole direction uniformly on the sphere."""

    state = np.empty(8, dtype=np.float64)
    state[0:2] = rng.uniform(-cart_range, cart_range, 2)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    horizontal_sq = direction[0] ** 2 + direction[2] ** 2
    horizontal = max(np.sqrt(horizontal_sq), 1e-6)
    state[2] = np.arctan2(direction[0], direction[2])
    state[3] = np.arctan2(direction[1], horizontal)
    state[4:6] = rng.uniform(-cart_velocity_range, cart_velocity_range, 2)
    direction_rate = rng.normal(size=3)
    direction_rate -= direction_rate.dot(direction) * direction
    norm = np.linalg.norm(direction_rate)
    direction_rate *= rng.uniform(0.0, pole_velocity_range) / max(norm, 1e-12)
    state[6] = (
        direction[2] * direction_rate[0] - direction[0] * direction_rate[2]
    ) / max(horizontal_sq, 1e-12)
    state[7] = direction_rate[1] / horizontal
    return state


def collect_rollouts(
    simulator: CartPoleSimulator,
    rng: np.random.Generator,
    policy_factory: Callable[[], PolicyFunction],
    *,
    rollouts: int,
    steps: int,
) -> tuple[TransitionDataset, list[RolloutSummary]]:
    if rollouts < 1:
        raise ValueError("rollouts must be positive")
    dataset = TransitionDataset.empty(simulator.control_dt)
    summaries: list[RolloutSummary] = []
    for _ in range(rollouts):
        rollout, summary = simulator.rollout(
            sample_initial_state(rng), policy_factory(), steps
        )
        dataset = dataset.append(rollout)
        summaries.append(summary)
    return dataset, summaries


def random_piecewise_policy(
    rng: np.random.Generator,
    *,
    hold_steps: int = 8,
) -> PolicyFunction:
    if hold_steps < 1:
        raise ValueError("hold_steps must be positive")
    action = np.zeros(2)

    def policy(_state: np.ndarray, step: int) -> np.ndarray:
        nonlocal action
        if step % hold_steps == 0:
            action = rng.uniform(-1.0, 1.0, 2)
        return action.copy()

    return policy
