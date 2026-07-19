"""Shared rollout helpers for the PILCO-style training/evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from controllers.pilco import (
    RBFPolicy,
    mujoco_state_from_physical,
    physical_state_from_mujoco,
    swingup_cost,
)
from envs import TwoAxisInvertedPendulum

PILCO_SCENARIOS = {
    "upright": np.array([0.0, 0.0, 0.05, -0.05, 0.0, 0.0, 0.0, 0.0]),
    "swing-x": np.array([0.0, 0.0, 2.8, 0.0, 0.0, 0.0, 0.5, 0.0]),
    "swing-y": np.array([0.0, 0.0, 0.0, 2.8, 0.0, 0.0, 0.0, 0.5]),
    "diagonal": np.array([0.0, 0.0, 1.6, -1.4, 0.0, 0.0, 0.2, -0.2]),
}

PolicyFn = Callable[[np.ndarray, int], np.ndarray]


def pilco_initial_state(
    scenario: str,
    rng: np.random.Generator,
    random_cart_range: float = 0.5,
    random_angle_range: float = 1.0,
    random_cart_velocity_range: float = 0.5,
    random_pole_velocity_range: float = 1.0,
) -> np.ndarray:
    """Return a named or randomized physical initial state for PILCO scripts."""

    if scenario == "random":
        return sample_physical_state(
            rng,
            cart_range=random_cart_range,
            angle_range=random_angle_range,
            cart_velocity_range=random_cart_velocity_range,
            pole_velocity_range=random_pole_velocity_range,
        )
    if scenario not in PILCO_SCENARIOS:
        choices = ", ".join((*PILCO_SCENARIOS.keys(), "random"))
        raise ValueError(f"unknown scenario {scenario!r}; expected one of: {choices}")
    return PILCO_SCENARIOS[scenario].copy()


@dataclass
class TransitionBatch:
    """Arrays collected from environment interaction."""

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray

    @classmethod
    def empty(cls) -> "TransitionBatch":
        return cls(
            states=np.empty((0, 8), dtype=float),
            actions=np.empty((0, 2), dtype=float),
            next_states=np.empty((0, 8), dtype=float),
        )

    def append(self, other: "TransitionBatch") -> "TransitionBatch":
        if len(self.states) == 0:
            return other
        if len(other.states) == 0:
            return self
        return TransitionBatch(
            states=np.vstack([self.states, other.states]),
            actions=np.vstack([self.actions, other.actions]),
            next_states=np.vstack([self.next_states, other.next_states]),
        )


def set_physical_state(env: TwoAxisInvertedPendulum, physical_state: np.ndarray) -> None:
    """Reset MuJoCo to a state written in the physical policy convention.

    The public backend observation is always the 8D universal-joint chart, but
    the loaded MuJoCo XML may store the pole internally as either two hinges or
    a ball-joint quaternion.  Go through ``TwoAxisInvertedPendulum.set_state``
    instead of writing ``data.qpos``/``data.qvel`` directly so both model
    parameterizations work.
    """

    mujoco_state = mujoco_state_from_physical(physical_state)
    env.reset()
    env.set_state(mujoco_state[:4], mujoco_state[4:])


def sample_physical_state(
    rng: np.random.Generator,
    cart_range: float = 1.0,
    angle_range: float = np.pi,
    cart_velocity_range: float = 1.0,
    pole_velocity_range: float = 3.0,
) -> np.ndarray:
    """Sample a reset state in physical coordinates."""

    state = np.zeros(8, dtype=float)
    state[0:2] = rng.uniform(-cart_range, cart_range, size=2)
    state[2:4] = rng.uniform(-angle_range, angle_range, size=2)
    state[4:6] = rng.uniform(-cart_velocity_range, cart_velocity_range, size=2)
    state[6:8] = rng.uniform(-pole_velocity_range, pole_velocity_range, size=2)
    return state


def rollout_env(
    env: TwoAxisInvertedPendulum,
    initial_state: np.ndarray,
    policy_fn: PolicyFn,
    steps: int,
    frame_skip: int = 1,
) -> tuple[TransitionBatch, dict[str, float]]:
    """Collect one rollout and return transition arrays plus summary metrics."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if frame_skip < 1:
        raise ValueError("frame_skip must be positive")

    set_physical_state(env, initial_state)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    total_cost = 0.0
    max_abs_cart = 0.0
    max_abs_angle = 0.0

    for step in range(steps):
        state = physical_state_from_mujoco(env.get_obs())
        action = np.asarray(policy_fn(state, step), dtype=float)
        if action.shape != (2,):
            raise ValueError(f"policy returned action with shape {action.shape}")
        for _ in range(frame_skip):
            applied = env.control(action)
        nxt = physical_state_from_mujoco(env.get_obs())

        states.append(state)
        actions.append(np.asarray(applied, dtype=float).copy())
        next_states.append(nxt)
        total_cost += swingup_cost(state, applied)
        max_abs_cart = max(max_abs_cart, float(np.max(np.abs(state[0:2]))))
        max_abs_angle = max(max_abs_angle, float(np.max(np.abs(state[2:4]))))

    batch = TransitionBatch(
        states=np.asarray(states, dtype=float),
        actions=np.asarray(actions, dtype=float),
        next_states=np.asarray(next_states, dtype=float),
    )
    summary = {
        "mean_cost": total_cost / steps,
        "max_abs_cart": max_abs_cart,
        "max_abs_angle": max_abs_angle,
        "final_angle_norm": float(np.linalg.norm(next_states[-1][2:4])),
    }
    return batch, summary


def collect_rollouts(
    env: TwoAxisInvertedPendulum,
    rng: np.random.Generator,
    policy_fn: PolicyFn,
    rollouts: int,
    steps: int,
    frame_skip: int,
    cart_range: float,
    angle_range: float,
    cart_velocity_range: float,
    pole_velocity_range: float,
) -> tuple[TransitionBatch, list[dict[str, float]]]:
    """Collect multiple rollouts from randomized initial states."""

    dataset = TransitionBatch.empty()
    summaries: list[dict[str, float]] = []
    for _ in range(rollouts):
        initial_state = sample_physical_state(
            rng,
            cart_range=cart_range,
            angle_range=angle_range,
            cart_velocity_range=cart_velocity_range,
            pole_velocity_range=pole_velocity_range,
        )
        batch, summary = rollout_env(env, initial_state, policy_fn, steps, frame_skip)
        dataset = dataset.append(batch)
        summaries.append(summary)
    return dataset, summaries


def evaluate_policy_scenarios(
    env: TwoAxisInvertedPendulum,
    policy: RBFPolicy,
    steps: int,
    frame_skip: int,
) -> dict[str, dict[str, float]]:
    """Deterministically evaluate the learned policy on the named scenarios."""

    results: dict[str, dict[str, float]] = {}
    for name, start in PILCO_SCENARIOS.items():
        set_physical_state(env, start)
        total_cost = 0.0
        for _ in range(steps):
            state = physical_state_from_mujoco(env.get_obs())
            action = policy.control_physical(state)
            for _ in range(frame_skip):
                env.control(action)
            total_cost += swingup_cost(state, action)
        final = physical_state_from_mujoco(env.get_obs())
        results[name] = {
            "mean_cost": total_cost / steps,
            "final_angle_norm": float(np.linalg.norm(final[2:4])),
            "final_rate_norm": float(np.linalg.norm(final[6:8])),
            "upright": float(
                np.linalg.norm(final[2:4]) < 0.25 and np.linalg.norm(final[6:8]) < 1.0
            ),
        }
    return results


def random_piecewise_policy(
    rng: np.random.Generator,
    action_limit: float = 1.0,
    hold_steps: int = 8,
) -> PolicyFn:
    """Return an exploration policy that holds each random action briefly."""

    if hold_steps < 1:
        raise ValueError("hold_steps must be positive")
    current = rng.uniform(-action_limit, action_limit, size=2)

    def policy(_: np.ndarray, step: int) -> np.ndarray:
        nonlocal current
        if step % hold_steps == 0:
            current = rng.uniform(-action_limit, action_limit, size=2)
        return current

    return policy


def sample_optimization_starts(
    rng: np.random.Generator,
    count: int,
    cart_range: float,
    angle_range: float,
    cart_velocity_range: float,
    pole_velocity_range: float,
) -> np.ndarray:
    """Sample starts used for cheap policy rollouts inside the learned model."""

    return np.vstack(
        [
            sample_physical_state(
                rng,
                cart_range=cart_range,
                angle_range=angle_range,
                cart_velocity_range=cart_velocity_range,
                pole_velocity_range=pole_velocity_range,
            )
            for _ in range(count)
        ]
    )
