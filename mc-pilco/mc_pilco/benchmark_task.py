"""Shared standalone swing-up task used by MC-PILCO and PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch

from mc_pilco.simulator import CartPoleSimulator
from mc_pilco.state import pole_direction_and_rate, policy_features

CONTROL_DT = 0.02
FRAME_SKIP = 10
EPISODE_STEPS = 500
HARD_CART_LIMIT = 4.8
OBSERVATION_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 4.0])


@dataclass(frozen=True)
class BenchmarkThresholds:
    upright_angle: float = 0.20
    tangent_rate: float = 1.0
    cart_position: float = 0.50
    cart_velocity: float = 0.75
    dwell_steps: int = 100


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    failed: bool
    total_cost: float
    capture_seconds: float
    max_cart_position: float
    action_rms: float
    action_saturation: float
    action_total_variation: float
    transitions: int


def state_from_direction(
    direction: np.ndarray,
    direction_rate: np.ndarray,
    *,
    cart_position: np.ndarray | None = None,
    cart_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Convert manifold pole state to the physical public chart."""

    direction = np.asarray(direction, dtype=np.float64)
    direction_rate = np.asarray(direction_rate, dtype=np.float64)
    horizontal_sq = float(direction[0] ** 2 + direction[2] ** 2)
    horizontal = max(np.sqrt(horizontal_sq), 1e-8)
    state = np.zeros(8, dtype=np.float64)
    state[0:2] = np.zeros(2) if cart_position is None else cart_position
    state[2] = np.arctan2(direction[0], direction[2])
    state[3] = np.arctan2(direction[1], horizontal)
    state[4:6] = np.zeros(2) if cart_velocity is None else cart_velocity
    state[6] = (
        direction[2] * direction_rate[0] - direction[0] * direction_rate[2]
    ) / max(horizontal_sq, 1e-12)
    state[7] = direction_rate[1] / horizontal
    return state


def sample_benchmark_state(
    rng: np.random.Generator,
    stratum: str | None = None,
) -> tuple[np.ndarray, str]:
    """Sample the basic x-axis swing-up task or local two-axis stabilization."""

    selected = stratum or ("stabilize" if rng.random() < 0.5 else "swingup")
    if selected not in {"stabilize", "swingup"}:
        raise ValueError("stratum must be 'stabilize' or 'swingup'")
    state = np.zeros(8, dtype=np.float64)
    state[0:2] = rng.uniform(-0.15, 0.15, 2)
    state[4:6] = rng.uniform(-0.10, 0.10, 2)
    if selected == "stabilize":
        state[2:4] = rng.uniform(-0.08, 0.08, 2)
        state[6:8] = rng.uniform(-0.15, 0.15, 2)
    else:
        state[2] = rng.uniform(2.72, 2.88)
        state[3] = rng.uniform(-0.03, 0.03)
        state[6] = rng.uniform(0.40, 0.60)
        state[7] = rng.uniform(-0.05, 0.05)
    return state, selected


def observation_tensor(state: torch.Tensor) -> torch.Tensor:
    scale = torch.as_tensor(OBSERVATION_SCALE, dtype=state.dtype, device=state.device)
    return policy_features(state) / scale


def benchmark_cost_tensor(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    direction, direction_rate = pole_direction_and_rate(state)
    pole_error = (1.0 - direction[..., 2]).clamp_min(0.0)
    upright_gate = torch.exp(-pole_error / 0.25)
    rail = (state[..., 0:2].abs() - 4.0).clamp_min(0.0)
    return (
        0.15 * state[..., 0:2].square().sum(-1)
        + 4.0 * pole_error
        + 0.02 * state[..., 4:6].square().sum(-1)
        + 0.35 * upright_gate * direction_rate.square().sum(-1)
        + 0.002 * action.square().sum(-1)
        + 200.0 * rail.square().sum(-1)
    )


def benchmark_cost(state: np.ndarray, action: np.ndarray) -> float:
    state_tensor = torch.as_tensor(state, dtype=torch.float64)
    action_tensor = torch.as_tensor(action, dtype=torch.float64)
    return float(benchmark_cost_tensor(state_tensor, action_tensor))


def inside_capture(state: np.ndarray, thresholds: BenchmarkThresholds | None = None) -> bool:
    limits = thresholds or BenchmarkThresholds()
    tensor = torch.as_tensor(state, dtype=torch.float64)
    direction, direction_rate = pole_direction_and_rate(tensor)
    angle = float(torch.acos(direction[2].clamp(-1.0, 1.0)))
    return bool(
        angle < limits.upright_angle
        and float(torch.linalg.vector_norm(direction_rate)) < limits.tangent_rate
        and np.max(np.abs(state[0:2])) < limits.cart_position
        and np.linalg.norm(state[4:6]) < limits.cart_velocity
    )


def run_episode(
    simulator: CartPoleSimulator,
    initial_state: np.ndarray,
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    steps: int = EPISODE_STEPS,
    hard_cart_limit: float = HARD_CART_LIMIT,
) -> EpisodeResult:
    state = simulator.set_state(initial_state)
    costs: list[float] = []
    actions: list[np.ndarray] = []
    capture_streak = 0
    first_capture: int | None = None
    max_cart = 0.0
    failed = False
    for step in range(steps):
        action = np.clip(np.asarray(policy(state), dtype=np.float64), -1.0, 1.0)
        costs.append(benchmark_cost(state, action))
        actions.append(action.copy())
        state, _, terminated = simulator.step(action)
        max_cart = max(max_cart, float(np.max(np.abs(state[0:2]))))
        if inside_capture(state):
            capture_streak += 1
            dwell_steps = (BenchmarkThresholds()).dwell_steps
            if capture_streak >= dwell_steps and first_capture is None:
                first_capture = step - dwell_steps + 1
        else:
            capture_streak = 0
        if terminated or max_cart >= hard_cart_limit:
            failed = True
            break
    missing = steps - len(costs)
    total_cost = float(np.sum(costs) + 200.0 * missing)
    action_array = np.asarray(actions)
    variation = float(np.abs(np.diff(action_array, axis=0)).sum()) if len(actions) > 1 else 0.0
    return EpisodeResult(
        success=not failed and capture_streak >= (BenchmarkThresholds()).dwell_steps,
        failed=failed,
        total_cost=total_cost,
        capture_seconds=(
            steps * simulator.control_dt
            if first_capture is None
            else first_capture * simulator.control_dt
        ),
        max_cart_position=max_cart,
        action_rms=float(np.sqrt(np.mean(action_array**2))),
        action_saturation=float(np.mean(np.any(np.abs(action_array) >= 0.999, axis=1))),
        action_total_variation=variation,
        transitions=len(costs),
    )


def evaluate_policy(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int,
    episodes_per_stratum: int = 20,
    steps: int = EPISODE_STEPS,
    hard_cart_limit: float = HARD_CART_LIMIT,
) -> dict[str, object]:
    simulator = CartPoleSimulator(FRAME_SKIP)
    rng = np.random.default_rng(seed)
    report: dict[str, object] = {"thresholds": asdict(BenchmarkThresholds())}
    all_results: list[EpisodeResult] = []
    for stratum in ("stabilize", "swingup"):
        results = [
            run_episode(
                simulator,
                sample_benchmark_state(rng, stratum)[0],
                policy,
                steps=steps,
                hard_cart_limit=hard_cart_limit,
            )
            for _ in range(episodes_per_stratum)
        ]
        all_results.extend(results)
        report[stratum] = {
            "success_rate": float(np.mean([result.success for result in results])),
            "mean_cost": float(np.mean([result.total_cost for result in results])),
            "median_capture_seconds_all_episodes": float(
                np.median([result.capture_seconds for result in results])
            ),
            "median_successful_capture_seconds": (
                float(
                    np.median(
                        [result.capture_seconds for result in results if result.success]
                    )
                )
                if any(result.success for result in results)
                else None
            ),
            "failure_rate": float(np.mean([result.failed for result in results])),
            "max_cart_position": float(max(result.max_cart_position for result in results)),
        }
    report["solved"] = bool(
        report["stabilize"]["success_rate"] >= 0.8
        and report["swingup"]["success_rate"] >= 0.8
    )
    report["mean_action_rms"] = float(np.mean([result.action_rms for result in all_results]))
    report["mean_action_saturation"] = float(
        np.mean([result.action_saturation for result in all_results])
    )
    report["episode_steps"] = steps
    report["hard_cart_limit"] = hard_cart_limit
    return report
