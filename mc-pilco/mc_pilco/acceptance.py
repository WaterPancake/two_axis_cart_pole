"""Long-horizon swing-up, centering, and disturbance acceptance checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from mc_pilco.robust import RobustMCPILCOController
from mc_pilco.policy import BoundedMLPPolicy
from mc_pilco.simulator import SCENARIOS, CartPoleSimulator, sample_initial_state


@dataclass(frozen=True)
class AcceptanceThresholds:
    upright_angle: float = 0.20
    tangent_rate: float = 1.0
    cart_position: float = 0.50
    cart_velocity: float = 0.75
    dwell_seconds: float = 2.0
    max_cart_excursion: float = 9.0


@dataclass(frozen=True)
class AcceptanceResult:
    name: str
    passed: bool
    survived: bool
    captured_before_disturbance: bool
    recovered_after_disturbance: bool
    first_capture_seconds: float | None
    final_angle: float
    final_tangent_rate: float
    final_cart_position: float
    final_cart_velocity: float
    max_cart_excursion: float
    disturbance_count: int
    disturbance_steps_applied: int
    effective_disturbance_steps: int
    affected_disturbance_count: int


def physical_metrics(state: np.ndarray) -> tuple[float, float, float, float]:
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
    return (
        float(np.arccos(np.clip(direction_z, -1.0, 1.0))),
        float(np.linalg.norm(direction_rate)),
        float(np.max(np.abs(state[0:2]))),
        float(np.linalg.norm(state[4:6])),
    )


def _inside_capture(state: np.ndarray, thresholds: AcceptanceThresholds) -> bool:
    angle, tangent_rate, cart_position, cart_velocity = physical_metrics(state)
    return bool(
        angle < thresholds.upright_angle
        and tangent_rate < thresholds.tangent_rate
        and cart_position < thresholds.cart_position
        and cart_velocity < thresholds.cart_velocity
    )


def run_acceptance_episode(
    name: str,
    initial_state: np.ndarray,
    *,
    seed: int,
    duration_seconds: float = 30.0,
    frame_skip: int = 5,
    disturbances: bool = True,
    thresholds: AcceptanceThresholds | None = None,
    learned_policy: BoundedMLPPolicy | None = None,
    residual_scale: float = 0.0,
) -> AcceptanceResult:
    """Run one episode, injecting WASD-like pulses after initial capture."""

    limits = thresholds or AcceptanceThresholds()
    simulator = CartPoleSimulator(frame_skip)
    controller = RobustMCPILCOController(
        control_dt=simulator.control_dt,
        actuator_gain=float(abs(simulator.backend.model.actuator_gear[0, 0])),
        learned_policy=learned_policy,
        residual_scale=residual_scale,
    )
    controller.reset()
    simulator.set_state(initial_state)
    rng = np.random.default_rng(seed)
    total_steps = int(round(duration_seconds / simulator.control_dt))
    dwell_steps = int(round(limits.dwell_seconds / simulator.control_dt))
    capture_streak = 0
    first_capture_step: int | None = None
    captured_before_disturbance = False
    disturbance_started = False
    disturbance_step = 0
    disturbance_count = 0
    disturbance_steps_applied = 0
    effective_disturbance_steps = 0
    affected_disturbance_count = 0
    pulse_effectful_steps = 0
    disturbance_complete = False
    recovery_streak = 0
    survived = True
    max_cart = 0.0
    # W, A, S, D plus two seeded diagonal perturbations.
    pulse_directions = [
        np.array([0.0, 0.75]),
        np.array([-0.75, 0.0]),
        np.array([0.0, -0.75]),
        np.array([0.75, 0.0]),
    ]
    random_angles = rng.uniform(-np.pi, np.pi, size=2)
    pulse_directions.extend(
        0.75 * np.stack((np.cos(random_angles), np.sin(random_angles)), axis=1)
    )
    pulse_steps = max(1, int(round(0.20 / simulator.control_dt)))
    gap_steps = max(1, int(round(0.30 / simulator.control_dt)))
    disturbance_schedule_steps = len(pulse_directions) * (pulse_steps + gap_steps)

    state = simulator.state()
    for step in range(total_steps):
        action = controller.control(state)
        if _inside_capture(state, limits):
            capture_streak += 1
            if disturbance_complete:
                recovery_streak += 1
        else:
            capture_streak = 0
            if disturbance_complete:
                recovery_streak = 0
        if capture_streak >= dwell_steps and first_capture_step is None:
            first_capture_step = step - dwell_steps + 1
            captured_before_disturbance = True
            if disturbances:
                disturbance_started = True
                disturbance_step = 0

        if disturbance_started and disturbance_step < disturbance_schedule_steps:
            cycle = pulse_steps + gap_steps
            pulse_index = disturbance_step // cycle
            within_cycle = disturbance_step % cycle
            if within_cycle < pulse_steps:
                base_action = action
                action = np.clip(base_action + pulse_directions[pulse_index], -1.0, 1.0)
                effect = float(np.max(np.abs(action - base_action)))
                disturbance_steps_applied += 1
                if effect >= 0.10:
                    pulse_effectful_steps += 1
                    effective_disturbance_steps += 1
                if within_cycle == 0:
                    disturbance_count += 1
                    pulse_effectful_steps = int(effect >= 0.10)
                required_effectful_steps = int(np.ceil(0.8 * pulse_steps))
                if (
                    within_cycle == pulse_steps - 1
                    and pulse_effectful_steps >= required_effectful_steps
                ):
                    affected_disturbance_count += 1
            disturbance_step += 1
            if disturbance_step == disturbance_schedule_steps:
                disturbance_complete = True
                recovery_streak = 0

        state, _, terminated = simulator.step(action)
        max_cart = max(max_cart, float(np.max(np.abs(state[0:2]))))
        if terminated or max_cart >= limits.max_cart_excursion:
            survived = False
            break

    final_angle, final_rate, final_position, final_velocity = physical_metrics(state)
    recovered = bool(
        disturbance_complete
        and disturbance_count == len(pulse_directions)
        and disturbance_steps_applied == len(pulse_directions) * pulse_steps
        and affected_disturbance_count == len(pulse_directions)
        and recovery_streak >= dwell_steps
    )
    passed = bool(
        survived
        and captured_before_disturbance
        and (recovered if disturbances else capture_streak >= dwell_steps)
    )
    return AcceptanceResult(
        name=name,
        passed=passed,
        survived=survived,
        captured_before_disturbance=captured_before_disturbance,
        recovered_after_disturbance=recovered,
        first_capture_seconds=(
            None if first_capture_step is None else first_capture_step * simulator.control_dt
        ),
        final_angle=final_angle,
        final_tangent_rate=final_rate,
        final_cart_position=final_position,
        final_cart_velocity=final_velocity,
        max_cart_excursion=max_cart,
        disturbance_count=disturbance_count,
        disturbance_steps_applied=disturbance_steps_applied,
        effective_disturbance_steps=effective_disturbance_steps,
        affected_disturbance_count=affected_disturbance_count,
    )


def run_acceptance_suite(
    *,
    seed: int = 17,
    random_trials: int = 4,
    duration_seconds: float = 30.0,
    disturbances: bool = True,
    frame_skip: int = 5,
    learned_policy: BoundedMLPPolicy | None = None,
    residual_scale: float = 0.0,
) -> dict[str, object]:
    if random_trials < 1:
        raise ValueError("random_trials must be at least one for robust acceptance")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")
    rng = np.random.default_rng(seed)
    starts = {name: SCENARIOS[name] for name in ("swing-x", "swing-y", "diagonal")}
    for index in range(random_trials):
        starts[f"random-{index + 1}"] = sample_initial_state(rng)
    results = [
        run_acceptance_episode(
            name,
            start,
            seed=seed + index,
            duration_seconds=duration_seconds,
            disturbances=disturbances,
            frame_skip=frame_skip,
            learned_policy=learned_policy,
            residual_scale=residual_scale,
        )
        for index, (name, start) in enumerate(starts.items())
    ]
    episode_checks_passed = all(result.passed for result in results)
    return {
        "passed": episode_checks_passed and disturbances,
        "episode_checks_passed": episode_checks_passed,
        "passed_episodes": sum(result.passed for result in results),
        "total_episodes": len(results),
        "thresholds": asdict(AcceptanceThresholds()),
        "controller": {
            "empirically_tested_hybrid_fallback": True,
            "learned_blend_active": learned_policy is not None and residual_scale > 0.0,
            "residual_scale": residual_scale,
        },
        "protocol": {
            "disturbances_enabled": disturbances,
            "duration_seconds": duration_seconds,
            "frame_skip": frame_skip,
            "random_trials": random_trials,
        },
        "results": [asdict(result) for result in results],
    }


def print_acceptance(result: dict[str, object], output: Path | None = None) -> None:
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is not None:
        project_dir = Path(__file__).resolve().parent.parent
        resolved = output.expanduser().resolve()
        try:
            resolved.relative_to(project_dir)
        except ValueError as error:
            raise ValueError(f"output must stay inside {project_dir}") from error
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
