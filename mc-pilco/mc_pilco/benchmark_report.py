"""Frozen-checkpoint nominal and adversarial benchmark reporting."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from mc_pilco.benchmark_task import (
    BenchmarkThresholds,
    EpisodeResult,
    evaluate_policy,
    inside_capture,
    run_episode,
    sample_benchmark_state,
    state_from_direction,
)
from mc_pilco.ppo_baseline import load_ppo_baseline
from mc_pilco.simulator import CartPoleSimulator
from mc_pilco.standalone_mc import load_standalone_mc

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MC_CHECKPOINT = (
    PROJECT_DIR / "artifacts" / "benchmark" / "mc-candidate" / "checkpoint-8828.pt"
)
DEFAULT_PPO_CHECKPOINT = (
    PROJECT_DIR / "artifacts" / "benchmark" / "ppo-fixed" / "checkpoint-100500.zip"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def _transfer_start(
    rng: np.random.Generator,
    family: str,
) -> np.ndarray:
    lean = rng.uniform(0.32, 0.44)
    if family == "y-axis":
        direction = np.array([0.0, np.sin(lean), -np.cos(lean)])
        direction_rate = np.array([0.0, rng.uniform(0.4, 0.6), 0.0])
        direction_rate -= direction.dot(direction_rate) * direction
    elif family == "diagonal":
        horizontal = np.sin(lean) / np.sqrt(2.0)
        direction = np.array([horizontal, horizontal, -np.cos(lean)])
        rate = rng.uniform(0.4, 0.6) / np.sqrt(2.0)
        direction_rate = np.array([rate, rate, 0.0])
        direction_rate -= direction.dot(direction_rate) * direction
    else:
        raise ValueError(f"unknown transfer family: {family}")
    return state_from_direction(
        direction,
        direction_rate,
        cart_position=rng.uniform(-0.15, 0.15, 2),
        cart_velocity=rng.uniform(-0.1, 0.1, 2),
    )


def _family_evaluation(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    family: str,
    seed: int,
    episodes: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    simulator = CartPoleSimulator(10)
    results = [
        run_episode(
            simulator,
            _transfer_start(rng, family),
            policy,
            steps=1_500,
            hard_cart_limit=9.0,
        )
        for _ in range(episodes)
    ]
    return {
        "success_rate": float(np.mean([result.success for result in results])),
        "failure_rate": float(np.mean([result.failed for result in results])),
        "median_capture_seconds_all_episodes": float(
            np.median([result.capture_seconds for result in results])
        ),
        "max_cart_position": float(max(result.max_cart_position for result in results)),
    }


def _wrapped_policy(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    rng: np.random.Generator,
    observation_noise: float = 0.0,
    delay_steps: int = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    queue = [np.zeros(2) for _ in range(delay_steps)]

    def control(state: np.ndarray) -> np.ndarray:
        observed = state + rng.normal(0.0, observation_noise, state.shape)
        action = np.asarray(policy(observed), dtype=np.float64)
        if not queue:
            return action
        queue.append(action)
        return queue.pop(0)

    return control


def _condition_evaluation(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int,
    episodes: int,
    observation_noise: float = 0.0,
    delay_steps: int = 0,
) -> dict[str, object]:
    reset_rng = np.random.default_rng(seed)
    simulator = CartPoleSimulator(10)
    report: dict[str, object] = {}
    all_results: list[EpisodeResult] = []
    for stratum_index, stratum in enumerate(("stabilize", "swingup")):
        results: list[EpisodeResult] = []
        for episode in range(episodes):
            state = sample_benchmark_state(reset_rng, stratum)[0]
            condition_rng = np.random.default_rng(
                seed + 10_000 * stratum_index + episode
            )
            conditioned = _wrapped_policy(
                policy,
                rng=condition_rng,
                observation_noise=observation_noise,
                delay_steps=delay_steps,
            )
            results.append(
                run_episode(
                    simulator,
                    state,
                    conditioned,
                    steps=1_500,
                    hard_cart_limit=9.0,
                )
            )
        all_results.extend(results)
        successful = [result.capture_seconds for result in results if result.success]
        report[stratum] = {
            "success_rate": float(np.mean([result.success for result in results])),
            "failure_rate": float(np.mean([result.failed for result in results])),
            "median_successful_capture_seconds": (
                float(np.median(successful)) if successful else None
            ),
            "max_cart_position": float(max(result.max_cart_position for result in results)),
        }
    report["solved"] = bool(
        report["stabilize"]["success_rate"] >= 0.8
        and report["swingup"]["success_rate"] >= 0.8
    )
    report["mean_action_saturation"] = float(
        np.mean([result.action_saturation for result in all_results])
    )
    return report


def _perturbation_recovery(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int,
    episodes: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    successes = 0
    captured = 0
    survived = 0
    for _ in range(episodes):
        simulator = CartPoleSimulator(10)
        state = simulator.set_state(sample_benchmark_state(rng, "swingup")[0])
        capture_streak = 0
        pulse_remaining = 0
        pulse_applied = False
        recovery_streak = 0
        failed = False
        direction = rng.uniform(-1.0, 1.0, 2)
        direction *= 0.75 / max(np.linalg.norm(direction), 1e-12)
        for _step in range(1_750):
            action = np.asarray(policy(state), dtype=np.float64)
            if inside_capture(state):
                capture_streak += 1
            else:
                capture_streak = 0
            if capture_streak >= 100 and not pulse_applied:
                pulse_applied = True
                pulse_remaining = 10
                captured += 1
            if pulse_remaining:
                action = np.clip(action + direction, -1.0, 1.0)
                pulse_remaining -= 1
                recovery_streak = 0
            elif pulse_applied:
                recovery_streak = recovery_streak + 1 if inside_capture(state) else 0
            state, _, terminated = simulator.step(action)
            if terminated or np.max(np.abs(state[0:2])) >= 9.0:
                failed = True
                break
        if not failed:
            survived += 1
        successes += int(not failed and pulse_applied and recovery_streak >= 100)
    return {
        "joint_capture_and_recovery_rate": successes / episodes,
        "capture_rate": captured / episodes,
        "recovery_given_capture_rate": successes / captured if captured else None,
        "survival_rate": survived / episodes,
    }


def adversarial_evaluation(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int,
    episodes: int = 20,
) -> dict[str, object]:
    return {
        "y_axis_transfer": _family_evaluation(
            policy, family="y-axis", seed=seed + 1, episodes=episodes
        ),
        "diagonal_transfer": _family_evaluation(
            policy, family="diagonal", seed=seed + 2, episodes=episodes
        ),
        "two_step_action_delay": _condition_evaluation(
            policy,
            seed=seed + 3,
            episodes=episodes,
            delay_steps=2,
        ),
        "observation_noise_0.01": _condition_evaluation(
            policy,
            seed=seed + 4,
            episodes=episodes,
            observation_noise=0.01,
        ),
        "post_capture_action_pulse": _perturbation_recovery(
            policy, seed=seed + 5, episodes=episodes
        ),
    }


def generate_report(
    *,
    mc_checkpoint: Path = DEFAULT_MC_CHECKPOINT,
    ppo_checkpoint: Path = DEFAULT_PPO_CHECKPOINT,
    episodes: int = 50,
    output: Path = PROJECT_DIR / "artifacts" / "benchmark" / "comparison.json",
) -> dict[str, object]:
    mc_policy = load_standalone_mc(mc_checkpoint)
    ppo_policy = load_ppo_baseline(ppo_checkpoint, require_prior=True)
    ppo_base = (
        ppo_checkpoint.with_suffix("")
        if ppo_checkpoint.suffix == ".zip"
        else ppo_checkpoint
    )
    ppo_prior = Path(f"{ppo_base}.prior.pt")
    mc_is_default = mc_checkpoint.resolve() == DEFAULT_MC_CHECKPOINT.resolve()
    ppo_is_default = ppo_checkpoint.resolve() == DEFAULT_PPO_CHECKPOINT.resolve()
    methods = {
        "mc_pilco": {
            "control": mc_policy.control,
            "checkpoint": _display_path(mc_checkpoint),
            "checkpoint_sha256": _sha256(mc_checkpoint),
            "real_transitions": 8_828 if mc_is_default else None,
            "equivalent_10s_trials": 17.656 if mc_is_default else None,
            "completed_training_trials": 20 if mc_is_default else None,
            "deployment_parameters": sum(
                parameter.numel() for parameter in mc_policy.parameters()
            ),
            "trainable_policy_parameters": sum(
                parameter.numel()
                for parameter in mc_policy.parameters()
                if parameter.requires_grad
            ),
            "imagined_transitions": 6_720_000 if mc_is_default else None,
            "approximate_training_wall_seconds": 352.0 if mc_is_default else None,
            "posterior_standard_deviation_scale_lineage": (
                [
                    {"scale": 0.15, "imagined_transitions": 5_760_000},
                    {"scale": 0.10, "imagined_transitions": 960_000},
                ]
                if mc_is_default
                else None
            ),
            "metadata_source": "curated two-stage lineage" if mc_is_default else "unavailable",
        },
        "ppo": {
            "control": ppo_policy.control,
            "checkpoint": _display_path(ppo_checkpoint),
            "checkpoint_sha256": _sha256(ppo_checkpoint),
            "neural_prior_sidecar": _display_path(ppo_prior),
            "neural_prior_sidecar_sha256": _sha256(ppo_prior),
            "real_transitions": 100_500 if ppo_is_default else None,
            "equivalent_10s_trials": 201.0 if ppo_is_default else None,
            "completed_training_trials": 236 if ppo_is_default else None,
            "deployment_parameters": 6_468,
            "imagined_transitions": 0,
            "ppo_on_policy_and_rehearsal_wall_seconds": 66.9 if ppo_is_default else None,
            "metadata_source": "ppo-fixed/report.json" if ppo_is_default else "unavailable",
            "training_composition": (
                {
                    "expert_trajectory_transitions": 10_000,
                    "dagger_transitions": 12_500,
                    "ppo_on_policy_transitions": 78_000,
                    "expert_and_dagger_trials": 45,
                }
                if ppo_is_default
                else None
            ),
        },
    }
    report: dict[str, object] = {
        "schema_version": 2,
        "protocol": {
            "training_episode_seconds": 10.0,
            "evaluation_episode_seconds": 30.0,
            "mc_training_hard_cart_limit": 4.8,
            "ppo_training_hard_cart_limit": 9.0,
            "ppo_expert_collection_limit": 9.95,
            "evaluation_hard_cart_limit": 9.0,
            "control_hz": 50,
            "nominal_episodes_per_stratum": episodes,
            "adversarial_episodes_per_case": min(episodes, 20),
            "thresholds": asdict(BenchmarkThresholds()),
            "deployment": "standalone neural policy; no runtime fallback",
            "common_prior": "zero-interaction local LQR behavior cloning",
            "prior_treatment": {
                "mc_pilco": "frozen learned local branch with gated residual",
                "ppo": "frozen learned local branch plus trainable residual",
            },
            "optimization_objectives": {
                "mc_pilco": "8-second benchmark cost plus terminal surrogate",
                "ppo": (
                    "10-second shaped reward plus expert initialization, DAgger, "
                    "and demonstration rehearsal"
                ),
            },
        },
        "methods": {},
        "development_budget_excluded": {
            "mc_pilco_real_transitions": 37_775,
            "ppo_real_transitions": 2_030_533,
            "reason": "hyperparameter and task-shaping pilots before frozen checkpoints",
        },
    }
    for name, method in methods.items():
        control = method.pop("control")
        report["methods"][name] = {
            **method,
            "nominal": evaluate_policy(
                control,
                seed=50_000,
                episodes_per_stratum=episodes,
                steps=1_500,
                hard_cart_limit=9.0,
            ),
            "adversarial": adversarial_evaluation(
                control, seed=60_000, episodes=min(episodes, 20)
            ),
        }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
