"""End-to-end MC-PILCO system-identification and policy-search loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from mc_pilco.data import TransitionDataset
from mc_pilco.gp import IndependentGaussianProcessDynamics
from mc_pilco.objective import particle_rollout
from mc_pilco.policy import BoundedMLPPolicy
from mc_pilco.simulator import (
    SCENARIOS,
    CartPoleSimulator,
    RolloutSummary,
    collect_rollouts,
    random_piecewise_policy,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "artifacts" / "latest"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    frame_skip: int = 5
    initial_rollouts: int = 5
    trial_rollouts: int = 2
    rollout_steps: int = 100
    iterations: int = 3
    max_gp_points: int = 350
    gp_steps: int = 75
    gp_learning_rate: float = 0.03
    policy_updates: int = 100
    policy_learning_rate: float = 0.01
    particles: int = 96
    imagination_horizon: int = 100
    gradient_clip: float = 10.0
    output_dir: Path = DEFAULT_OUTPUT_DIR


def _local_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_DIR)
    except ValueError as error:
        raise ValueError(f"output_dir must stay inside {PROJECT_DIR}") from error
    return resolved


def initial_particles(
    count: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample a fixed mixture of stabilization and swing-up starts."""

    if count < 1:
        raise ValueError("count must be positive")
    bases = torch.as_tensor(np.stack(tuple(SCENARIOS.values())), dtype=dtype)
    selected = bases[torch.arange(count) % len(bases)]
    scale = torch.tensor([0.15, 0.15, 0.12, 0.12, 0.15, 0.15, 0.3, 0.3], dtype=dtype)
    return selected + scale * torch.randn((count, 8), generator=generator, dtype=dtype)


def optimize_policy(
    model: IndependentGaussianProcessDynamics,
    policy: BoundedMLPPolicy,
    *,
    particles: int,
    horizon: int,
    updates: int,
    learning_rate: float,
    gradient_clip: float,
    control_dt: float,
    generator: torch.Generator,
) -> list[float]:
    if horizon < 1 or updates < 0:
        raise ValueError("horizon must be positive and updates must be non-negative")
    model.requires_grad_(False)
    starts = initial_particles(particles, generator=generator)
    noise = torch.randn(
        (horizon, particles, model.output_dim), generator=generator, dtype=starts.dtype
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    history: list[float] = []
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = particle_rollout(
            model, policy, starts, noise, control_dt=control_dt
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), gradient_clip)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def evaluate_policy(
    simulator: CartPoleSimulator,
    policy: BoundedMLPPolicy,
    *,
    steps: int,
) -> dict[str, dict[str, float | bool]]:
    results: dict[str, dict[str, float | bool]] = {}
    for name, initial_state in SCENARIOS.items():
        started = time.perf_counter()
        _, summary = simulator.rollout(
            initial_state,
            lambda state, _step: policy.control(state),
            steps,
        )
        elapsed = time.perf_counter() - started
        values = asdict(summary)
        values["upright"] = bool(
            summary.final_angle_norm < 0.25 and summary.final_rate_norm < 1.0
        )
        values["mean_step_ms"] = 1000.0 * elapsed / steps
        results[name] = values
    return results


@torch.no_grad()
def model_rmse(
    model: IndependentGaussianProcessDynamics,
    dataset: TransitionDataset,
    *,
    max_points: int = 500,
) -> list[float]:
    indices = np.arange(min(len(dataset), max_points))
    inputs, targets = dataset.model_tensors(indices=indices)
    mean, _ = model.posterior(inputs)
    return torch.sqrt((mean - targets).square().mean(0)).cpu().tolist()


def _summary_mean(summaries: list[RolloutSummary]) -> float:
    return float(np.mean([summary.mean_cost for summary in summaries]))


def _save_artifacts(
    output_dir: Path,
    config: ExperimentConfig,
    policy: BoundedMLPPolicy,
    model: IndependentGaussianProcessDynamics,
    dataset: TransitionDataset,
    metrics: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save(output_dir / "transitions.npz")
    torch.save(
        {
            "format_version": 1,
            "config": {**asdict(config), "output_dir": str(output_dir)},
            "policy_hidden_sizes": policy.hidden_sizes,
            "policy_state_dict": policy.state_dict(),
            "model_state_dict": model.state_dict(),
        },
        output_dir / "checkpoint.pt",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_checkpoint(checkpoint: Path) -> dict[str, object]:
    return torch.load(checkpoint, map_location="cpu", weights_only=True)


def load_policy(checkpoint: Path) -> BoundedMLPPolicy:
    payload = _load_checkpoint(checkpoint)
    state_dict = payload["policy_state_dict"]
    dtype = state_dict["network.0.weight"].dtype
    policy = BoundedMLPPolicy(tuple(payload["policy_hidden_sizes"]), dtype=dtype)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def load_dynamics(checkpoint: Path) -> IndependentGaussianProcessDynamics:
    payload = _load_checkpoint(checkpoint)
    model = IndependentGaussianProcessDynamics.from_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def checkpoint_frame_skip(checkpoint: Path) -> int:
    payload = _load_checkpoint(checkpoint)
    return int(payload["config"]["frame_skip"])


def train(config: ExperimentConfig) -> dict[str, object]:
    """Run iterative MC-PILCO and save a deployable policy checkpoint."""

    if min(
        config.initial_rollouts,
        config.trial_rollouts,
        config.rollout_steps,
        config.max_gp_points,
        config.particles,
        config.imagination_horizon,
    ) < 1:
        raise ValueError("rollout, GP, particle, and horizon counts must be positive")
    if config.iterations < 1 or config.gp_steps < 0 or config.policy_updates < 0:
        raise ValueError("iterations must be positive; optimization steps may be zero")
    output_dir = _local_output_dir(config.output_dir)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator().manual_seed(config.seed)

    simulator = CartPoleSimulator(config.frame_skip)
    dataset, initial_summaries = collect_rollouts(
        simulator,
        rng,
        lambda: random_piecewise_policy(rng),
        rollouts=config.initial_rollouts,
        steps=config.rollout_steps,
    )
    policy = BoundedMLPPolicy()
    model = IndependentGaussianProcessDynamics()
    iteration_metrics: list[dict[str, object]] = []

    for iteration in range(config.iterations):
        count = min(len(dataset), config.max_gp_points)
        indices = rng.choice(len(dataset), count, replace=False)
        inputs, targets = dataset.model_tensors(indices=indices)
        gp_history = model.fit(
            inputs,
            targets,
            steps=config.gp_steps,
            learning_rate=config.gp_learning_rate,
        )
        policy_history = optimize_policy(
            model,
            policy,
            particles=config.particles,
            horizon=config.imagination_horizon,
            updates=config.policy_updates,
            learning_rate=config.policy_learning_rate,
            gradient_clip=config.gradient_clip,
            control_dt=simulator.control_dt,
            generator=generator,
        )
        new_data, trial_summaries = collect_rollouts(
            simulator,
            rng,
            lambda: (lambda state, _step: policy.control(state)),
            rollouts=config.trial_rollouts,
            steps=config.rollout_steps,
        )
        iteration_metrics.append(
            {
                "iteration": iteration + 1,
                "dataset_size_before_trial": len(dataset),
                "gp_training_points": count,
                "gp_nll_start": gp_history[0] if gp_history else None,
                "gp_nll_end": gp_history[-1] if gp_history else None,
                "imagined_cost_start": policy_history[0] if policy_history else None,
                "imagined_cost_end": policy_history[-1] if policy_history else None,
                "real_trial_mean_cost": _summary_mean(trial_summaries),
            }
        )
        dataset = dataset.append(new_data)

    final_count = min(len(dataset), config.max_gp_points)
    final_indices = rng.choice(len(dataset), final_count, replace=False)
    final_inputs, final_targets = dataset.model_tensors(indices=final_indices)
    final_gp_history = model.fit(
        final_inputs,
        final_targets,
        steps=config.gp_steps,
        learning_rate=config.gp_learning_rate,
    )
    evaluation = evaluate_policy(simulator, policy, steps=config.rollout_steps)
    metrics: dict[str, object] = {
        "seed": config.seed,
        "physics_dt": simulator.physics_dt,
        "control_dt": simulator.control_dt,
        "initial_random_mean_cost": _summary_mean(initial_summaries),
        "dataset_size": len(dataset),
        "one_step_manifold_velocity_change_rmse": model_rmse(model, dataset),
        "final_gp_training_points": final_count,
        "final_gp_nll": final_gp_history[-1] if final_gp_history else None,
        "iterations": iteration_metrics,
        "evaluation": evaluation,
    }
    _save_artifacts(output_dir, config, policy, model, dataset, metrics)
    return metrics
