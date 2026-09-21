"""Long-horizon standalone MC-PILCO trainer for the benchmark task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from mc_pilco.benchmark_task import (
    EPISODE_STEPS,
    FRAME_SKIP,
    HARD_CART_LIMIT,
    OBSERVATION_SCALE,
    benchmark_cost_tensor,
    evaluate_policy,
    sample_benchmark_state,
)
from mc_pilco.data import TransitionDataset
from mc_pilco.gp import IndependentGaussianProcessDynamics
from mc_pilco.policy import AxisGatedPriorPolicy, BoundedMLPPolicy
from mc_pilco.policy_prior import initialize_mc_policy, prior_dataset
from mc_pilco.simulator import CartPoleSimulator
from mc_pilco.state import dynamics_features, integrate_velocity_change, pole_direction_and_rate


@dataclass(frozen=True)
class MCPILCOConfig:
    seed: int = 29
    initial_trials: int = 6
    iterations: int = 12
    trials_per_iteration: int = 2
    dictionary_size: int = 256
    gp_steps: int = 40
    gp_learning_rate: float = 0.03
    particles: int = 32
    imagination_horizon: int = 400
    policy_updates: int = 75
    policy_learning_rate: float = 0.003
    gradient_clip: float = 10.0
    exploration_std: float = 0.12
    evaluation_episodes: int = 20
    prior_steps: int = 1_000
    prior_retention: float = 0.0
    uncertainty_scale: float = 0.10
    resume_checkpoint: Path | None = None
    resume_dataset: Path | None = None
    output_dir: Path = (
        Path(__file__).resolve().parent.parent / "artifacts" / "benchmark" / "mc-pilco"
    )


def _collect_trial(
    simulator: CartPoleSimulator,
    initial_state: np.ndarray,
    policy: BoundedMLPPolicy | AxisGatedPriorPolicy | None,
    rng: np.random.Generator,
    *,
    exploration_std: float,
) -> TransitionDataset:
    state = simulator.set_state(initial_state)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    exploration = np.zeros(2)
    for step in range(EPISODE_STEPS):
        if step % 5 == 0:
            exploration = (
                rng.uniform(-1.0, 1.0, 2)
                if policy is None
                else rng.normal(0.0, exploration_std, 2)
            )
        base_action = np.zeros(2) if policy is None else policy.control(state)
        requested = np.clip(base_action + exploration, -1.0, 1.0)
        next_state, applied, terminated = simulator.step(requested)
        states.append(state.copy())
        actions.append(applied.copy())
        next_states.append(next_state.copy())
        state = next_state
        if terminated or np.max(np.abs(state[0:2])) >= HARD_CART_LIMIT:
            break
    return TransitionDataset(
        np.asarray(states), np.asarray(actions), np.asarray(next_states), simulator.control_dt
    )


def _farthest_point_dictionary(
    dataset: TransitionDataset,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select a coverage dictionary while seeding down, upright, and recent data."""

    if count >= len(dataset):
        return np.arange(len(dataset))
    states = torch.as_tensor(dataset.states, dtype=torch.float64)
    actions = torch.as_tensor(dataset.actions, dtype=torch.float64)
    features = dynamics_features(states, actions).numpy()
    standardized = (features - features.mean(0)) / np.maximum(features.std(0), 1e-6)
    direction, _ = pole_direction_and_rate(states)
    height = direction[:, 2].numpy()
    seeds: list[int] = []
    for candidates in (np.flatnonzero(height < -0.5), np.flatnonzero(height > 0.5)):
        if len(candidates):
            seeds.append(int(rng.choice(candidates)))
    seeds.append(len(dataset) - 1)
    selected = list(dict.fromkeys(seeds))
    minimum_distance = np.full(len(dataset), np.inf)
    for index in selected:
        distance = np.sum((standardized - standardized[index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    minimum_distance[selected] = -np.inf
    while len(selected) < count:
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        distance = np.sum((standardized - standardized[index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
    return np.asarray(selected)


def _initial_particles(
    rng: np.random.Generator,
    count: int,
    dtype: torch.dtype,
    stabilization_fraction: float,
) -> torch.Tensor:
    stabilization_count = int(round(count * stabilization_fraction))
    states = [
        sample_benchmark_state(
            rng, "stabilize" if index < stabilization_count else "swingup"
        )[0]
        for index in range(count)
    ]
    return torch.as_tensor(np.asarray(states), dtype=dtype)


def _particle_loss(
    model: IndependentGaussianProcessDynamics,
    policy: AxisGatedPriorPolicy,
    initial_states: torch.Tensor,
    noise: torch.Tensor,
    control_dt: float,
    uncertainty_scale: float,
) -> torch.Tensor:
    state = initial_states
    total = torch.zeros(initial_states.shape[0], dtype=state.dtype)
    for step in range(noise.shape[0]):
        action = policy(state)
        total = total + benchmark_cost_tensor(state, action)
        mean, variance = model.posterior(dynamics_features(state, action))
        velocity_change = mean + uncertainty_scale * variance.sqrt() * noise[step]
        limits = torch.tensor([2.0, 2.0, 5.0, 5.0, 5.0], dtype=state.dtype)
        state = integrate_velocity_change(
            state, torch.maximum(torch.minimum(velocity_change, limits), -limits), control_dt
        )
    terminal_action = torch.zeros((state.shape[0], 2), dtype=state.dtype)
    total = total + 25.0 * benchmark_cost_tensor(state, terminal_action)
    return total.mean() / (noise.shape[0] + 25.0)


def _optimize_policy(
    model: IndependentGaussianProcessDynamics,
    policy: AxisGatedPriorPolicy,
    config: MCPILCOConfig,
    rng: np.random.Generator,
    generator: torch.Generator,
    control_dt: float,
    stabilization_fraction: float,
    prior_states: torch.Tensor,
    prior_actions: torch.Tensor,
) -> list[float]:
    model.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_learning_rate)
    history: list[float] = []
    half_particles = config.particles // 2
    for _ in range(config.policy_updates):
        starts = _initial_particles(
            rng,
            config.particles,
            next(policy.parameters()).dtype,
            stabilization_fraction,
        )
        half_noise = torch.randn(
            (config.imagination_horizon, half_particles, model.output_dim),
            generator=generator,
            dtype=starts.dtype,
        )
        noise = torch.cat((half_noise, -half_noise), dim=1)
        optimizer.zero_grad(set_to_none=True)
        loss = _particle_loss(
            model, policy, starts, noise, control_dt, config.uncertainty_scale
        )
        prior_indices = torch.randint(
            0, prior_states.shape[0], (256,), generator=generator
        )
        prior_loss = (
            policy(prior_states[prior_indices]) - prior_actions[prior_indices]
        ).square().mean()
        loss = loss + config.prior_retention * prior_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def _save_checkpoint(
    path: Path,
    policy: AxisGatedPriorPolicy,
    model: IndependentGaussianProcessDynamics,
    config: MCPILCOConfig,
    transitions: int,
    completed_trials: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    torch.save(
        {
            "format_version": 2,
            "method": "mc-pilco-standalone",
            "config": {
                **config_payload,
                "output_dir": str(config.output_dir),
                "frame_skip": FRAME_SKIP,
            },
            "real_transitions": transitions,
            "completed_training_trials": completed_trials,
            "policy_type": "axis-gated-prior",
            "policy_state_dict": policy.state_dict(),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_standalone_mc(path: Path) -> AxisGatedPriorPolicy:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = payload["policy_state_dict"]
    prior = BoundedMLPPolicy(feature_scale=OBSERVATION_SCALE, dtype=torch.float32)
    policy = AxisGatedPriorPolicy(prior, dtype=torch.float32)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def train_mc_pilco(config: MCPILCOConfig) -> dict[str, object]:
    if config.particles < 2 or config.particles % 2:
        raise ValueError("particles must be a positive even number")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    simulator = CartPoleSimulator(FRAME_SKIP)
    dtype = torch.float32
    if (config.resume_checkpoint is None) != (config.resume_dataset is None):
        raise ValueError("resume_checkpoint and resume_dataset must be supplied together")
    resumed = config.resume_checkpoint is not None and config.resume_dataset is not None
    if resumed:
        payload = torch.load(config.resume_checkpoint, map_location="cpu", weights_only=True)
        policy = load_standalone_mc(config.resume_checkpoint)
        model = IndependentGaussianProcessDynamics.from_state_dict(
            payload["model_state_dict"], jitter=1e-4
        )
        dataset = TransitionDataset.load(config.resume_dataset)
        trials = int(payload.get("completed_training_trials", 0))
        prior_history: list[float] = []
    else:
        dataset = TransitionDataset.empty(simulator.control_dt)
        trials = 0
        for trial in range(config.initial_trials):
            stratum = "stabilize" if trial % 2 == 0 else "swingup"
            initial_state, _ = sample_benchmark_state(rng, stratum)
            dataset = dataset.append(
                _collect_trial(simulator, initial_state, None, rng, exploration_std=1.0)
            )
            trials += 1
        prior = BoundedMLPPolicy(feature_scale=OBSERVATION_SCALE, dtype=dtype)
        prior_history = initialize_mc_policy(
            prior,
            seed=config.seed,
            steps=config.prior_steps,
            control_dt=simulator.control_dt,
            actuator_gain=float(abs(simulator.backend.model.actuator_gear[0, 0])),
        )
        policy = AxisGatedPriorPolicy(prior, dtype=dtype)
        model = IndependentGaussianProcessDynamics(jitter=1e-4, dtype=dtype)
    prior_state_values, prior_action_values = prior_dataset(
        seed=config.seed,
        samples=4096,
        control_dt=simulator.control_dt,
        actuator_gain=float(abs(simulator.backend.model.actuator_gear[0, 0])),
    )
    prior_states = torch.as_tensor(prior_state_values, dtype=dtype)
    prior_actions = torch.as_tensor(prior_action_values, dtype=dtype)
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history: list[dict[str, object]] = []
    imagined_transitions = 0
    for iteration in range(config.iterations):
        dictionary = _farthest_point_dictionary(dataset, config.dictionary_size, rng)
        inputs, targets = dataset.model_tensors(dtype=dtype, indices=dictionary)
        gp_history = model.fit(
            inputs,
            targets,
            steps=config.gp_steps,
            learning_rate=config.gp_learning_rate,
        )
        stabilization_fraction = (
            0.5
            if resumed
            else 1.0
            if iteration == 0
            else 0.75
            if iteration == 1
            else 0.5
        )
        policy_history = _optimize_policy(
            model,
            policy,
            config,
            rng,
            generator,
            simulator.control_dt,
            stabilization_fraction,
            prior_states,
            prior_actions,
        )
        imagined_transitions += (
            config.policy_updates * config.imagination_horizon * config.particles
        )
        for trial in range(config.trials_per_iteration):
            stratum = "stabilize" if trial % 2 == 0 else "swingup"
            initial_state, _ = sample_benchmark_state(rng, stratum)
            dataset = dataset.append(
                _collect_trial(
                    simulator,
                    initial_state,
                    policy,
                    rng,
                    exploration_std=config.exploration_std,
                )
            )
            trials += 1
        evaluation = evaluate_policy(
            policy.control,
            seed=10_000 + config.seed,
            episodes_per_stratum=config.evaluation_episodes,
        )
        history.append(
            {
                "iteration": iteration + 1,
                "real_transitions": len(dataset),
                "equivalent_trials": len(dataset) / EPISODE_STEPS,
                "completed_training_trials": trials,
                "dictionary_size": len(dictionary),
                "gp_nll_start": gp_history[0] if gp_history else None,
                "gp_nll_end": gp_history[-1] if gp_history else None,
                "imagined_cost_start": policy_history[0] if policy_history else None,
                "imagined_cost_end": policy_history[-1] if policy_history else None,
                "evaluation": evaluation,
            }
        )
        _save_checkpoint(
            output_dir / f"checkpoint-{len(dataset)}.pt",
            policy,
            model,
            config,
            len(dataset),
            trials,
        )
    report = {
        "method": "mc-pilco-standalone",
        "config": {**asdict(config), "output_dir": str(output_dir)},
        "real_transitions": len(dataset),
        "equivalent_trials": len(dataset) / EPISODE_STEPS,
        "completed_training_trials": trials,
        "imagined_transitions": imagined_transitions,
        "wall_seconds": time.perf_counter() - started,
        "deployment_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "trainable_policy_parameters": sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        ),
        "dynamics_hyperparameters": sum(parameter.numel() for parameter in model.parameters()),
        "analytic_prior": {
            "real_transitions": 0,
            "supervised_steps": config.prior_steps,
            "mse_start": prior_history[0] if prior_history else None,
            "mse_end": prior_history[-1] if prior_history else None,
        },
        "history": history,
    }
    dataset.save(output_dir / "transitions.npz")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report
