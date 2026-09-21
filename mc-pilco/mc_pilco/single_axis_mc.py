"""End-to-end MC-PILCO trainer for the single-axis swing-up task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from mc_pilco.gp import IndependentGaussianProcessDynamics
from mc_pilco.single_axis_task import (
    EPISODE_STEPS,
    HARD_CART_LIMIT,
    TASK_ID,
    SingleAxisPolicy,
    SingleAxisSimulator,
    dynamics_features,
    evaluate_policy,
    integrate_velocity_change,
    sample_state,
    task_cost_tensor,
)


@dataclass(frozen=True)
class SingleAxisMCConfig:
    seed: int = 79
    initial_trials: int = 12
    iterations: int = 8
    trials_per_iteration: int = 2
    dictionary_size: int = 256
    gp_steps: int = 30
    gp_learning_rate: float = 0.03
    particles: int = 32
    imagination_horizon: int = 350
    policy_updates: int = 60
    policy_learning_rate: float = 0.003
    exploration_std: float = 0.10
    uncertainty_scale: float = 0.1
    evaluation_episodes: int = 8
    output_dir: Path = (
        Path(__file__).resolve().parent.parent / "artifacts" / "single-axis" / "mc-pilco"
    )


@dataclass
class SingleAxisDataset:
    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    control_dt: float

    @classmethod
    def empty(cls, control_dt: float) -> "SingleAxisDataset":
        return cls(np.empty((0, 4)), np.empty((0, 1)), np.empty((0, 4)), control_dt)

    def __len__(self) -> int:
        return self.states.shape[0]

    def append(self, other: "SingleAxisDataset") -> "SingleAxisDataset":
        if not np.isclose(self.control_dt, other.control_dt):
            raise ValueError("cannot combine datasets with different control intervals")
        return SingleAxisDataset(
            np.concatenate((self.states, other.states)),
            np.concatenate((self.actions, other.actions)),
            np.concatenate((self.next_states, other.next_states)),
            self.control_dt,
        )

    def model_tensors(
        self, indices: np.ndarray, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = torch.as_tensor(self.states[indices], dtype=dtype)
        actions = torch.as_tensor(self.actions[indices], dtype=dtype)
        next_states = torch.as_tensor(self.next_states[indices], dtype=dtype)
        inputs = dynamics_features(states, actions)
        targets = next_states[..., 2:4] - states[..., 2:4]
        return inputs, targets

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            states=self.states,
            actions=self.actions,
            next_states=self.next_states,
            control_dt=np.array(self.control_dt),
        )


def _collect_trial(
    simulator: SingleAxisSimulator,
    initial_state: np.ndarray,
    policy: SingleAxisPolicy | None,
    rng: np.random.Generator,
    exploration_std: float,
) -> SingleAxisDataset:
    state = simulator.set_state(initial_state)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    exploration = np.zeros(1)
    for step in range(EPISODE_STEPS):
        if step % 5 == 0:
            exploration = (
                rng.uniform(-1.0, 1.0, 1)
                if policy is None
                else rng.normal(0.0, exploration_std, 1)
            )
        base = np.zeros(1) if policy is None else policy.control(state)
        next_state, applied, terminated = simulator.step(np.clip(base + exploration, -1.0, 1.0))
        states.append(state.copy())
        actions.append(applied.copy())
        next_states.append(next_state.copy())
        state = next_state
        if terminated or abs(state[0]) >= HARD_CART_LIMIT:
            break
    return SingleAxisDataset(
        np.asarray(states), np.asarray(actions), np.asarray(next_states), simulator.control_dt
    )


def _dictionary(dataset: SingleAxisDataset, count: int, rng: np.random.Generator) -> np.ndarray:
    if count >= len(dataset):
        return np.arange(len(dataset))
    states = torch.as_tensor(dataset.states, dtype=torch.float64)
    actions = torch.as_tensor(dataset.actions, dtype=torch.float64)
    features = dynamics_features(states, actions).numpy()
    standardized = (features - features.mean(0)) / np.maximum(features.std(0), 1e-6)
    height = np.cos(dataset.states[:, 1])
    seeds: list[int] = [len(dataset) - 1]
    for candidates in (np.flatnonzero(height < -0.7), np.flatnonzero(height > 0.7)):
        if len(candidates):
            seeds.append(int(rng.choice(candidates)))
    selected = list(dict.fromkeys(seeds))
    nearest = np.full(len(dataset), np.inf)
    for index in selected:
        nearest = np.minimum(nearest, np.sum((standardized - standardized[index]) ** 2, axis=1))
    nearest[selected] = -np.inf
    while len(selected) < count:
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.sum((standardized - standardized[index]) ** 2, axis=1))
        nearest[selected] = -np.inf
    return np.asarray(selected)


def _particle_loss(
    model: IndependentGaussianProcessDynamics,
    policy: SingleAxisPolicy,
    initial_states: torch.Tensor,
    noise: torch.Tensor,
    control_dt: float,
    uncertainty_scale: float,
) -> torch.Tensor:
    state = initial_states
    total = torch.zeros(initial_states.shape[0], dtype=state.dtype)
    discount = torch.ones_like(total)
    for step in range(noise.shape[0]):
        action = policy(state)
        total = total + discount * task_cost_tensor(state, action)
        mean, variance = model.posterior(dynamics_features(state, action))
        velocity_change = mean + uncertainty_scale * variance.sqrt() * noise[step]
        limits = torch.tensor([2.0, 6.0], dtype=state.dtype, device=state.device)
        state = integrate_velocity_change(
            state, velocity_change.clamp(-limits, limits), control_dt
        )
        discount = discount * 0.998
    terminal_action = torch.zeros((state.shape[0], 1), dtype=state.dtype)
    total = total + 50.0 * discount * task_cost_tensor(state, terminal_action)
    return total.mean() / (noise.shape[0] + 50.0)


def _optimize_policy(
    model: IndependentGaussianProcessDynamics,
    policy: SingleAxisPolicy,
    config: SingleAxisMCConfig,
    rng: np.random.Generator,
    generator: torch.Generator,
    control_dt: float,
    curriculum_angle: float,
) -> list[float]:
    model.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy_learning_rate)
    history: list[float] = []
    half = config.particles // 2
    for _ in range(config.policy_updates):
        if curriculum_angle <= 0.25:
            starts = np.zeros((half, 4))
            amplitudes = (0.4, 0.15, 0.4, 0.5)
            for index in range(half):
                coordinate = index % 4
                starts[index, coordinate] = rng.uniform(
                    -amplitudes[coordinate], amplitudes[coordinate]
                )
        else:
            starts = np.asarray(
                [
                    sample_state(
                        rng,
                        "stabilize" if index % 2 == 0 else "swingup",
                        curriculum_angle=curriculum_angle,
                    )[0]
                    for index in range(half)
                ]
            )
        initial = torch.as_tensor(np.concatenate((starts, starts)), dtype=torch.float32)
        half_noise = torch.randn(
            (config.imagination_horizon, half, 2), generator=generator
        )
        noise = torch.cat((half_noise, -half_noise), dim=1)
        optimizer.zero_grad(set_to_none=True)
        loss = _particle_loss(
            model, policy, initial, noise, control_dt, config.uncertainty_scale
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def _progressive_local_refinement(
    model: IndependentGaussianProcessDynamics,
    policy: SingleAxisPolicy,
    config: SingleAxisMCConfig,
    rng: np.random.Generator,
    generator: torch.Generator,
    control_dt: float,
) -> tuple[list[float], int]:
    history: list[float] = []
    imagined_transitions = 0
    for horizon, learning_rate in (
        (15, 0.001),
        (30, 0.001),
        (60, 0.0005),
        (120, 0.0003),
    ):
        refinement = replace(
            config,
            policy_updates=250,
            imagination_horizon=horizon,
            policy_learning_rate=learning_rate,
        )
        history.extend(
            _optimize_policy(
                model,
                policy,
                refinement,
                rng,
                generator,
                control_dt,
                0.20,
            )
        )
        imagined_transitions += 250 * horizon * config.particles
    return history, imagined_transitions


def _save_checkpoint(
    path: Path,
    policy: SingleAxisPolicy,
    model: IndependentGaussianProcessDynamics,
    config: SingleAxisMCConfig,
    transitions: int,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "task": TASK_ID,
            "method": "mc-pilco-end-to-end",
            "config": {**asdict(config), "output_dir": str(config.output_dir)},
            "real_transitions": transitions,
            "policy_state_dict": policy.state_dict(),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_single_axis_mc(path: Path) -> SingleAxisPolicy:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("task") != TASK_ID:
        raise ValueError(f"checkpoint is not for {TASK_ID}")
    policy = SingleAxisPolicy()
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    return policy


def train_single_axis_mc(config: SingleAxisMCConfig) -> dict[str, object]:
    if config.particles < 2 or config.particles % 2:
        raise ValueError("particles must be a positive even number")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    simulator = SingleAxisSimulator()
    dataset = SingleAxisDataset.empty(simulator.control_dt)
    for trial in range(config.initial_trials):
        initial, _ = sample_state(rng, "stabilize" if trial % 2 == 0 else "explore")
        dataset = dataset.append(_collect_trial(simulator, initial, None, rng, 1.0))
    policy = SingleAxisPolicy()
    model = IndependentGaussianProcessDynamics(
        input_dim=6, output_dim=2, jitter=1e-4, dtype=torch.float32
    )
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = np.concatenate(
        (
            np.linspace(0.20, 2.4, max(config.iterations - 2, 1)),
            np.full(min(config.iterations, 2), np.pi),
        )
    )[: config.iterations]
    history: list[dict[str, object]] = []
    imagined_transitions = 0
    started = time.perf_counter()
    for iteration, angle in enumerate(stages, start=1):
        indices = _dictionary(dataset, config.dictionary_size, rng)
        inputs, targets = dataset.model_tensors(indices, dtype=torch.float32)
        gp_history = model.fit(
            inputs,
            targets,
            steps=config.gp_steps,
            learning_rate=config.gp_learning_rate,
        )
        if iteration == 1:
            policy_history, warmup_transitions = _progressive_local_refinement(
                model, policy, config, rng, generator, simulator.control_dt
            )
            imagined_transitions += warmup_transitions
        else:
            policy_history = _optimize_policy(
                model, policy, config, rng, generator, simulator.control_dt, float(angle)
            )
            imagined_transitions += (
                config.policy_updates * config.imagination_horizon * config.particles
            )
        for trial in range(config.trials_per_iteration):
            stratum = "stabilize" if trial == 0 else "swingup"
            initial, _ = sample_state(rng, stratum, curriculum_angle=float(angle))
            dataset = dataset.append(
                _collect_trial(
                    simulator, initial, policy, rng, config.exploration_std
                )
            )
        evaluation = evaluate_policy(
            policy.control,
            seed=20_000 + config.seed,
            episodes_per_stratum=config.evaluation_episodes,
        )
        checkpoint = output_dir / f"checkpoint-{len(dataset)}.pt"
        _save_checkpoint(checkpoint, policy, model, config, len(dataset))
        history.append(
            {
                "iteration": iteration,
                "curriculum_angle": float(angle),
                "real_transitions": len(dataset),
                "gp_nll_start": gp_history[0] if gp_history else None,
                "gp_nll_end": gp_history[-1] if gp_history else None,
                "imagined_cost_start": policy_history[0],
                "imagined_cost_end": policy_history[-1],
                "evaluation": evaluation,
                "checkpoint": str(checkpoint),
            }
        )
        if evaluation["solved"]:
            break
    if not history[-1]["evaluation"]["solved"]:
        policy_history, refinement_transitions = _progressive_local_refinement(
            model, policy, config, rng, generator, simulator.control_dt
        )
        imagined_transitions += refinement_transitions
        evaluation = evaluate_policy(
            policy.control,
            seed=20_000 + config.seed,
            episodes_per_stratum=config.evaluation_episodes,
        )
        checkpoint = output_dir / f"checkpoint-{len(dataset)}-refined.pt"
        _save_checkpoint(checkpoint, policy, model, config, len(dataset))
        history.append(
            {
                "phase": "final_model_refinement",
                "real_transitions": len(dataset),
                "imagined_cost_start": policy_history[0],
                "imagined_cost_end": policy_history[-1],
                "evaluation": evaluation,
                "checkpoint": str(checkpoint),
            }
        )
    final_path = output_dir / "checkpoint-final.pt"
    _save_checkpoint(final_path, policy, model, config, len(dataset))
    dataset.save(output_dir / "transitions.npz")
    report = {
        "task": TASK_ID,
        "method": "mc-pilco-end-to-end",
        "uses_expert_or_classical_controller": False,
        "config": {**asdict(config), "output_dir": str(output_dir)},
        "real_transitions": len(dataset),
        "imagined_transitions": imagined_transitions,
        "evaluation_transitions": (
            len(history) * 2 * config.evaluation_episodes * EPISODE_STEPS
        ),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint": str(final_path),
        "history": history,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report
