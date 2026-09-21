"""Stable-Baselines3 PPO baseline on the exact shared benchmark task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from mc_pilco.benchmark_task import (
    CONTROL_DT,
    EPISODE_STEPS,
    FRAME_SKIP,
    OBSERVATION_SCALE,
    benchmark_cost,
    evaluate_policy,
    inside_capture,
    observation_tensor,
    sample_benchmark_state,
)
from mc_pilco.simulator import CartPoleSimulator
from mc_pilco.policy import BoundedMLPPolicy
from mc_pilco.policy_prior import initialize_mc_policy
from mc_pilco.robust import RobustMCPILCOController
from mc_pilco.state import pole_direction_and_rate
from mc_pilco.state import backend_from_physical

try:
    from stable_baselines3 import PPO
except ImportError as error:  # pragma: no cover - depends on optional benchmark extra
    raise ImportError("install the benchmark extra: pip install -e '.[benchmark]'") from error


@dataclass(frozen=True)
class PPOBaselineConfig:
    seed: int = 54
    total_transitions: int = 100_000
    stabilization_curriculum_transitions: int = 0
    evaluation_interval: int = 20_000
    evaluation_episodes: int = 20
    learning_rate: float = 3e-4
    n_steps: int = 1_000
    batch_size: int = 250
    n_epochs: int = 10
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    prior_steps: int = 1_000
    curriculum_stage_transitions: int = 1
    residual_gain: float = 20.0
    resume_checkpoint: Path | None = None
    base_transitions: int = 0
    base_ppo_timesteps: int = 0
    base_completed_trials: int = 0
    expert_trials: int = 20
    expert_pretraining_steps: int = 5_000
    synthetic_swing_samples: int = 30_000
    synthetic_swing_steps: int = 10_000
    dagger_rounds: int = 5
    dagger_trials_per_round: int = 5
    dagger_steps_per_round: int = 3_000
    rehearsal_steps: int = 2_000
    output_dir: Path = Path(__file__).resolve().parent.parent / "artifacts" / "benchmark" / "ppo"


def _blend_residual_action(
    state: np.ndarray,
    prior: BoundedMLPPolicy,
    residual: np.ndarray,
    gain: float,
) -> np.ndarray:
    tensor = torch.as_tensor(state, dtype=torch.float32)
    direction, _ = pole_direction_and_rate(tensor)
    gate = float((0.5 * (1.0 - direction[2])).clamp(0.0, 1.0))
    prior_action = np.clip(prior.control(state), -0.999, 0.999)
    latent = np.arctanh(prior_action) + gain * gate * np.asarray(residual)
    return np.tanh(latent)


class BenchmarkEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int,
        prior: BoundedMLPPolicy,
        stratum: str | None = None,
        residual_gain: float = 2.0,
    ) -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(10,), dtype=np.float32
        )
        self.simulator = CartPoleSimulator(FRAME_SKIP)
        self.rng = np.random.default_rng(seed)
        self.stratum = stratum
        self.curriculum_angle: float | None = None
        self.prior = prior
        self.residual_gain = float(residual_gain)
        self.episode_step = 0
        self.real_transitions = 0
        self.completed_trials = 0
        self.state = np.zeros(8)

    @staticmethod
    def _observation(state: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(state, dtype=torch.float32)
        return observation_tensor(tensor).numpy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.curriculum_angle is not None and self.rng.random() >= 0.25:
            initial_state = np.zeros(8, dtype=np.float64)
            initial_state[0:2] = self.rng.uniform(-0.15, 0.15, 2)
            sampled_angle = (
                self.rng.uniform(0.30, self.curriculum_angle)
                if self.rng.random() < 0.25
                else self.curriculum_angle
            )
            initial_state[2] = self.rng.uniform(
                max(0.05, sampled_angle - 0.08),
                min(2.88, sampled_angle + 0.08),
            )
            initial_state[3] = self.rng.uniform(-0.03, 0.03)
            initial_state[4:6] = self.rng.uniform(-0.1, 0.1, 2)
            initial_state[6] = (
                self.rng.uniform(0.35, 0.55)
                if sampled_angle >= 2.4
                else self.rng.uniform(-0.2, 0.2)
            )
            initial_state[7] = self.rng.uniform(-0.05, 0.05)
            selected = f"curriculum-{self.curriculum_angle:.2f}"
        else:
            initial_state, selected = sample_benchmark_state(
                self.rng, "stabilize" if self.curriculum_angle is not None else self.stratum
            )
        self.state = self.simulator.set_state(initial_state)
        self.episode_step = 0
        return self._observation(self.state), {"stratum": selected}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        residual = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        applied = _blend_residual_action(
            self.state, self.prior, residual, self.residual_gain
        )
        cost = benchmark_cost(self.state, applied)
        current_direction, _ = pole_direction_and_rate(
            torch.as_tensor(self.state, dtype=torch.float64)
        )
        next_state, _, backend_terminated = self.simulator.step(applied)
        next_direction, _ = pole_direction_and_rate(
            torch.as_tensor(next_state, dtype=torch.float64)
        )
        self.episode_step += 1
        self.real_transitions += 1
        hard_failure = backend_terminated or bool(
            np.max(np.abs(next_state[0:2])) >= 9.0
        )
        truncated = self.episode_step >= EPISODE_STEPS
        terminated = hard_failure
        height_progress = float(next_direction[2] - current_direction[2])
        reward = (
            CONTROL_DT * (1.0 - cost / 8.0)
            + 5.0 * height_progress
            + (0.10 if inside_capture(next_state) else 0.0)
            - (4.0 if hard_failure else 0.0)
        )
        self.state = next_state
        if terminated or truncated:
            self.completed_trials += 1
        return (
            self._observation(next_state),
            float(reward),
            terminated,
            truncated,
            {
                "cost": cost,
                "hard_failure": hard_failure,
                "residual_action": residual,
                "applied_action": applied,
            },
        )


class PPOController:
    def __init__(
        self,
        model: PPO,
        prior: BoundedMLPPolicy | None = None,
        residual_gain: float = 2.0,
    ) -> None:
        self.model = model
        self.prior = prior
        self.residual_gain = float(residual_gain)

    def control(self, state: np.ndarray) -> np.ndarray:
        observation = BenchmarkEnv._observation(state)
        residual, _ = self.model.predict(observation, deterministic=True)
        if self.prior is None:
            return np.asarray(residual, dtype=np.float64)
        return _blend_residual_action(
            state, self.prior, np.asarray(residual), self.residual_gain
        )


def load_ppo_baseline(path: Path, *, require_prior: bool = False) -> PPOController:
    model = PPO.load(path, device="cpu")
    base = path.with_suffix("") if path.suffix == ".zip" else path
    prior_path = Path(f"{base}.prior.pt")
    if not prior_path.is_file():
        if require_prior:
            raise FileNotFoundError(f"required PPO neural-prior sidecar is missing: {prior_path}")
        return PPOController(model)
    payload = torch.load(prior_path, map_location="cpu", weights_only=True)
    prior = BoundedMLPPolicy(feature_scale=OBSERVATION_SCALE, dtype=torch.float32)
    prior.load_state_dict(payload["state_dict"])
    prior.eval()
    return PPOController(model, prior, float(payload["residual_gain"]))


def _deployment_parameters(model: PPO) -> int:
    modules = (model.policy.mlp_extractor.policy_net, model.policy.action_net)
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def _residual_target(
    state: np.ndarray,
    expert_action: np.ndarray,
    prior: BoundedMLPPolicy,
    gain: float,
) -> np.ndarray:
    tensor = torch.as_tensor(state, dtype=torch.float32)
    direction, _ = pole_direction_and_rate(tensor)
    gate = float((0.5 * (1.0 - direction[2])).clamp(0.0, 1.0))
    if gate < 1e-3:
        return np.zeros(2, dtype=np.float32)
    prior_action = np.clip(prior.control(state), -0.999, 0.999)
    target_action = np.clip(expert_action, -0.999, 0.999)
    residual = (
        np.arctanh(target_action) - np.arctanh(prior_action)
    ) / (gain * gate)
    return np.clip(residual, -1.0, 1.0).astype(np.float32)


def _collect_expert_dataset(
    *,
    seed: int,
    trials: int,
    prior: BoundedMLPPolicy,
    residual_gain: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    rng = np.random.default_rng(seed)
    simulator = CartPoleSimulator(FRAME_SKIP)
    expert = RobustMCPILCOController(
        control_dt=simulator.control_dt,
        actuator_gain=float(abs(simulator.backend.model.actuator_gear[0, 0])),
    )
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    states: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    transitions = 0
    for trial in range(trials):
        stratum = "stabilize" if trial % 2 == 0 else "swingup"
        state = simulator.set_state(sample_benchmark_state(rng, stratum)[0])
        expert.reset()
        for _ in range(EPISODE_STEPS):
            action = expert.control(state)
            states.append(state.copy())
            expert_actions.append(np.asarray(action, dtype=np.float32))
            observations.append(BenchmarkEnv._observation(state))
            targets.append(_residual_target(state, action, prior, residual_gain))
            state, _, terminated = simulator.step(action)
            transitions += 1
            if terminated:
                break
    return (
        torch.as_tensor(np.asarray(observations), dtype=torch.float32),
        torch.as_tensor(np.asarray(targets), dtype=torch.float32),
        torch.as_tensor(np.asarray(states), dtype=torch.float32),
        torch.as_tensor(np.asarray(expert_actions), dtype=torch.float32),
        transitions,
    )


def _synthetic_swing_dataset(
    prior: BoundedMLPPolicy,
    *,
    seed: int,
    samples: int,
    residual_gain: float,
    control_dt: float,
    actuator_gain: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    states = np.zeros((samples, 8), dtype=np.float32)
    states[:, 0:2] = rng.uniform(-2.0, 2.0, (samples, 2))
    states[:, 2] = rng.uniform(-np.pi, np.pi, samples)
    states[:, 3] = rng.uniform(-0.7, 0.7, samples)
    states[:, 4:6] = rng.uniform(-2.0, 2.0, (samples, 2))
    states[:, 6:8] = rng.uniform(-5.0, 5.0, (samples, 2))
    source = RobustMCPILCOController(
        control_dt=control_dt,
        actuator_gain=actuator_gain,
    )
    swing = source.hybrid.swing_up
    actions = np.asarray(
        [swing.control(backend_from_physical(state)) for state in states]
    )
    observations = np.asarray([BenchmarkEnv._observation(state) for state in states])
    targets = np.asarray(
        [
            _residual_target(state, action, prior, residual_gain)
            for state, action in zip(states, actions)
        ]
    )
    return torch.as_tensor(observations), torch.as_tensor(targets)


def _fit_expert_residual(
    model: PPO,
    observations: torch.Tensor,
    targets: torch.Tensor,
    *,
    seed: int,
    steps: int,
) -> list[float]:
    actor_parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(
        model.policy.action_net.parameters()
    )
    optimizer = torch.optim.Adam(actor_parameters, lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    history: list[float] = []
    for _ in range(steps):
        indices = torch.randint(
            0, observations.shape[0], (512,), generator=generator
        )
        latent = model.policy.mlp_extractor.policy_net(observations[indices])
        prediction = model.policy.action_net(latent)
        loss = (prediction - targets[indices]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def _collect_dagger_dataset(
    controller: PPOController,
    prior: BoundedMLPPolicy,
    *,
    seed: int,
    trials: int,
    residual_gain: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rng = np.random.default_rng(seed)
    simulator = CartPoleSimulator(FRAME_SKIP)
    expert = RobustMCPILCOController(
        control_dt=simulator.control_dt,
        actuator_gain=float(abs(simulator.backend.model.actuator_gear[0, 0])),
    )
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    transitions = 0
    for trial in range(trials):
        stratum = "stabilize" if trial % 2 == 0 else "swingup"
        state = simulator.set_state(sample_benchmark_state(rng, stratum)[0])
        expert.reset()
        for _ in range(EPISODE_STEPS):
            label = expert.control(state)
            observations.append(BenchmarkEnv._observation(state))
            targets.append(_residual_target(state, label, prior, residual_gain))
            state, _, terminated = simulator.step(controller.control(state))
            transitions += 1
            if terminated:
                break
    return (
        torch.as_tensor(np.asarray(observations), dtype=torch.float32),
        torch.as_tensor(np.asarray(targets), dtype=torch.float32),
        transitions,
    )


def train_ppo_baseline(config: PPOBaselineConfig) -> dict[str, object]:
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    simulator = CartPoleSimulator(FRAME_SKIP)
    if config.resume_checkpoint is not None:
        loaded_controller = load_ppo_baseline(config.resume_checkpoint)
        if loaded_controller.prior is None:
            raise ValueError("resume checkpoint must include a neural prior sidecar")
        prior = loaded_controller.prior
        prior_history: list[float] = []
    else:
        prior = BoundedMLPPolicy(feature_scale=OBSERVATION_SCALE, dtype=torch.float32)
        prior_history = initialize_mc_policy(
            prior,
            seed=config.seed,
            steps=config.prior_steps,
            control_dt=simulator.control_dt,
            actuator_gain=float(abs(simulator.backend.model.actuator_gear[0, 0])),
        )
    prior.requires_grad_(False)
    prior.eval()
    env = BenchmarkEnv(
        config.seed,
        prior,
        stratum="stabilize",
        residual_gain=config.residual_gain,
    )
    if config.resume_checkpoint is not None:
        model = PPO.load(config.resume_checkpoint, env=env, device="cpu")
        model.num_timesteps = config.base_ppo_timesteps
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={
                "activation_fn": torch.nn.Tanh,
                "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                "log_std_init": -1.0,
            },
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=config.seed,
            device="cpu",
            verbose=0,
        )
    controller = PPOController(model, prior, config.residual_gain)
    expert_history: list[float] = []
    synthetic_history: list[float] = []
    expert_transitions = 0
    dagger_transitions = 0
    rehearsal_observations: torch.Tensor | None = None
    rehearsal_targets: torch.Tensor | None = None
    if config.resume_checkpoint is None and config.expert_trials > 0:
        (
            expert_observations,
            expert_targets,
            _expert_states,
            _expert_actions,
            expert_transitions,
        ) = _collect_expert_dataset(
            seed=config.seed + 1_000,
            trials=config.expert_trials,
            prior=prior,
            residual_gain=config.residual_gain,
        )
        synthetic_observations, synthetic_targets = _synthetic_swing_dataset(
            prior,
            seed=config.seed + 2_000,
            samples=config.synthetic_swing_samples,
            residual_gain=config.residual_gain,
            control_dt=env.simulator.control_dt,
            actuator_gain=float(abs(env.simulator.backend.model.actuator_gear[0, 0])),
        )
        synthetic_history = _fit_expert_residual(
            model,
            synthetic_observations,
            synthetic_targets,
            seed=config.seed + 1,
            steps=config.synthetic_swing_steps,
        )
        expert_history = _fit_expert_residual(
            model,
            expert_observations,
            expert_targets,
            seed=config.seed,
            steps=config.expert_pretraining_steps,
        )
        aggregate_observations = [expert_observations, synthetic_observations]
        aggregate_targets = [expert_targets, synthetic_targets]
        for round_index in range(config.dagger_rounds):
            dagger_observations, dagger_targets, new_transitions = (
                _collect_dagger_dataset(
                    controller,
                    prior,
                    seed=config.seed + 3_000 + round_index,
                    trials=config.dagger_trials_per_round,
                    residual_gain=config.residual_gain,
                )
            )
            dagger_transitions += new_transitions
            aggregate_observations.append(dagger_observations)
            aggregate_targets.append(dagger_targets)
            _fit_expert_residual(
                model,
                torch.cat(aggregate_observations),
                torch.cat(aggregate_targets),
                seed=config.seed + 100 + round_index,
                steps=config.dagger_steps_per_round,
            )
        rehearsal_observations = torch.cat(aggregate_observations)
        rehearsal_targets = torch.cat(aggregate_targets)
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    target = config.evaluation_interval
    env.real_transitions = config.base_transitions + expert_transitions + dagger_transitions
    env.completed_trials = config.base_completed_trials + (
        config.expert_trials
        + config.dagger_rounds * config.dagger_trials_per_round
        if config.resume_checkpoint is None
        else 0
    )
    first_learn = config.resume_checkpoint is None
    if expert_transitions:
        expert_evaluation = evaluate_policy(
            controller.control,
            seed=10_000 + config.seed,
            episodes_per_stratum=config.evaluation_episodes,
        )
        checkpoint = output_dir / f"checkpoint-{env.real_transitions}"
        model.save(checkpoint)
        torch.save(
            {
                "state_dict": prior.state_dict(),
                "residual_gain": config.residual_gain,
            },
            Path(f"{checkpoint}.prior.pt"),
        )
        history.append(
            {
                "transitions": env.real_transitions,
                "equivalent_trials": env.real_transitions / EPISODE_STEPS,
                "completed_training_trials": env.completed_trials,
                "phase": "expert_residual_initialization",
                "evaluation": expert_evaluation,
            }
        )
    while env.real_transitions < config.total_transitions:
        if env.real_transitions < config.stabilization_curriculum_transitions:
            env.stratum = "stabilize"
            env.curriculum_angle = None
        else:
            env.stratum = None
            stage = (
                env.real_transitions - config.stabilization_curriculum_transitions
            ) // config.curriculum_stage_transitions
            curriculum_angles = (0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8)
            env.curriculum_angle = curriculum_angles[min(stage, len(curriculum_angles) - 1)]
        remaining = config.total_transitions - env.real_transitions
        chunk = min(config.evaluation_interval, remaining)
        model.learn(total_timesteps=chunk, reset_num_timesteps=first_learn)
        first_learn = False
        if (
            rehearsal_observations is not None
            and rehearsal_targets is not None
            and config.rehearsal_steps > 0
        ):
            _fit_expert_residual(
                model,
                rehearsal_observations,
                rehearsal_targets,
                seed=config.seed + env.real_transitions,
                steps=config.rehearsal_steps,
            )
        evaluation = evaluate_policy(
            controller.control,
            seed=10_000 + config.seed,
            episodes_per_stratum=config.evaluation_episodes,
        )
        checkpoint = output_dir / f"checkpoint-{env.real_transitions}"
        model.save(checkpoint)
        torch.save(
            {
                "state_dict": prior.state_dict(),
                "residual_gain": config.residual_gain,
            },
            Path(f"{checkpoint}.prior.pt"),
        )
        history.append(
            {
                "transitions": env.real_transitions,
                "equivalent_trials": env.real_transitions / EPISODE_STEPS,
                "completed_training_trials": env.completed_trials,
                "evaluation": evaluation,
            }
        )
        target += config.evaluation_interval
    report = {
        "method": "ppo",
        "implementation": "stable-baselines3",
        "config": {**asdict(config), "output_dir": str(output_dir)},
        "real_transitions": env.real_transitions,
        "equivalent_trials": env.real_transitions / EPISODE_STEPS,
        "completed_training_trials": env.completed_trials,
        "wall_seconds": time.perf_counter() - started,
        "deployment_parameters": _deployment_parameters(model)
        + sum(parameter.numel() for parameter in prior.parameters()),
        "ppo_residual_parameters": _deployment_parameters(model),
        "training_parameters": sum(parameter.numel() for parameter in model.policy.parameters()),
        "analytic_prior": {
            "real_transitions": 0,
            "supervised_steps": config.prior_steps,
            "mse_start": prior_history[0] if prior_history else None,
            "mse_end": prior_history[-1] if prior_history else None,
        },
        "expert_initialization": {
            "real_transitions": expert_transitions,
            "trials": config.expert_trials,
            "supervised_steps": config.expert_pretraining_steps,
            "mse_start": expert_history[0] if expert_history else None,
            "mse_end": expert_history[-1] if expert_history else None,
            "synthetic_swing_samples": config.synthetic_swing_samples,
            "synthetic_swing_steps": config.synthetic_swing_steps,
            "synthetic_mse_start": (
                synthetic_history[0] if synthetic_history else None
            ),
            "synthetic_mse_end": (
                synthetic_history[-1] if synthetic_history else None
            ),
            "dagger_real_transitions": dagger_transitions,
            "dagger_rounds": config.dagger_rounds,
        },
        "history": history,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    env.close()
    return report
