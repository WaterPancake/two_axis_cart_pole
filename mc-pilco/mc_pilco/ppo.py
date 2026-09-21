"""Small standalone PPO baseline for the shared swing-up benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from mc_pilco.benchmark_task import (
    CONTROL_DT,
    EPISODE_STEPS,
    FRAME_SKIP,
    HARD_CART_LIMIT,
    benchmark_cost,
    evaluate_policy,
    observation_tensor,
    sample_benchmark_state,
)
from mc_pilco.simulator import CartPoleSimulator


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 31
    total_transitions: int = 100_000
    steps_per_update: int = 2_000
    epochs: int = 10
    minibatch_size: int = 250
    learning_rate: float = 3e-4
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_gradient_norm: float = 0.5
    target_kl: float = 0.03
    evaluation_interval: int = 10_000
    evaluation_episodes: int = 20
    stabilization_curriculum_transitions: int = 20_000
    output_dir: Path = Path(__file__).resolve().parent.parent / "artifacts" / "benchmark" / "ppo"


def _mlp(sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_size, output_size in zip(sizes[:-2], sizes[1:-1]):
        layer = nn.Linear(input_size, output_size)
        nn.init.orthogonal_(layer.weight, np.sqrt(2.0))
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.Tanh()))
    output = nn.Linear(sizes[-2], sizes[-1])
    nn.init.orthogonal_(output.weight, 0.01)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class PPOActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = _mlp((10, 32, 32, 2))
        self.critic = _mlp((10, 64, 64, 1))
        self.log_std = nn.Parameter(torch.full((2,), -0.5))

    def _distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(observation), self.log_std.exp())

    @staticmethod
    def _squashed_log_prob(
        distribution: torch.distributions.Normal,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(1.0 - action.square() + 1e-6)
        return (distribution.log_prob(latent) - correction).sum(-1)

    def sample(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observation)
        latent = distribution.rsample()
        action = torch.tanh(latent)
        log_probability = self._squashed_log_prob(distribution, latent, action)
        value = self.critic(observation).squeeze(-1)
        return action, log_probability, value

    def evaluate_actions(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clipped = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        latent = torch.atanh(clipped)
        distribution = self._distribution(observation)
        log_probability = self._squashed_log_prob(distribution, latent, clipped)
        entropy = distribution.entropy().sum(-1)
        value = self.critic(observation).squeeze(-1)
        return log_probability, entropy, value

    @torch.no_grad()
    def control(self, state: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(state, dtype=torch.float32)
        observation = observation_tensor(tensor)
        return torch.tanh(self.actor(observation)).cpu().numpy()

    @property
    def deployment_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.actor.parameters())


def save_ppo(policy: PPOActorCritic, path: Path, config: PPOConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "method": "ppo",
            "config": {**asdict(config), "output_dir": str(config.output_dir)},
            "state_dict": policy.state_dict(),
        },
        path,
    )


def load_ppo(path: Path) -> PPOActorCritic:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    policy = PPOActorCritic()
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy


def _collect_batch(
    simulator: CartPoleSimulator,
    policy: PPOActorCritic,
    rng: np.random.Generator,
    steps: int,
    stratum: str | None,
) -> tuple[dict[str, torch.Tensor], int]:
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[float] = []
    next_values: list[float] = []
    dones: list[float] = []
    trials = 0
    state, _ = sample_benchmark_state(rng, stratum)
    state = simulator.set_state(state)
    episode_step = 0
    for _ in range(steps):
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        observation = observation_tensor(state_tensor)
        with torch.no_grad():
            action, log_probability, value = policy.sample(observation)
        action_numpy = action.cpu().numpy()
        cost = benchmark_cost(state, action_numpy)
        next_state, _, terminated = simulator.step(action_numpy)
        episode_step += 1
        hard_failure = terminated or np.max(np.abs(next_state[0:2])) >= HARD_CART_LIMIT
        done = hard_failure or episode_step >= EPISODE_STEPS
        reward = CONTROL_DT * (1.0 - cost / 8.0) - (4.0 if hard_failure else 0.0)
        with torch.no_grad():
            next_value = (
                0.0
                if done
                else float(
                    policy.critic(
                        observation_tensor(torch.as_tensor(next_state, dtype=torch.float32))
                    ).squeeze()
                )
            )
        observations.append(observation)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        rewards.append(reward)
        next_values.append(next_value)
        dones.append(float(done))
        state = next_state
        if done:
            trials += 1
            state, _ = sample_benchmark_state(rng, stratum)
            state = simulator.set_state(state)
            episode_step = 0

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    value_tensor = torch.stack(values)
    next_value_tensor = torch.tensor(next_values, dtype=torch.float32)
    done_tensor = torch.tensor(dones, dtype=torch.float32)
    advantages = torch.zeros_like(reward_tensor)
    running_advantage = torch.tensor(0.0)
    for index in range(steps - 1, -1, -1):
        continuation = 1.0 - done_tensor[index]
        delta = (
            reward_tensor[index]
            + 0.998 * next_value_tensor[index]
            - value_tensor[index]
        )
        running_advantage = delta + 0.998 * 0.95 * continuation * running_advantage
        advantages[index] = running_advantage
    returns = advantages + value_tensor
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    return (
        {
            "observations": torch.stack(observations),
            "actions": torch.stack(actions),
            "old_log_probabilities": torch.stack(log_probabilities),
            "advantages": advantages,
            "returns": returns,
        },
        trials,
    )


def _update(
    policy: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config: PPOConfig,
    generator: torch.Generator,
) -> tuple[int, float]:
    count = batch["observations"].shape[0]
    updates = 0
    final_kl = 0.0
    for _ in range(config.epochs):
        permutation = torch.randperm(count, generator=generator)
        stop = False
        for start in range(0, count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            log_probability, entropy, value = policy.evaluate_actions(
                batch["observations"][indices], batch["actions"][indices]
            )
            old_log_probability = batch["old_log_probabilities"][indices]
            ratio = torch.exp(log_probability - old_log_probability)
            advantage = batch["advantages"][indices]
            policy_loss = -torch.minimum(
                ratio * advantage,
                ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantage,
            ).mean()
            value_loss = (value - batch["returns"][indices]).square().mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy.mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_gradient_norm)
            optimizer.step()
            updates += 1
            final_kl = float((old_log_probability - log_probability).mean().detach())
            if final_kl > config.target_kl:
                stop = True
                break
        if stop:
            break
    return updates, final_kl


def train_ppo(config: PPOConfig) -> dict[str, object]:
    if config.total_transitions < 1 or config.steps_per_update < 1:
        raise ValueError("transition counts must be positive")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    simulator = CartPoleSimulator(FRAME_SKIP)
    policy = PPOActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    transitions = 0
    trials = 0
    optimizer_updates = 0
    history: list[dict[str, object]] = []
    next_evaluation = config.evaluation_interval
    while transitions < config.total_transitions:
        batch_steps = min(config.steps_per_update, config.total_transitions - transitions)
        curriculum_stratum = (
            "stabilize"
            if transitions < config.stabilization_curriculum_transitions
            else None
        )
        batch, new_trials = _collect_batch(
            simulator, policy, rng, batch_steps, curriculum_stratum
        )
        updates, kl = _update(policy, optimizer, batch, config, generator)
        transitions += batch_steps
        trials += new_trials
        optimizer_updates += updates
        if transitions >= next_evaluation or transitions == config.total_transitions:
            evaluation = evaluate_policy(
                policy.control,
                seed=10_000 + config.seed,
                episodes_per_stratum=config.evaluation_episodes,
            )
            history.append(
                {
                    "transitions": transitions,
                    "equivalent_trials": transitions / EPISODE_STEPS,
                    "completed_training_trials": trials,
                    "optimizer_updates": optimizer_updates,
                    "approximate_kl": kl,
                    "evaluation": evaluation,
                }
            )
            save_ppo(policy, output_dir / f"checkpoint-{transitions}.pt", config)
            next_evaluation += config.evaluation_interval
    report = {
        "method": "ppo",
        "config": {**asdict(config), "output_dir": str(output_dir)},
        "real_transitions": transitions,
        "equivalent_trials": transitions / EPISODE_STEPS,
        "completed_training_trials": trials,
        "optimizer_updates": optimizer_updates,
        "wall_seconds": time.perf_counter() - started,
        "deployment_parameters": policy.deployment_parameter_count,
        "training_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "history": history,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report
