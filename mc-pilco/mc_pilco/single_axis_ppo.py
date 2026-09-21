"""End-to-end PPO trainer for the single-axis swing-up task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from mc_pilco.single_axis_task import (
    EPISODE_STEPS,
    HARD_CART_LIMIT,
    TASK_ID,
    SingleAxisSimulator,
    evaluate_policy,
    sample_state,
)


@dataclass(frozen=True)
class SingleAxisPPOConfig:
    seed: int = 83
    total_transitions: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2_048
    batch_size: int = 256
    n_epochs: int = 10
    evaluation_episodes: int = 20
    output_dir: Path = (
        Path(__file__).resolve().parent.parent / "artifacts" / "single-axis" / "ppo"
    )


class SingleAxisSwingUpEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.simulator = SingleAxisSimulator()
        self.rng = np.random.default_rng(seed)
        self.curriculum_angle = 0.35
        self.stabilization_probability = 1.0
        self.state = np.zeros(4)
        self.steps = 0
        self.real_transitions = 0

    @staticmethod
    def observation(state: np.ndarray) -> np.ndarray:
        x, theta, velocity, angle_rate = state
        return np.array(
            [x / 2.0, np.sin(theta), np.cos(theta), velocity / 4.0, angle_rate / 8.0],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        stratum = (
            "stabilize"
            if self.rng.random() < self.stabilization_probability
            else "swingup"
        )
        initial, _ = sample_state(
            self.rng, stratum, curriculum_angle=self.curriculum_angle
        )
        self.state = self.simulator.set_state(initial)
        self.steps = 0
        return self.observation(self.state), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        applied_request = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        previous_height = np.cos(self.state[1])
        next_state, applied, terminated = self.simulator.step(applied_request)
        self.steps += 1
        self.real_transitions += 1
        height = np.cos(next_state[1])
        angle = np.arctan2(np.sin(next_state[1]), np.cos(next_state[1]))
        capture_score = np.exp(
            -0.5
            * (
                (angle / 0.25) ** 2
                + (next_state[3] / 1.0) ** 2
                + (next_state[0] / 0.5) ** 2
                + (next_state[2] / 0.75) ** 2
            )
        )
        reward = (
            2.0 * height
            + 4.0 * capture_score
            + 3.0 * (height - previous_height)
            - 0.20 * next_state[0] ** 2
            - 0.03 * next_state[2] ** 2
            - 0.03 * next_state[3] ** 2
            - 0.002 * applied[0] ** 2
        )
        hard_failure = terminated or abs(next_state[0]) >= HARD_CART_LIMIT
        if hard_failure:
            reward -= 25.0
        truncated = self.steps >= EPISODE_STEPS
        self.state = next_state
        return self.observation(next_state), float(reward), hard_failure, truncated, {}


class SingleAxisPPOController:
    def __init__(self, model: PPO) -> None:
        self.model = model

    def control(self, state: np.ndarray) -> np.ndarray:
        observation = SingleAxisSwingUpEnv.observation(state)
        action, _ = self.model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)


def load_single_axis_ppo(path: Path) -> SingleAxisPPOController:
    return SingleAxisPPOController(PPO.load(path, device="cpu"))


def train_single_axis_ppo(config: SingleAxisPPOConfig) -> dict[str, object]:
    if config.total_transitions < 1:
        raise ValueError("total_transitions must be positive")
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = SingleAxisSwingUpEnv(config.seed)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=config.seed,
        device="cpu",
        verbose=0,
    )
    stages = (
        (0.10, 1.0),
        (0.35, 0.5),
        (0.7, 0.3),
        (1.2, 0.25),
        (1.8, 0.20),
        (2.4, 0.20),
        (np.pi, 0.20),
    )
    preliminary_budget = min(25_000, max(config.total_transitions // 10, 1))
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for stage_index, (angle, stabilization_probability) in enumerate(stages):
        env.curriculum_angle = angle
        env.stabilization_probability = stabilization_probability
        remaining = config.total_transitions - model.num_timesteps
        if remaining <= 0:
            break
        model.learn(
            total_timesteps=(
                remaining
                if stage_index == len(stages) - 1
                else min(preliminary_budget, remaining)
            ),
            reset_num_timesteps=stage_index == 0,
        )
        controller = SingleAxisPPOController(model)
        evaluation = evaluate_policy(
            controller.control,
            seed=20_000 + config.seed,
            episodes_per_stratum=config.evaluation_episodes,
        )
        history.append(
            {
                "curriculum_angle": float(angle),
                "stabilization_probability": stabilization_probability,
                "real_transitions": env.real_transitions,
                "evaluation": evaluation,
            }
        )
        checkpoint = output_dir / f"checkpoint-{env.real_transitions}"
        model.save(checkpoint)
    final_path = output_dir / "checkpoint-final"
    model.save(final_path)
    report = {
        "task": TASK_ID,
        "method": "ppo-end-to-end",
        "uses_expert_or_classical_controller": False,
        "config": {**asdict(config), "output_dir": str(output_dir)},
        "real_transitions": env.real_transitions,
        "evaluation_transitions": (
            len(history) * 2 * config.evaluation_episodes * EPISODE_STEPS
        ),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint": str(final_path.with_suffix(".zip")),
        "history": history,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    env.close()
    return report
