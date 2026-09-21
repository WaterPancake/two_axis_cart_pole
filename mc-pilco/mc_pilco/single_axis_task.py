"""Single-axis cart-pole swing-up task on the MuJoCo x-axis plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from mc_pilco.simulator import CartPoleSimulator
from mc_pilco.state import wrap_angle, wrap_angle_numpy

TASK_ID = "single-axis-cart-pole-swingup-v1"
FRAME_SKIP = 10
EPISODE_STEPS = 750
HARD_CART_LIMIT = 4.8
STATE_INDICES = np.array([0, 2, 4, 6])
FEATURE_SCALE = torch.tensor([2.0, 1.0, 1.0, 4.0, 8.0])


def embed_state(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("single-axis state must have shape (4,)")
    full = np.zeros(8, dtype=np.float64)
    full[STATE_INDICES] = values
    return full


def project_state(state: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (8,):
        raise ValueError("full state must have shape (8,)")
    return values[STATE_INDICES].copy()


def state_features(state: torch.Tensor) -> torch.Tensor:
    if state.shape[-1] != 4:
        raise ValueError("single-axis state must have last dimension 4")
    features = torch.stack(
        (state[..., 0], state[..., 1].sin(), state[..., 1].cos(), state[..., 2], state[..., 3]),
        dim=-1,
    )
    return features / FEATURE_SCALE.to(dtype=state.dtype, device=state.device)


def dynamics_features(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    if action.shape != state.shape[:-1] + (1,):
        raise ValueError("action must match state batch dimensions and end in dimension 1")
    return torch.cat((state_features(state), action), dim=-1)


def integrate_velocity_change(
    state: torch.Tensor, velocity_change: torch.Tensor, control_dt: float
) -> torch.Tensor:
    if state.shape[-1] != 4 or velocity_change.shape != state.shape[:-1] + (2,):
        raise ValueError("expected state (..., 4) and velocity change (..., 2)")
    next_velocity = state[..., 2:4] + velocity_change
    next_position = state[..., 0:2] + 0.5 * float(control_dt) * (
        state[..., 2:4] + next_velocity
    )
    return torch.cat(
        (next_position[..., 0:1], wrap_angle(next_position[..., 1:2]), next_velocity), dim=-1
    )


class SingleAxisSimulator:
    """Reduced scalar-action interface to the planar MuJoCo submanifold."""

    def __init__(self, frame_skip: int = FRAME_SKIP) -> None:
        self.full = CartPoleSimulator(frame_skip)
        self.control_dt = self.full.control_dt
        self.rail_limit = self.full.rail_limit

    @property
    def backend(self):  # type: ignore[no-untyped-def]
        return self.full.backend

    def set_state(self, state: np.ndarray) -> np.ndarray:
        return project_state(self.full.set_state(embed_state(state)))

    def state(self) -> np.ndarray:
        return project_state(self.full.state())

    def planar_leakage(self) -> float:
        return float(np.max(np.abs(self.full.state()[[1, 3, 5, 7]])))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (1,):
            raise ValueError("single-axis action must have shape (1,)")
        next_state, applied, terminated = self.full.step(np.array([value[0], 0.0]))
        return project_state(next_state), applied[0:1], terminated


class SingleAxisPolicy(nn.Module):
    """Direct reflection-equivariant bounded policy with no prior or fallback."""

    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 32, dtype=dtype),
            nn.Tanh(),
            nn.Linear(32, 32, dtype=dtype),
            nn.Tanh(),
            nn.Linear(32, 1, dtype=dtype),
            nn.Tanh(),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        output = self.network[-2]
        assert isinstance(output, nn.Linear)
        nn.init.orthogonal_(output.weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = state_features(state)
        mirrored = features * torch.tensor(
            [-1.0, -1.0, 1.0, -1.0, -1.0],
            dtype=features.dtype,
            device=features.device,
        )
        return 0.5 * (self.network(features) - self.network(mirrored))

    @torch.no_grad()
    def control(self, state: np.ndarray) -> np.ndarray:
        value = torch.as_tensor(state, dtype=next(self.parameters()).dtype)
        return self(value).cpu().numpy()


def task_cost_tensor(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    pole_error = (1.0 - state[..., 1].cos()).clamp_min(0.0)
    upright_gate = torch.exp(-pole_error / 0.25)
    rail = (state[..., 0].abs() - 4.0).clamp_min(0.0)
    return (
        0.30 * state[..., 0].square()
        + 4.0 * pole_error
        + 0.04 * state[..., 2].square()
        + (0.03 + 0.70 * upright_gate) * state[..., 3].square()
        + 0.002 * action[..., 0].square()
        + 200.0 * rail.square()
    )


def task_cost(state: np.ndarray, action: np.ndarray) -> float:
    return float(
        task_cost_tensor(
            torch.as_tensor(state, dtype=torch.float64),
            torch.as_tensor(action, dtype=torch.float64),
        )
    )


def sample_state(
    rng: np.random.Generator,
    stratum: str | None = None,
    *,
    curriculum_angle: float = np.pi,
) -> tuple[np.ndarray, str]:
    selected = stratum or ("stabilize" if rng.random() < 0.25 else "swingup")
    if selected not in {"stabilize", "swingup", "explore"}:
        raise ValueError("unknown single-axis state stratum")
    state = np.zeros(4, dtype=np.float64)
    state[0] = rng.uniform(-0.15, 0.15)
    state[2] = rng.uniform(-0.10, 0.10)
    if selected == "stabilize":
        state[1] = rng.uniform(-0.08, 0.08)
        state[3] = rng.uniform(-0.15, 0.15)
    elif selected == "explore":
        state[1] = rng.uniform(-np.pi, np.pi)
        state[3] = rng.uniform(-2.0, 2.0)
    else:
        angle = min(max(float(curriculum_angle), 0.1), np.pi)
        magnitude = rng.uniform(max(0.1, 0.85 * angle), angle)
        sign = rng.choice((-1.0, 1.0))
        state[1] = sign * magnitude
        state[3] = sign * rng.uniform(-0.15, 0.15)
    return state, selected


@dataclass(frozen=True)
class CaptureThresholds:
    angle: float = 0.20
    angle_rate: float = 1.0
    cart_position: float = 0.50
    cart_velocity: float = 0.75
    dwell_steps: int = 100


def inside_capture(state: np.ndarray) -> bool:
    limits = CaptureThresholds()
    return bool(
        abs(float(wrap_angle_numpy(np.asarray(state[1])))) < limits.angle
        and abs(state[3]) < limits.angle_rate
        and abs(state[0]) < limits.cart_position
        and abs(state[2]) < limits.cart_velocity
    )


def evaluate_policy(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    seed: int = 7001,
    episodes_per_stratum: int = 20,
    steps: int = EPISODE_STEPS,
) -> dict[str, object]:
    simulator = SingleAxisSimulator()
    rng = np.random.default_rng(seed)
    report: dict[str, object] = {"thresholds": asdict(CaptureThresholds())}
    for stratum in ("stabilize", "swingup"):
        outcomes: list[dict[str, float | bool]] = []
        for _ in range(episodes_per_stratum):
            state = simulator.set_state(sample_state(rng, stratum)[0])
            streak = 0
            first_capture: int | None = None
            total_cost = 0.0
            max_cart = 0.0
            failed = False
            for step in range(steps):
                action = np.clip(np.asarray(policy(state), dtype=np.float64), -1.0, 1.0)
                total_cost += task_cost(state, action)
                state, _, terminated = simulator.step(action)
                max_cart = max(max_cart, abs(state[0]))
                streak = streak + 1 if inside_capture(state) else 0
                if streak >= CaptureThresholds().dwell_steps and first_capture is None:
                    first_capture = step - CaptureThresholds().dwell_steps + 1
                if terminated or max_cart >= HARD_CART_LIMIT:
                    failed = True
                    total_cost += 200.0 * (steps - step - 1)
                    break
            success = not failed and streak >= CaptureThresholds().dwell_steps
            outcomes.append(
                {
                    "success": success,
                    "failed": failed,
                    "cost": total_cost,
                    "capture_seconds": (
                        steps if first_capture is None else first_capture
                    )
                    * simulator.control_dt,
                    "max_cart": max_cart,
                }
            )
        successes = [item for item in outcomes if item["success"]]
        report[stratum] = {
            "success_rate": float(np.mean([item["success"] for item in outcomes])),
            "failure_rate": float(np.mean([item["failed"] for item in outcomes])),
            "mean_cost": float(np.mean([item["cost"] for item in outcomes])),
            "median_successful_capture_seconds": (
                float(np.median([item["capture_seconds"] for item in successes]))
                if successes
                else None
            ),
            "max_cart_position": float(max(item["max_cart"] for item in outcomes)),
        }
    report["solved"] = bool(
        report["stabilize"]["success_rate"] >= 0.8
        and report["swingup"]["success_rate"] >= 0.8
    )
    report["episode_steps"] = steps
    return report
