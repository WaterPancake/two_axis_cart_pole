"""State conventions and differentiable feature transforms."""

from __future__ import annotations

import math

import numpy as np
import torch

STATE_DIM = 8
ACTION_DIM = 2
POLICY_FEATURE_DIM = 10
DYNAMICS_FEATURE_DIM = 12
VELOCITY_DIM = 5


def wrap_angle_numpy(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Differentiably wrap angles to [-pi, pi)."""

    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def physical_from_backend(state: np.ndarray) -> np.ndarray:
    """Convert the backend chart into the physical policy convention."""

    values = np.asarray(state, dtype=np.float64).copy()
    if values.shape != (STATE_DIM,):
        raise ValueError(f"state must have shape ({STATE_DIM},), got {values.shape}")
    values[2:4] = wrap_angle_numpy(values[2:4])
    values[3] *= -1.0
    values[7] *= -1.0
    return values


def backend_from_physical(state: np.ndarray) -> np.ndarray:
    """Convert a physical policy state into the backend chart."""

    values = np.asarray(state, dtype=np.float64).copy()
    if values.shape != (STATE_DIM,):
        raise ValueError(f"state must have shape ({STATE_DIM},), got {values.shape}")
    values[3] *= -1.0
    values[7] *= -1.0
    values[2:4] = wrap_angle_numpy(values[2:4])
    return values


def policy_features(state: torch.Tensor) -> torch.Tensor:
    """Map (..., 8) chart states to manifold-safe (..., 10) features."""

    if state.shape[-1] != STATE_DIM:
        raise ValueError(f"state must have last dimension {STATE_DIM}")
    direction, direction_rate = pole_direction_and_rate(state)
    return torch.cat((state[..., 0:2], direction, state[..., 4:6], direction_rate), dim=-1)


def pole_direction_and_rate(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the physical two-angle chart to pole direction and tangent rate."""

    theta_x = state[..., 2]
    theta_y = state[..., 3]
    theta_x_dot = state[..., 6]
    theta_y_dot = state[..., 7]
    sin_x, cos_x = theta_x.sin(), theta_x.cos()
    sin_y, cos_y = theta_y.sin(), theta_y.cos()
    direction = torch.stack((sin_x * cos_y, sin_y, cos_x * cos_y), dim=-1)
    direction_rate = torch.stack(
        (
            cos_x * cos_y * theta_x_dot - sin_x * sin_y * theta_y_dot,
            cos_y * theta_y_dot,
            -sin_x * cos_y * theta_x_dot - cos_x * sin_y * theta_y_dot,
        ),
        dim=-1,
    )
    return direction, direction_rate


def manifold_velocity(state: torch.Tensor) -> torch.Tensor:
    """Return cart velocity and pole tangent velocity as (..., 5)."""

    _, direction_rate = pole_direction_and_rate(state)
    return torch.cat((state[..., 4:6], direction_rate), dim=-1)


def dynamics_features(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Build (..., 12) GP inputs from physical state and normalized action."""

    if action.shape[:-1] != state.shape[:-1] or action.shape[-1] != ACTION_DIM:
        raise ValueError("state and action batch dimensions must match")
    return torch.cat((policy_features(state), action), dim=-1)


def integrate_velocity_change(
    state: torch.Tensor,
    velocity_change: torch.Tensor,
    control_dt: float,
) -> torch.Tensor:
    """Apply cart and tangent-velocity increments on the pole sphere."""

    if state.shape[-1] != STATE_DIM or velocity_change.shape != state.shape[:-1] + (5,):
        raise ValueError("expected state (..., 8) and velocity_change (..., 5)")
    direction, direction_rate = pole_direction_and_rate(state)
    next_cart_velocity = state[..., 4:6] + velocity_change[..., 0:2]
    next_direction_rate = direction_rate + velocity_change[..., 2:5]
    next_direction_rate = next_direction_rate - (
        next_direction_rate * direction
    ).sum(-1, keepdim=True) * direction

    average_rate = 0.5 * (direction_rate + next_direction_rate)
    next_direction = direction + float(control_dt) * average_rate
    next_direction = next_direction / torch.linalg.vector_norm(
        next_direction, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    next_direction_rate = next_direction_rate - (
        next_direction_rate * next_direction
    ).sum(-1, keepdim=True) * next_direction

    next_cart_position = state[..., 0:2] + 0.5 * float(control_dt) * (
        state[..., 4:6] + next_cart_velocity
    )
    horizontal_sq = next_direction[..., 0].square() + next_direction[..., 2].square()
    horizontal = horizontal_sq.clamp_min(1e-12).sqrt()
    theta_x = torch.atan2(next_direction[..., 0], next_direction[..., 2])
    theta_y = torch.atan2(next_direction[..., 1], horizontal)
    theta_x_dot = (
        next_direction[..., 2] * next_direction_rate[..., 0]
        - next_direction[..., 0] * next_direction_rate[..., 2]
    ) / horizontal_sq.clamp_min(1e-12)
    theta_y_dot = next_direction_rate[..., 1] / horizontal
    return torch.cat(
        (
            next_cart_position,
            theta_x[..., None],
            theta_y[..., None],
            next_cart_velocity,
            theta_x_dot[..., None],
            theta_y_dot[..., None],
        ),
        dim=-1,
    )
