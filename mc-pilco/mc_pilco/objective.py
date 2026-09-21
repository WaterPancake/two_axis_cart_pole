"""Differentiable MC-PILCO cost and stochastic particle propagation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mc_pilco.gp import IndependentGaussianProcessDynamics
from mc_pilco.policy import BoundedMLPPolicy
from mc_pilco.state import (
    dynamics_features,
    integrate_velocity_change,
    pole_direction_and_rate,
)


@dataclass(frozen=True)
class CostWeights:
    cart_position: float = 0.15
    pole_angle: float = 4.0
    cart_velocity: float = 0.02
    pole_velocity: float = 0.04
    upright_pole_velocity: float = 0.35
    upright_angle_scale: float = 0.5
    action: float = 0.002
    rail_violation: float = 200.0
    terminal: float = 4.0
    discount: float = 0.997


def swingup_cost(
    state: torch.Tensor,
    action: torch.Tensor,
    weights: CostWeights | None = None,
    *,
    cart_limit: float = 4.8,
) -> torch.Tensor:
    """Return one cost per batch item; lower is better."""

    w = weights or CostWeights()
    rail_violation = (state[..., 0:2].abs() - cart_limit).clamp_min(0.0)
    direction, direction_rate = pole_direction_and_rate(state)
    pole_error = (1.0 - direction[..., 2]).clamp_min(0.0)
    gate = torch.exp(-pole_error / max(w.upright_angle_scale, 1e-6) ** 2)
    pole_rate_sq = direction_rate.square().sum(-1)
    return (
        w.cart_position * state[..., 0:2].square().sum(-1)
        + w.pole_angle * pole_error
        + w.cart_velocity * state[..., 4:6].square().sum(-1)
        + w.pole_velocity * pole_rate_sq
        + w.upright_pole_velocity * gate * pole_rate_sq
        + w.action * action.square().sum(-1)
        + w.rail_violation * rail_violation.square().sum(-1)
    )


def particle_rollout(
    model: IndependentGaussianProcessDynamics,
    policy: BoundedMLPPolicy,
    initial_states: torch.Tensor,
    noise: torch.Tensor,
    *,
    control_dt: float,
    weights: CostWeights | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate particles with reparameterized GP samples.

    Returns the scalar mean objective and a `(horizon + 1, particles, 8)`
    state trace. Supplying noise explicitly enables deterministic gradient tests
    and common-random-number policy updates.
    """

    if initial_states.ndim != 2 or initial_states.shape[1] != 8:
        raise ValueError("initial_states must have shape (particles, 8)")
    if noise.ndim != 3 or noise.shape[1:] != (initial_states.shape[0], model.output_dim):
        raise ValueError(
            f"noise must have shape (horizon, particles, {model.output_dim})"
        )
    w = weights or CostWeights()
    state = initial_states
    states = [state]
    total = torch.zeros(initial_states.shape[0], dtype=state.dtype, device=state.device)
    discount = 1.0
    for step in range(noise.shape[0]):
        action = policy(state)
        total = total + discount * swingup_cost(state, action, w)
        velocity_change = model.sample(dynamics_features(state, action), noise[step])
        state = integrate_velocity_change(state, velocity_change, control_dt)
        states.append(state)
        discount *= w.discount
    terminal_action = torch.zeros((*state.shape[:-1], 2), dtype=state.dtype, device=state.device)
    total = total + discount * w.terminal * swingup_cost(state, terminal_action, w)
    return total.mean(), torch.stack(states)
