"""Bounded differentiable policy used both in imagination and MuJoCo."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mc_pilco.state import POLICY_FEATURE_DIM, policy_features


class BoundedMLPPolicy(nn.Module):
    """Small tanh policy with normalized trigonometric state features."""

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (32, 32),
        *,
        action_limit: float = 1.0,
        feature_scale: torch.Tensor | np.ndarray | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if not hidden_sizes or any(size < 1 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values")
        if action_limit <= 0.0:
            raise ValueError("action_limit must be positive")
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.action_limit = float(action_limit)
        layers: list[nn.Module] = []
        previous = POLICY_FEATURE_DIM
        for size in self.hidden_sizes:
            layers.extend((nn.Linear(previous, size, dtype=dtype), nn.Tanh()))
            previous = size
        layers.append(nn.Linear(previous, 2, dtype=dtype))
        self.network = nn.Sequential(*layers)
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(
                [5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 5.0, 10.0, 10.0, 10.0]
                if feature_scale is None
                else feature_scale,
                dtype=dtype,
            ).clone(),
        )
        if self.feature_scale.shape != (POLICY_FEATURE_DIM,):
            raise ValueError(f"feature_scale must have shape ({POLICY_FEATURE_DIM},)")
        self._initialize()

    def _initialize(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.25)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = policy_features(state) / self.feature_scale
        return self.action_limit * torch.tanh(self.network(features))

    @torch.no_grad()
    def control(self, state: np.ndarray) -> np.ndarray:
        parameter = next(self.parameters())
        tensor = torch.as_tensor(state, dtype=parameter.dtype, device=parameter.device)
        if tensor.shape != (8,):
            raise ValueError("state must have shape (8,)")
        return self(tensor).cpu().numpy()


class GatedPriorPolicy(nn.Module):
    """Frozen learned local prior plus a trainable pole-height-gated residual."""

    def __init__(
        self,
        prior: BoundedMLPPolicy,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.prior.requires_grad_(False)
        self.residual = BoundedMLPPolicy(
            prior.hidden_sizes,
            feature_scale=prior.feature_scale.detach().clone(),
            dtype=dtype,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = policy_features(state) / self.residual.feature_scale
        direction_z = policy_features(state)[..., 4:5]
        gate = (0.5 * (1.0 - direction_z)).clamp(0.0, 1.0)
        prior_action = self.prior(state).clamp(-0.999, 0.999)
        prior_latent = torch.atanh(prior_action)
        residual_latent = self.residual.network(features)
        return torch.tanh(prior_latent + gate * residual_latent)

    @torch.no_grad()
    def control(self, state: np.ndarray) -> np.ndarray:
        parameter = next(self.residual.parameters())
        tensor = torch.as_tensor(state, dtype=parameter.dtype, device=parameter.device)
        return self(tensor).cpu().numpy()


class AxisGatedPriorPolicy(nn.Module):
    """Local neural prior with a shared axis-equivariant swing-up residual."""

    def __init__(self, prior: BoundedMLPPolicy, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.prior = prior
        self.prior.requires_grad_(False)
        self.register_buffer(
            "axis_scale", torch.tensor([1.0, 1.0, 1.0, 2.0, 4.0], dtype=dtype)
        )
        self.residual = nn.Sequential(
            nn.Linear(5, 32, dtype=dtype),
            nn.Tanh(),
            nn.Linear(32, 32, dtype=dtype),
            nn.Tanh(),
            nn.Linear(32, 1, dtype=dtype),
        )
        for module in self.residual:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.25)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = policy_features(state)
        direction_z = features[..., 4:5]
        x_axis = torch.stack(
            (
                features[..., 0],
                features[..., 2],
                features[..., 4],
                features[..., 5],
                features[..., 7],
            ),
            dim=-1,
        )
        y_axis = torch.stack(
            (
                features[..., 1],
                features[..., 3],
                features[..., 4],
                features[..., 6],
                features[..., 8],
            ),
            dim=-1,
        )
        axis_input = torch.stack((x_axis, y_axis), dim=-2) / self.axis_scale
        residual_latent = self.residual(axis_input).squeeze(-1)
        gate = (0.5 * (1.0 - direction_z)).clamp(0.0, 1.0)
        prior_action = self.prior(state).clamp(-0.999, 0.999)
        return torch.tanh(torch.atanh(prior_action) + gate * residual_latent)

    @torch.no_grad()
    def control(self, state: np.ndarray) -> np.ndarray:
        parameter = next(self.residual.parameters())
        tensor = torch.as_tensor(state, dtype=parameter.dtype, device=parameter.device)
        return self(tensor).cpu().numpy()
