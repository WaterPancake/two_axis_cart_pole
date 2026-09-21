"""Differentiable independent-output exact GP dynamics."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("positive parameter initializers must be positive")
    return math.log(math.expm1(value))


class IndependentGaussianProcessDynamics(nn.Module):
    """Five exact ARD-RBF GPs with differentiable posterior sampling.

    Inputs and targets are standardized from the current transition dataset.
    Outputs remain conditionally independent, matching the common MC-PILCO
    approximation while allowing gradients from sampled rollouts to actions.
    """

    def __init__(
        self,
        input_dim: int = 12,
        output_dim: int = 5,
        *,
        jitter: float = 1e-6,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1 or jitter <= 0.0:
            raise ValueError("dimensions and jitter must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.jitter = float(jitter)
        self.raw_length_scale = nn.Parameter(
            torch.full(
                (output_dim, input_dim), _inverse_softplus(1.0), dtype=dtype
            )
        )
        self.raw_output_scale = nn.Parameter(
            torch.full((output_dim,), _inverse_softplus(1.0), dtype=dtype)
        )
        self.raw_noise = nn.Parameter(
            torch.full((output_dim,), _inverse_softplus(0.08), dtype=dtype)
        )
        self.register_buffer("input_mean", torch.zeros(input_dim, dtype=dtype))
        self.register_buffer("input_std", torch.ones(input_dim, dtype=dtype))
        self.register_buffer("target_mean", torch.zeros(output_dim, dtype=dtype))
        self.register_buffer("target_std", torch.ones(output_dim, dtype=dtype))
        self.register_buffer("train_x", torch.empty((0, input_dim), dtype=dtype))
        self.register_buffer("train_y", torch.empty((output_dim, 0), dtype=dtype))
        self.register_buffer("cached_cholesky", torch.empty((output_dim, 0, 0), dtype=dtype))
        self.register_buffer("cached_alpha", torch.empty((output_dim, 0, 1), dtype=dtype))

    @property
    def length_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_length_scale).clamp_min(1e-4)

    @property
    def output_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_output_scale).clamp_min(1e-5)

    @property
    def noise(self) -> torch.Tensor:
        return F.softplus(self.raw_noise).clamp_min(1e-5)

    def _kernel(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        scaled_a = x_a[None, :, :] / self.length_scale[:, None, :]
        scaled_b = x_b[None, :, :] / self.length_scale[:, None, :]
        squared_distance = (
            scaled_a.square().sum(-1).unsqueeze(-1)
            + scaled_b.square().sum(-1).unsqueeze(-2)
            - 2.0 * scaled_a @ scaled_b.transpose(-1, -2)
        ).clamp_min(0.0)
        return self.output_scale[:, None, None].square() * torch.exp(-0.5 * squared_distance)

    def set_data(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Standardize and retain a fixed GP training set."""

        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(f"inputs must have shape (n, {self.input_dim})")
        if targets.shape != (inputs.shape[0], self.output_dim):
            raise ValueError(f"targets must have shape (n, {self.output_dim})")
        if inputs.shape[0] < 2:
            raise ValueError("at least two training transitions are required")
        parameter = self.raw_length_scale
        inputs = inputs.to(dtype=parameter.dtype, device=parameter.device)
        targets = targets.to(dtype=parameter.dtype, device=parameter.device)
        self.input_mean.copy_(inputs.mean(0))
        self.input_std.copy_(inputs.std(0, unbiased=False).clamp_min(1e-6))
        self.target_mean.copy_(targets.mean(0))
        self.target_std.copy_(targets.std(0, unbiased=False).clamp_min(1e-6))
        self.train_x = ((inputs - self.input_mean) / self.input_std).detach()
        self.train_y = ((targets - self.target_mean) / self.target_std).transpose(0, 1).detach()
        self._refresh_cache()

    def _factor(self) -> torch.Tensor:
        covariance = self._kernel(self.train_x, self.train_x)
        diagonal = self.noise.square() + self.jitter
        identity = torch.eye(
            self.train_x.shape[0], dtype=self.train_x.dtype, device=self.train_x.device
        )
        return torch.linalg.cholesky(covariance + diagonal[:, None, None] * identity)

    def negative_log_marginal_likelihood(self) -> torch.Tensor:
        if self.train_x.shape[0] == 0:
            raise RuntimeError("call set_data before fitting")
        cholesky = self._factor()
        target = self.train_y[..., None]
        alpha = torch.cholesky_solve(target, cholesky)
        data_fit = 0.5 * (target.transpose(-2, -1) @ alpha).flatten()
        log_determinant = torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(-1)
        normalizer = 0.5 * self.train_x.shape[0] * math.log(2.0 * math.pi)
        return (data_fit + log_determinant + normalizer).sum()

    def fit(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        steps: int = 100,
        learning_rate: float = 0.03,
    ) -> list[float]:
        if steps < 0 or learning_rate <= 0.0:
            raise ValueError("steps must be non-negative and learning_rate must be positive")
        self.requires_grad_(True)
        self.set_data(inputs, targets)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        history: list[float] = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = self.negative_log_marginal_likelihood()
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        self._refresh_cache()
        return history

    @torch.no_grad()
    def _refresh_cache(self) -> None:
        if self.train_x.shape[0] == 0:
            return
        self.cached_cholesky = self._factor().detach()
        self.cached_alpha = torch.cholesky_solve(
            self.train_y[..., None], self.cached_cholesky
        ).detach()

    def posterior(
        self,
        query: torch.Tensor,
        *,
        include_observation_noise: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return physical-unit posterior marginal means and variances."""

        if query.ndim != 2 or query.shape[1] != self.input_dim:
            raise ValueError(f"query must have shape (n, {self.input_dim})")
        if self.cached_cholesky.shape[-1] == 0:
            raise RuntimeError("fit or set_data must be called before prediction")
        query = query.to(dtype=self.input_mean.dtype, device=self.input_mean.device)
        normalized_query = (query - self.input_mean) / self.input_std
        cross_covariance = self._kernel(self.train_x, normalized_query)
        mean = (cross_covariance.transpose(-2, -1) @ self.cached_alpha).squeeze(-1)
        solved = torch.linalg.solve_triangular(
            self.cached_cholesky, cross_covariance, upper=False
        )
        variance = self.output_scale[:, None].square() - solved.square().sum(dim=-2)
        if include_observation_noise:
            variance = variance + self.noise[:, None].square()
        mean = mean.transpose(0, 1) * self.target_std + self.target_mean
        variance = variance.transpose(0, 1).clamp_min(1e-12) * self.target_std.square()
        return mean, variance

    def sample(self, query: torch.Tensor, standard_normal: torch.Tensor) -> torch.Tensor:
        """Draw reparameterized independent posterior marginal samples."""

        mean, variance = self.posterior(query)
        if standard_normal.shape != mean.shape:
            raise ValueError(f"standard_normal must have shape {tuple(mean.shape)}")
        return mean + variance.sqrt() * standard_normal

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], *, jitter: float = 1e-6
    ) -> "IndependentGaussianProcessDynamics":
        """Restore parameters and variable-sized posterior cache tensors."""

        length_scales = state_dict["raw_length_scale"]
        model = cls(
            input_dim=length_scales.shape[1],
            output_dim=length_scales.shape[0],
            jitter=jitter,
            dtype=length_scales.dtype,
        )
        for name in ("train_x", "train_y", "cached_cholesky", "cached_alpha"):
            setattr(model, name, state_dict[name].clone())
        model.load_state_dict(state_dict)
        return model
