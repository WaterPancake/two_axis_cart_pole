"""PILCO-inspired learned controller components.

This module implements the small, dependency-light pieces needed for a
model-based policy-search loop:

1. collect transitions from MuJoCo,
2. fit a probabilistic Gaussian-process dynamics model,
3. optimize a squashed RBF policy inside the learned model, and
4. save/load the resulting controller artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

DEFAULT_POLICY_FEATURE_SCALE = np.array(
    [5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 10.0, 10.0], dtype=float
)


@dataclass(frozen=True)
class SwingUpCostWeights:
    """Weights for model-rollout policy optimization."""

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


@dataclass(frozen=True)
class PolicyOptimizationResult:
    """Summary returned by CEM policy optimization."""

    best_cost: float
    mean_cost_history: tuple[float, ...]
    best_cost_history: tuple[float, ...]


@dataclass(frozen=True)
class PilcoArtifact:
    """Loaded controller artifact."""

    policy: "RBFPolicy"
    model: "GaussianProcessDynamicsModel | None"
    metadata: dict[str, Any]


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians into [-pi, pi]."""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def physical_state_from_mujoco(
    state: np.ndarray, mujoco_y_axis: bool = True
) -> np.ndarray:
    """Convert MuJoCo qpos/qvel state into the physical convention used by policies."""

    values = np.asarray(state, dtype=float).copy()
    if values.shape != (8,):
        raise ValueError(f"state must have shape (8,), got {values.shape}")

    values[2:4] = wrap_to_pi(values[2:4])
    if mujoco_y_axis:
        values[3] *= -1.0
        values[7] *= -1.0
    return values


def mujoco_state_from_physical(
    state: np.ndarray, mujoco_y_axis: bool = True
) -> np.ndarray:
    """Convert a physical-convention state back to MuJoCo hinge coordinates."""

    values = np.asarray(state, dtype=float).copy()
    if values.shape != (8,):
        raise ValueError(f"state must have shape (8,), got {values.shape}")

    if mujoco_y_axis:
        values[3] *= -1.0
        values[7] *= -1.0
    values[2:4] = wrap_to_pi(values[2:4])
    return values


def policy_features(physical_state: np.ndarray) -> np.ndarray:
    """Return continuous policy/model features with sine-cosine angle encoding."""

    state = np.asarray(physical_state, dtype=float)
    if state.shape != (8,):
        raise ValueError(f"physical_state must have shape (8,), got {state.shape}")

    theta_x, theta_y = state[2:4]
    return np.array(
        [
            state[0],
            state[1],
            np.sin(theta_x),
            np.cos(theta_x),
            np.sin(theta_y),
            np.cos(theta_y),
            state[4],
            state[5],
            state[6],
            state[7],
        ],
        dtype=float,
    )


def dynamics_features(physical_state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Features for one-step dynamics: policy features plus the two controls."""

    act = np.asarray(action, dtype=float)
    if act.shape != (2,):
        raise ValueError(f"action must have shape (2,), got {act.shape}")
    return np.concatenate([policy_features(physical_state), act])


def policy_features_batch(physical_states: np.ndarray) -> np.ndarray:
    """Vectorized :func:`policy_features` for a ``(..., 8)`` state array."""

    states = np.asarray(physical_states, dtype=float)
    if states.shape[-1] != 8:
        raise ValueError(
            f"physical_states must have last dimension 8, got {states.shape}"
        )

    theta_x = states[..., 2]
    theta_y = states[..., 3]
    return np.stack(
        [
            states[..., 0],
            states[..., 1],
            np.sin(theta_x),
            np.cos(theta_x),
            np.sin(theta_y),
            np.cos(theta_y),
            states[..., 4],
            states[..., 5],
            states[..., 6],
            states[..., 7],
        ],
        axis=-1,
    )


def apply_state_delta_batch(states: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Vectorized :func:`apply_state_delta` for ``(..., 8)`` arrays."""

    nxt = np.asarray(states, dtype=float) + np.asarray(deltas, dtype=float)
    nxt[..., 2:4] = wrap_to_pi(nxt[..., 2:4])
    return nxt


def state_delta(next_state: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Delta target for dynamics learning, with wrapped angle differences."""

    nxt = np.asarray(next_state, dtype=float)
    cur = np.asarray(state, dtype=float)
    delta = nxt - cur
    delta[2:4] = wrap_to_pi(nxt[2:4] - cur[2:4])
    return delta


def apply_state_delta(state: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a learned delta and keep angles on the principal branch."""

    nxt = np.asarray(state, dtype=float) + np.asarray(delta, dtype=float)
    nxt[2:4] = wrap_to_pi(nxt[2:4])
    return nxt


def swingup_cost(
    physical_state: np.ndarray,
    action: np.ndarray,
    weights: SwingUpCostWeights | None = None,
    cart_limit: float = 4.8,
) -> float:
    """Dense cost for upright, centered, low-energy behavior."""

    w = weights or SwingUpCostWeights()
    state = np.asarray(physical_state, dtype=float)
    act = np.asarray(action, dtype=float)

    rail_violation = np.maximum(np.abs(state[0:2]) - cart_limit, 0.0)
    angle_norm = float(np.linalg.norm(state[2:4]))
    upright_scale = max(float(w.upright_angle_scale), 1e-6)
    upright_gate = float(np.exp(-0.5 * (angle_norm / upright_scale) ** 2))
    return float(
        w.cart_position * np.sum(state[0:2] ** 2)
        + w.pole_angle * np.sum(1.0 - np.cos(state[2:4]))
        + w.cart_velocity * np.sum(state[4:6] ** 2)
        + w.pole_velocity * np.sum(state[6:8] ** 2)
        + w.upright_pole_velocity * upright_gate * np.sum(state[6:8] ** 2)
        + w.action * np.sum(act**2)
        + w.rail_violation * np.sum(rail_violation**2)
    )


def swingup_cost_batch(
    physical_states: np.ndarray,
    actions: np.ndarray,
    weights: SwingUpCostWeights | None = None,
    cart_limit: float = 4.8,
) -> np.ndarray:
    """Vectorized :func:`swingup_cost` for ``(..., 8)`` states and ``(..., 2)`` actions."""

    w = weights or SwingUpCostWeights()
    states = np.asarray(physical_states, dtype=float)
    acts = np.asarray(actions, dtype=float)

    rail_violation = np.maximum(np.abs(states[..., 0:2]) - cart_limit, 0.0)
    angle_norm = np.linalg.norm(states[..., 2:4], axis=-1)
    upright_scale = max(float(w.upright_angle_scale), 1e-6)
    upright_gate = np.exp(-0.5 * (angle_norm / upright_scale) ** 2)
    pole_velocity_sq = np.sum(states[..., 6:8] ** 2, axis=-1)
    return (
        w.cart_position * np.sum(states[..., 0:2] ** 2, axis=-1)
        + w.pole_angle * np.sum(1.0 - np.cos(states[..., 2:4]), axis=-1)
        + w.cart_velocity * np.sum(states[..., 4:6] ** 2, axis=-1)
        + w.pole_velocity * pole_velocity_sq
        + w.upright_pole_velocity * upright_gate * pole_velocity_sq
        + w.action * np.sum(acts**2, axis=-1)
        + w.rail_violation * np.sum(rail_violation**2, axis=-1)
    )


class RBFPolicy:
    """Squashed radial-basis policy used by the PILCO-style learner."""

    def __init__(
        self,
        centers: np.ndarray,
        weights: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        length_scale: float | np.ndarray = 1.0,
        action_limit: float = 1.0,
        feature_scale: np.ndarray = DEFAULT_POLICY_FEATURE_SCALE,
    ) -> None:
        self.centers = np.asarray(centers, dtype=float)
        if self.centers.ndim != 2 or self.centers.shape[1] != 10:
            raise ValueError("centers must have shape (n_centers, 10)")

        self.weights = (
            np.zeros((self.centers.shape[0], 2), dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        self.bias = (
            np.zeros(2, dtype=float) if bias is None else np.asarray(bias, dtype=float)
        )
        self.length_scale = np.asarray(length_scale, dtype=float)
        self.action_limit = float(action_limit)
        self.feature_scale = np.asarray(feature_scale, dtype=float)

        if self.weights.shape != (self.centers.shape[0], 2):
            raise ValueError("weights must have shape (n_centers, 2)")
        if self.bias.shape != (2,):
            raise ValueError("bias must have shape (2,)")
        if np.any(self.feature_scale <= 0.0):
            raise ValueError("feature_scale entries must be positive")
        if np.any(self.length_scale <= 0.0):
            raise ValueError("length_scale entries must be positive")
        if self.action_limit <= 0.0:
            raise ValueError("action_limit must be positive")

    @classmethod
    def from_states(
        cls,
        states: np.ndarray,
        num_centers: int = 12,
        rng: np.random.Generator | None = None,
        length_scale: float = 1.25,
        action_limit: float = 1.0,
    ) -> "RBFPolicy":
        """Initialize centers by sampling normalized features from collected states."""

        if num_centers < 1:
            raise ValueError("num_centers must be at least one")
        rng = rng or np.random.default_rng()
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != 8:
            raise ValueError("states must have shape (n_samples, 8)")
        if len(states) == 0:
            raise ValueError("at least one state is required")

        features = np.vstack([policy_features(state) for state in states])
        features = features / DEFAULT_POLICY_FEATURE_SCALE
        if len(features) >= num_centers:
            indices = rng.choice(len(features), size=num_centers, replace=False)
        else:
            indices = rng.choice(len(features), size=num_centers, replace=True)
        return cls(
            centers=features[indices],
            length_scale=length_scale,
            action_limit=action_limit,
        )

    @property
    def num_parameters(self) -> int:
        return int(self.weights.size + self.bias.size)

    def parameters(self) -> np.ndarray:
        return np.concatenate([self.weights.ravel(), self.bias])

    def with_parameters(self, parameters: np.ndarray) -> "RBFPolicy":
        values = np.asarray(parameters, dtype=float)
        if values.shape != (self.num_parameters,):
            raise ValueError(f"parameters must have shape ({self.num_parameters},)")

        split = self.weights.size
        weights = values[:split].reshape(self.weights.shape)
        bias = values[split:]
        return RBFPolicy(
            centers=self.centers.copy(),
            weights=weights.copy(),
            bias=bias.copy(),
            length_scale=self.length_scale.copy(),
            action_limit=self.action_limit,
            feature_scale=self.feature_scale.copy(),
        )

    def _basis_from_features(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.shape != (10,):
            raise ValueError(f"features must have shape (10,), got {values.shape}")
        scaled_features = values / self.feature_scale
        diff = scaled_features[None, :] - self.centers
        return np.exp(-0.5 * np.sum((diff / self.length_scale) ** 2, axis=1))

    def _basis(self, physical_state: np.ndarray) -> np.ndarray:
        return self._basis_from_features(policy_features(physical_state))

    def control_features(self, features: np.ndarray) -> np.ndarray:
        """Act from a 10D trig observation, such as ``TwoAxisCartPoleEnv`` emits."""

        basis = self._basis_from_features(features)
        return self.action_limit * np.tanh(basis @ self.weights + self.bias)

    def control_physical(self, physical_state: np.ndarray) -> np.ndarray:
        return self.control_features(policy_features(physical_state))

    def control_physical_batch(self, physical_states: np.ndarray) -> np.ndarray:
        """Vectorized :meth:`control_physical` for a ``(..., 8)`` state array."""

        scaled = policy_features_batch(physical_states) / self.feature_scale
        diff = scaled[..., None, :] - self.centers
        basis = np.exp(-0.5 * np.sum((diff / self.length_scale) ** 2, axis=-1))
        return self.action_limit * np.tanh(basis @ self.weights + self.bias)

    def control(
        self, mujoco_state: np.ndarray, mujoco_y_axis: bool = True
    ) -> np.ndarray:
        """Controller-compatible method accepting the raw MuJoCo state vector."""

        return self.control_physical(
            physical_state_from_mujoco(mujoco_state, mujoco_y_axis)
        )

class GaussianProcessDynamicsModel:
    """Independent-output GP for one-step state deltas."""

    def __init__(
        self,
        length_scale: float = 1.5,
        noise: float = 1e-5,
        max_points: int = 700,
    ) -> None:
        if length_scale <= 0.0:
            raise ValueError("length_scale must be positive")
        if noise <= 0.0:
            raise ValueError("noise must be positive")
        if max_points < 1:
            raise ValueError("max_points must be positive")

        self.length_scale = float(length_scale)
        self.noise = float(noise)
        self.max_points = int(max_points)
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        self.x_train: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.cholesky: tuple[np.ndarray, bool] | None = None

    @staticmethod
    def _kernel(x_a: np.ndarray, x_b: np.ndarray, length_scale: float) -> np.ndarray:
        diff = x_a[:, None, :] - x_b[None, :, :]
        sqdist = np.sum(diff**2, axis=2)
        return np.exp(-0.5 * sqdist / (length_scale**2))

    @property
    def is_fit(self) -> bool:
        return (
            self.x_train is not None
            and self.alpha is not None
            and self.cholesky is not None
        )

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> "GaussianProcessDynamicsModel":
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        next_states = np.asarray(next_states, dtype=float)
        if states.ndim != 2 or states.shape[1] != 8:
            raise ValueError("states must have shape (n_samples, 8)")
        if actions.shape != (states.shape[0], 2):
            raise ValueError("actions must have shape (n_samples, 2)")
        if next_states.shape != states.shape:
            raise ValueError("next_states must have the same shape as states")
        if len(states) < 2:
            raise ValueError("at least two transitions are required")

        rng = rng or np.random.default_rng()
        x = np.vstack(
            [dynamics_features(state, action) for state, action in zip(states, actions)]
        )
        y = np.vstack(
            [state_delta(nxt, state) for state, nxt in zip(states, next_states)]
        )

        if len(x) > self.max_points:
            indices = rng.choice(len(x), size=self.max_points, replace=False)
            x = x[indices]
            y = y[indices]

        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + 1e-8
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0) + 1e-8

        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        kernel = self._kernel(x_norm, x_norm, self.length_scale)
        kernel.flat[:: kernel.shape[0] + 1] += self.noise
        self.cholesky = scipy.linalg.cho_factor(kernel, lower=True, check_finite=False)
        self.alpha = scipy.linalg.cho_solve(self.cholesky, y_norm, check_finite=False)
        self.x_train = x_norm
        return self

    def predict_delta(
        self,
        physical_state: np.ndarray,
        action: np.ndarray,
        return_variance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not self.is_fit:
            raise RuntimeError(
                "GaussianProcessDynamicsModel must be fit before prediction"
            )
        assert self.x_mean is not None
        assert self.x_std is not None
        assert self.y_mean is not None
        assert self.y_std is not None
        assert self.x_train is not None
        assert self.alpha is not None
        assert self.cholesky is not None

        x = dynamics_features(physical_state, action)[None, :]
        x_norm = (x - self.x_mean) / self.x_std
        k_star = self._kernel(x_norm, self.x_train, self.length_scale)
        mean_norm = k_star @ self.alpha
        mean = mean_norm[0] * self.y_std + self.y_mean

        if not return_variance:
            return mean

        v = scipy.linalg.solve_triangular(
            self.cholesky[0], k_star.T, lower=self.cholesky[1], check_finite=False
        )
        variance_norm = max(0.0, float(1.0 - np.sum(v**2)))
        variance = variance_norm * (self.y_std**2)
        return mean, variance

    def predict_delta_batch(
        self,
        physical_states: np.ndarray,
        actions: np.ndarray,
        return_variance: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Vectorized :meth:`predict_delta` for ``(..., 8)`` states, ``(..., 2)`` actions."""

        if not self.is_fit:
            raise RuntimeError(
                "GaussianProcessDynamicsModel must be fit before prediction"
            )
        assert self.x_mean is not None
        assert self.x_std is not None
        assert self.y_mean is not None
        assert self.y_std is not None
        assert self.x_train is not None
        assert self.alpha is not None
        assert self.cholesky is not None

        states = np.asarray(physical_states, dtype=float)
        acts = np.asarray(actions, dtype=float)
        batch_shape = states.shape[:-1]
        features = np.concatenate(
            [policy_features_batch(states), acts], axis=-1
        ).reshape(-1, 12)
        x_norm = (features - self.x_mean) / self.x_std

        sqdist = (
            np.sum(x_norm**2, axis=1)[:, None]
            - 2.0 * x_norm @ self.x_train.T
            + np.sum(self.x_train**2, axis=1)
        )
        k_star = np.exp(-0.5 * sqdist / (self.length_scale**2))
        mean = (k_star @ self.alpha) * self.y_std + self.y_mean
        mean = mean.reshape(*batch_shape, 8)

        if not return_variance:
            return mean

        v = scipy.linalg.solve_triangular(
            self.cholesky[0], k_star.T, lower=self.cholesky[1], check_finite=False
        )
        variance_norm = np.maximum(0.0, 1.0 - np.sum(v**2, axis=0))
        variance = variance_norm[:, None] * (self.y_std**2)
        return mean, variance.reshape(*batch_shape, 8)

    def to_arrays(self, prefix: str = "model_") -> dict[str, np.ndarray]:
        if not self.is_fit:
            raise RuntimeError("cannot serialize an unfitted dynamics model")
        assert self.x_mean is not None
        assert self.x_std is not None
        assert self.y_mean is not None
        assert self.y_std is not None
        assert self.x_train is not None
        assert self.alpha is not None
        assert self.cholesky is not None

        return {
            f"{prefix}length_scale": np.array(self.length_scale),
            f"{prefix}noise": np.array(self.noise),
            f"{prefix}max_points": np.array(self.max_points),
            f"{prefix}x_mean": self.x_mean,
            f"{prefix}x_std": self.x_std,
            f"{prefix}y_mean": self.y_mean,
            f"{prefix}y_std": self.y_std,
            f"{prefix}x_train": self.x_train,
            f"{prefix}alpha": self.alpha,
            f"{prefix}chol_matrix": self.cholesky[0],
            f"{prefix}chol_lower": np.array(self.cholesky[1]),
        }

    @classmethod
    def from_arrays(
        cls,
        arrays: dict[str, np.ndarray],
        prefix: str = "model_",
    ) -> "GaussianProcessDynamicsModel":
        model = cls(
            length_scale=float(arrays[f"{prefix}length_scale"]),
            noise=float(arrays[f"{prefix}noise"]),
            max_points=int(arrays[f"{prefix}max_points"]),
        )
        model.x_mean = np.asarray(arrays[f"{prefix}x_mean"], dtype=float)
        model.x_std = np.asarray(arrays[f"{prefix}x_std"], dtype=float)
        model.y_mean = np.asarray(arrays[f"{prefix}y_mean"], dtype=float)
        model.y_std = np.asarray(arrays[f"{prefix}y_std"], dtype=float)
        model.x_train = np.asarray(arrays[f"{prefix}x_train"], dtype=float)
        model.alpha = np.asarray(arrays[f"{prefix}alpha"], dtype=float)
        model.cholesky = (
            np.asarray(arrays[f"{prefix}chol_matrix"], dtype=float),
            bool(arrays[f"{prefix}chol_lower"]),
        )
        return model


class PilcoController:
    """Controller wrapper around a learned PILCO-style RBF policy."""

    def __init__(
        self,
        policy: RBFPolicy,
        mujoco_y_axis: bool = True,
    ) -> None:
        self.policy = policy
        self.mujoco_y_axis = bool(mujoco_y_axis)

    @classmethod
    def from_artifact(cls, path: str | Path, **kwargs: Any) -> "PilcoController":
        artifact = load_pilco_artifact(path)
        return cls(artifact.policy, **kwargs)

    def reset(self) -> None:
        """Reset controller state; the basic policy is stateless."""

    def control(self, state: np.ndarray) -> np.ndarray:
        physical = physical_state_from_mujoco(state, self.mujoco_y_axis)
        return self.policy.control_physical(physical)


def simulate_model_rollout(
    model: GaussianProcessDynamicsModel,
    policy: RBFPolicy,
    start_state: np.ndarray,
    horizon: int,
    cost_weights: SwingUpCostWeights | None = None,
    uncertainty_weight: float = 0.0,
) -> float:
    """Return discounted cost for one policy rollout inside the learned model."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    weights = cost_weights or SwingUpCostWeights()
    state = np.asarray(start_state, dtype=float).copy()
    total = 0.0
    discount = 1.0

    for _ in range(horizon):
        action = policy.control_physical(state)
        delta, variance = model.predict_delta(state, action, return_variance=True)
        total += discount * (
            swingup_cost(state, action, weights)
            + uncertainty_weight * float(np.sum(variance))
        )
        state = apply_state_delta(state, delta)
        discount *= weights.discount

    total += discount * weights.terminal * swingup_cost(state, np.zeros(2), weights)
    return float(total)


def _population_rollout_costs(
    model: GaussianProcessDynamicsModel,
    policy: RBFPolicy,
    parameter_population: np.ndarray,
    start_states: np.ndarray,
    horizon: int,
    weights: SwingUpCostWeights,
    uncertainty_weight: float,
) -> np.ndarray:
    """Discounted model-rollout cost for a whole CEM population at once.

    Candidates share the policy's centers/length scale, so the basis tensor is
    computed once per step for the full ``(population, starts)`` state block.
    Returns the per-candidate cost averaged over start states.
    """

    population = len(parameter_population)
    num_centers = policy.centers.shape[0]
    split = policy.weights.size
    pop_weights = parameter_population[:, :split].reshape(population, num_centers, 2)
    pop_bias = parameter_population[:, split:]

    states = np.broadcast_to(start_states, (population, len(start_states), 8)).copy()
    total = np.zeros(population)
    discount = 1.0

    for _ in range(horizon):
        scaled = policy_features_batch(states) / policy.feature_scale
        diff = scaled[:, :, None, :] - policy.centers
        basis = np.exp(-0.5 * np.sum((diff / policy.length_scale) ** 2, axis=-1))
        raw = np.einsum("psk,pko->pso", basis, pop_weights) + pop_bias[:, None, :]
        actions = policy.action_limit * np.tanh(raw)

        step_costs = swingup_cost_batch(states, actions, weights)
        if uncertainty_weight > 0.0:
            deltas, variances = model.predict_delta_batch(
                states, actions, return_variance=True
            )
            step_costs = step_costs + uncertainty_weight * np.sum(variances, axis=-1)
        else:
            deltas = model.predict_delta_batch(states, actions)

        total += discount * step_costs.mean(axis=1)
        states = apply_state_delta_batch(states, deltas)
        discount *= weights.discount

    terminal_costs = swingup_cost_batch(states, np.zeros_like(states[..., :2]), weights)
    total += discount * weights.terminal * terminal_costs.mean(axis=1)
    return total


def optimize_policy_cem(
    model: GaussianProcessDynamicsModel,
    policy: RBFPolicy,
    start_states: np.ndarray,
    horizon: int = 200,
    iterations: int = 8,
    population: int = 48,
    elite_fraction: float = 0.2,
    initial_std: float = 0.5,
    min_std: float = 0.02,
    uncertainty_weight: float = 0.0,
    rng: np.random.Generator | None = None,
    cost_weights: SwingUpCostWeights | None = None,
) -> tuple[RBFPolicy, PolicyOptimizationResult]:
    """Optimize policy parameters with the cross-entropy method."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if population < 2:
        raise ValueError("population must be at least two")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_fraction must be in (0, 1]")
    if initial_std <= 0.0 or min_std <= 0.0:
        raise ValueError("initial_std and min_std must be positive")

    starts = np.asarray(start_states, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != 8:
        raise ValueError("start_states must have shape (n_starts, 8)")

    rng = rng or np.random.default_rng()
    weights = cost_weights or SwingUpCostWeights()
    mean = policy.parameters()
    std = np.full_like(mean, float(initial_std))
    elite_count = max(1, int(round(population * elite_fraction)))
    best_params = mean.copy()
    best_cost = np.inf
    mean_history: list[float] = []
    best_history: list[float] = []

    for _ in range(iterations):
        samples = rng.normal(mean, std, size=(population, mean.size))
        samples[0] = best_params
        costs = _population_rollout_costs(
            model, policy, samples, starts, horizon, weights, uncertainty_weight
        )
        order = np.argsort(costs)
        elites = samples[order[:elite_count]]
        elite_costs = costs[order[:elite_count]]
        if float(elite_costs[0]) < best_cost:
            best_cost = float(elite_costs[0])
            best_params = elites[0].copy()
        mean = elites.mean(axis=0)
        std = np.maximum(elites.std(axis=0), min_std)
        mean_history.append(float(np.mean(costs)))
        best_history.append(best_cost)

    return policy.with_parameters(best_params), PolicyOptimizationResult(
        best_cost=best_cost,
        mean_cost_history=tuple(mean_history),
        best_cost_history=tuple(best_history),
    )


def save_pilco_artifact(
    path: str | Path,
    policy: RBFPolicy,
    model: GaussianProcessDynamicsModel | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save policy, optional dynamics model, and JSON metadata to an npz file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "policy_centers": policy.centers,
        "policy_weights": policy.weights,
        "policy_bias": policy.bias,
        "policy_length_scale": policy.length_scale,
        "policy_action_limit": np.array(policy.action_limit),
        "policy_feature_scale": policy.feature_scale,
        "has_model": np.array(model is not None and model.is_fit),
        "metadata_json": np.array(json.dumps(metadata or {}, sort_keys=True)),
    }
    if model is not None and model.is_fit:
        arrays.update(model.to_arrays())
    np.savez_compressed(output, **arrays)


def load_pilco_artifact(path: str | Path) -> PilcoArtifact:
    """Load a PILCO artifact written by :func:`save_pilco_artifact`."""

    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    policy = RBFPolicy(
        centers=arrays["policy_centers"],
        weights=arrays["policy_weights"],
        bias=arrays["policy_bias"],
        length_scale=arrays["policy_length_scale"],
        action_limit=float(arrays["policy_action_limit"]),
        feature_scale=arrays["policy_feature_scale"],
    )
    model = None
    if bool(arrays["has_model"]):
        model = GaussianProcessDynamicsModel.from_arrays(arrays)
    metadata = json.loads(str(arrays["metadata_json"].item()))
    return PilcoArtifact(policy=policy, model=model, metadata=metadata)
