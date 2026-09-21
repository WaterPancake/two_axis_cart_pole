"""Transition storage and structured dynamics training targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mc_pilco.state import ACTION_DIM, STATE_DIM, dynamics_features, manifold_velocity


@dataclass(frozen=True)
class TransitionDataset:
    """Aligned physical-state transitions collected at one controller interval."""

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    control_dt: float

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.float64)
        actions = np.asarray(self.actions, dtype=np.float64)
        next_states = np.asarray(self.next_states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError("states must have shape (n, 8)")
        if actions.shape != (states.shape[0], ACTION_DIM):
            raise ValueError("actions must have shape (n, 2)")
        if next_states.shape != states.shape:
            raise ValueError("next_states must have the same shape as states")
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if not all(np.all(np.isfinite(array)) for array in (states, actions, next_states)):
            raise ValueError("transition arrays must be finite")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "next_states", next_states)

    def __len__(self) -> int:
        return self.states.shape[0]

    @classmethod
    def empty(cls, control_dt: float) -> "TransitionDataset":
        return cls(
            np.empty((0, STATE_DIM)),
            np.empty((0, ACTION_DIM)),
            np.empty((0, STATE_DIM)),
            control_dt,
        )

    def append(self, other: "TransitionDataset") -> "TransitionDataset":
        if not np.isclose(self.control_dt, other.control_dt):
            raise ValueError("cannot combine datasets with different controller intervals")
        return TransitionDataset(
            np.concatenate((self.states, other.states), axis=0),
            np.concatenate((self.actions, other.actions), axis=0),
            np.concatenate((self.next_states, other.next_states), axis=0),
            self.control_dt,
        )

    def model_tensors(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        indices: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return GP inputs and five manifold-velocity-increment targets."""

        selected = np.arange(len(self)) if indices is None else np.asarray(indices)
        states = torch.as_tensor(self.states[selected], dtype=dtype, device=device)
        actions = torch.as_tensor(self.actions[selected], dtype=dtype, device=device)
        next_states = torch.as_tensor(self.next_states[selected], dtype=dtype, device=device)
        return dynamics_features(states, actions), manifold_velocity(next_states) - manifold_velocity(
            states
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            states=self.states,
            actions=self.actions,
            next_states=self.next_states,
            control_dt=np.array(self.control_dt),
        )

    @classmethod
    def load(cls, path: Path) -> "TransitionDataset":
        with np.load(path) as payload:
            return cls(
                payload["states"],
                payload["actions"],
                payload["next_states"],
                float(payload["control_dt"]),
            )
