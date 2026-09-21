"""Common zero-interaction local stabilizing prior for benchmark actors."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from mc_pilco.benchmark_task import observation_tensor
from mc_pilco.policy import BoundedMLPPolicy
from mc_pilco.robust import RobustMCPILCOController
from mc_pilco.state import backend_from_physical


def prior_dataset(
    *,
    seed: int,
    samples: int,
    control_dt: float,
    actuator_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.empty((samples, 8), dtype=np.float32)
    states[:, 0:2] = rng.uniform(-0.5, 0.5, (samples, 2))
    states[:, 2:4] = rng.uniform(-0.20, 0.20, (samples, 2))
    states[:, 4:6] = rng.uniform(-0.5, 0.5, (samples, 2))
    states[:, 6:8] = rng.uniform(-1.0, 1.0, (samples, 2))
    source = RobustMCPILCOController(
        control_dt=control_dt,
        actuator_gain=actuator_gain,
    )
    lqr = source.hybrid.lqr
    actions = np.asarray(
        [lqr.control(backend_from_physical(state)) for state in states], dtype=np.float32
    )
    return states, actions


def fit_actor_prior(
    actor: Callable[[torch.Tensor], torch.Tensor],
    parameters: list[torch.nn.Parameter],
    *,
    seed: int,
    steps: int,
    control_dt: float,
    actuator_gain: float,
    samples: int = 4096,
) -> list[float]:
    states, actions = prior_dataset(
        seed=seed,
        samples=samples,
        control_dt=control_dt,
        actuator_gain=actuator_gain,
    )
    state_tensor = torch.as_tensor(states)
    observation = observation_tensor(state_tensor)
    target = torch.as_tensor(actions)
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    history: list[float] = []
    for _ in range(steps):
        indices = torch.randint(0, samples, (256,), generator=generator)
        prediction = actor(observation[indices])
        loss = (prediction - target[indices]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def initialize_mc_policy(
    policy: BoundedMLPPolicy,
    *,
    seed: int,
    steps: int,
    control_dt: float,
    actuator_gain: float,
) -> list[float]:
    return fit_actor_prior(
        lambda observation: torch.tanh(policy.network(observation)),
        list(policy.parameters()),
        seed=seed,
        steps=steps,
        control_dt=control_dt,
        actuator_gain=actuator_gain,
    )
