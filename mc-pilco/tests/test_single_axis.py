import numpy as np
import torch

from mc_pilco.single_axis_task import (
    SingleAxisPolicy,
    SingleAxisSimulator,
    dynamics_features,
    embed_state,
    integrate_velocity_change,
    project_state,
    state_features,
    task_cost,
)


def test_single_axis_embedding_and_planar_dynamics() -> None:
    state = np.array([0.2, 2.8, -0.1, 0.4])
    np.testing.assert_allclose(project_state(embed_state(state)), state)
    simulator = SingleAxisSimulator()
    simulator.set_state(state)
    next_state, applied, terminated = simulator.step(np.array([0.3]))
    assert next_state.shape == (4,)
    np.testing.assert_allclose(applied, [0.3])
    assert not terminated
    assert simulator.planar_leakage() < 1e-12


def test_single_axis_features_and_cost_respect_reflection() -> None:
    state = np.array([0.4, 1.2, -0.3, 0.7])
    reflected = -state
    reflected[1] = -state[1]
    action = np.array([0.25])
    assert np.isclose(task_cost(state, action), task_cost(reflected, -action))
    wrapped = torch.as_tensor(
        np.stack((state, state + np.array([0.0, 2 * np.pi, 0.0, 0.0])))
    )
    torch.testing.assert_close(state_features(wrapped)[0], state_features(wrapped)[1])


def test_single_axis_policy_and_integrator_are_differentiable() -> None:
    policy = SingleAxisPolicy()
    state = torch.tensor([[0.2, 0.5, -0.1, 0.3]], requires_grad=True)
    reflected = -state.detach()
    reflected[:, 1] = -state.detach()[:, 1]
    action = policy(state)
    torch.testing.assert_close(action, -policy(reflected), atol=1e-6, rtol=0)
    model_input = dynamics_features(state, action)
    next_state = integrate_velocity_change(state, model_input[:, -2:], 0.02)
    next_state.square().sum().backward()
    assert state.grad is not None
    assert torch.all(torch.isfinite(state.grad))
