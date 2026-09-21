import math

import numpy as np
import torch

from mc_pilco.state import (
    backend_from_physical,
    dynamics_features,
    integrate_velocity_change,
    pole_direction_and_rate,
    physical_from_backend,
    policy_features,
)


def test_backend_physical_conversion_round_trip() -> None:
    backend = np.array([0.2, -0.4, 3.4, 0.7, 0.1, 0.2, -0.3, 0.6])
    physical = physical_from_backend(backend)
    restored = backend_from_physical(physical)
    np.testing.assert_allclose(restored, [0.2, -0.4, 3.4 - 2 * math.pi, 0.7, 0.1, 0.2, -0.3, 0.6])


def test_features_and_structured_integration() -> None:
    state = torch.tensor(
        [[1.0, 2.0, 0.2, -0.4, 0.1, 0.2, 0.3, 0.4]],
        dtype=torch.float64,
    )
    action = torch.tensor([[0.5, -0.5]], dtype=torch.float64)
    features = policy_features(state)
    assert features.shape == (1, 10)
    assert dynamics_features(state, action).shape == (1, 12)
    torch.testing.assert_close(
        torch.linalg.vector_norm(features[0, 2:5]), torch.tensor(1.0, dtype=torch.float64)
    )

    next_state = integrate_velocity_change(
        state,
        torch.tensor([[0.1, -0.1, 0.02, -0.01, 0.03]], dtype=torch.float64),
        0.01,
    )
    torch.testing.assert_close(
        next_state[0, 4:6], torch.tensor([0.2, 0.1], dtype=torch.float64)
    )
    torch.testing.assert_close(
        next_state[0, 0:2], torch.tensor([1.0015, 2.0015], dtype=torch.float64)
    )
    direction, direction_rate = pole_direction_and_rate(next_state)
    torch.testing.assert_close(
        torch.linalg.vector_norm(direction, dim=-1), torch.ones(1, dtype=torch.float64)
    )
    torch.testing.assert_close(
        (direction * direction_rate).sum(-1),
        torch.zeros(1, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


def test_manifold_integration_handles_chart_fold() -> None:
    state = torch.tensor(
        [[0.0, 0.0, 0.0, 1.55, 0.0, 0.0, 0.0, 5.0]], dtype=torch.float64
    )
    next_state = integrate_velocity_change(
        state, torch.zeros((1, 5), dtype=torch.float64), 0.01
    )
    assert abs(float(next_state[0, 2])) > 3.0
    assert float(next_state[0, 3]) < math.pi / 2
