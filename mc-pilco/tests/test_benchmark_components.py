import numpy as np
import torch

from mc_pilco.benchmark_task import sample_benchmark_state
from mc_pilco.policy import AxisGatedPriorPolicy, BoundedMLPPolicy
from mc_pilco.ppo_baseline import _blend_residual_action


def test_basic_benchmark_has_stabilization_and_x_swing_strata() -> None:
    rng = np.random.default_rng(4)
    stabilize, _ = sample_benchmark_state(rng, "stabilize")
    swing, _ = sample_benchmark_state(rng, "swingup")
    assert np.linalg.norm(stabilize[2:4]) < 0.2
    assert 2.72 <= swing[2] <= 2.88
    assert abs(swing[3]) <= 0.03
    assert 0.4 <= swing[6] <= 0.6


def test_axis_residual_is_equivariant_under_xy_swap() -> None:
    prior = BoundedMLPPolicy(dtype=torch.float32)
    for parameter in prior.parameters():
        torch.nn.init.zeros_(parameter)
    policy = AxisGatedPriorPolicy(prior)
    state = torch.tensor(
        [[0.2, -0.1, 2.7, 0.2, 0.3, -0.2, 0.5, -0.4]], dtype=torch.float32
    )
    swapped = state[:, [1, 0, 3, 2, 5, 4, 7, 6]]
    original_residual = policy(state) - prior(state)
    swapped_residual = policy(swapped) - prior(swapped)
    torch.testing.assert_close(original_residual[:, 0], swapped_residual[:, 1], atol=2e-3, rtol=0)


def test_ppo_residual_has_no_authority_upright_and_full_authority_downward() -> None:
    prior = BoundedMLPPolicy(dtype=torch.float32)
    for parameter in prior.parameters():
        torch.nn.init.zeros_(parameter)
    residual = np.array([1.0, -1.0])
    upright = np.zeros(8)
    downward = np.zeros(8)
    downward[2] = np.pi
    np.testing.assert_allclose(
        _blend_residual_action(upright, prior, residual, 20.0), [0.0, 0.0], atol=1e-7
    )
    action = _blend_residual_action(downward, prior, residual, 20.0)
    assert action[0] > 0.999
    assert action[1] < -0.999
