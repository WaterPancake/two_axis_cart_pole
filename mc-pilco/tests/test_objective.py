import torch

from mc_pilco.gp import IndependentGaussianProcessDynamics
from mc_pilco.objective import CostWeights, particle_rollout
from mc_pilco.policy import BoundedMLPPolicy


def test_particle_rollout_backpropagates_to_policy() -> None:
    generator = torch.Generator().manual_seed(11)
    train_x = torch.randn((18, 12), generator=generator, dtype=torch.float64)
    train_y = 0.02 * torch.randn((18, 5), generator=generator, dtype=torch.float64)
    model = IndependentGaussianProcessDynamics()
    model.fit(train_x, train_y, steps=0)
    model.requires_grad_(False)
    policy = BoundedMLPPolicy(hidden_sizes=(8,))
    initial = 0.1 * torch.randn((6, 8), generator=generator, dtype=torch.float64)
    noise = torch.randn((4, 6, 5), generator=generator, dtype=torch.float64)

    loss, trace = particle_rollout(
        model,
        policy,
        initial,
        noise,
        control_dt=0.01,
        weights=CostWeights(action=0.0),
    )
    assert trace.shape == (5, 6, 8)
    loss.backward()
    gradients = [parameter.grad for parameter in policy.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.all(torch.isfinite(gradient)) for gradient in gradients if gradient is not None)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0.0
