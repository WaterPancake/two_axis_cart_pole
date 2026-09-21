import torch

from mc_pilco.gp import IndependentGaussianProcessDynamics


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn((24, 12), generator=generator, dtype=torch.float64)
    y = torch.stack(
        (
            0.2 * x[:, 0] + x[:, 10].sin(),
            -0.1 * x[:, 1] + x[:, 11],
            x[:, 2] * x[:, 10] * 0.1,
            x[:, 4].cos() * 0.2,
            0.05 * x[:, 7] - 0.1 * x[:, 11],
        ),
        dim=1,
    )
    return x, y


def test_gp_posterior_has_query_gradients_and_positive_variance() -> None:
    x, y = _training_data()
    model = IndependentGaussianProcessDynamics()
    history = model.fit(x, y, steps=2, learning_rate=0.02)
    assert len(history) == 2

    query = x[:5].clone().requires_grad_()
    mean, variance = model.posterior(query)
    assert mean.shape == variance.shape == (5, 5)
    assert torch.all(variance > 0.0)
    mean.sum().backward()
    assert query.grad is not None
    assert torch.all(torch.isfinite(query.grad))


def test_reparameterized_samples_are_repeatable() -> None:
    x, y = _training_data()
    model = IndependentGaussianProcessDynamics()
    model.fit(x, y, steps=0)
    noise = torch.ones((3, 5), dtype=torch.float64)
    first = model.sample(x[:3], noise)
    second = model.sample(x[:3], noise)
    torch.testing.assert_close(first, second)
