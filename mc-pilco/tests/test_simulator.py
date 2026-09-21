import numpy as np

from mc_pilco.simulator import CartPoleSimulator


def test_simulator_uses_expected_timing_and_clips_action() -> None:
    simulator = CartPoleSimulator(frame_skip=5)
    assert np.isclose(simulator.physics_dt, 0.002)
    assert np.isclose(simulator.control_dt, 0.01)
    initial = np.array([0.1, -0.1, 0.2, -0.2, 0.0, 0.0, 0.1, -0.1])
    restored = simulator.set_state(initial)
    np.testing.assert_allclose(restored, initial, atol=1e-10)
    state, applied, terminated = simulator.step(np.array([2.0, -2.0]))
    np.testing.assert_allclose(applied, [1.0, -1.0])
    assert state.shape == (8,)
    assert not terminated
