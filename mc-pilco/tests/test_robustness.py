import numpy as np

from mc_pilco.acceptance import run_acceptance_suite
from mc_pilco.interactive import WASDDisturbance


def test_wasd_disturbance_mapping_and_decay() -> None:
    disturbance = WASDDisturbance(step=0.5, decay=0.5)
    disturbance.on_key(ord("w"))
    disturbance.on_key(ord("D"))
    np.testing.assert_allclose(disturbance.command(), [0.5, 0.5])
    np.testing.assert_allclose(disturbance.command(), [0.25, 0.25])
    disturbance.on_key(ord(" "))
    np.testing.assert_allclose(disturbance.command(), [0.0, 0.0])


def test_required_scenarios_capture_center_and_recover() -> None:
    result = run_acceptance_suite(seed=17, random_trials=1, duration_seconds=16.0)
    assert result["passed"]
    assert result["total_episodes"] == 4
    for episode in result["results"]:
        assert episode["survived"]
        assert episode["captured_before_disturbance"]
        assert episode["recovered_after_disturbance"]
        assert episode["disturbance_count"] == 6
        assert episode["disturbance_steps_applied"] == 120
        assert episode["effective_disturbance_steps"] >= 96
        assert episode["affected_disturbance_count"] == 6
        assert episode["final_cart_position"] < 0.5
