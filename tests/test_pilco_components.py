"""Unit checks for the PILCO-inspired controller components."""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from controllers.pilco import (
    GaussianProcessDynamicsModel,
    PilcoController,
    RBFPolicy,
    SwingUpCostWeights,
    _population_rollout_costs,
    apply_state_delta,
    load_pilco_artifact,
    optimize_policy_cem,
    physical_state_from_mujoco,
    policy_features,
    save_pilco_artifact,
    simulate_model_rollout,
    state_delta,
    swingup_cost,
    swingup_cost_batch,
)


class PilcoComponentsTest(unittest.TestCase):
    def make_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(123)
        states = rng.normal(scale=0.2, size=(18, 8))
        states[:, 2:4] = rng.uniform(-0.5, 0.5, size=(18, 2))
        actions = rng.uniform(-0.3, 0.3, size=(18, 2))
        deltas = np.zeros_like(states)
        deltas[:, 0:2] = 0.01 * states[:, 4:6]
        deltas[:, 2:4] = 0.01 * states[:, 6:8]
        deltas[:, 4:6] = 0.02 * actions
        deltas[:, 6:8] = -0.03 * states[:, 2:4] + 0.01 * actions
        next_states = np.vstack(
            [apply_state_delta(state, delta) for state, delta in zip(states, deltas)]
        )
        return states, actions, next_states

    def test_physical_state_wraps_angles_and_flips_y_axis(self) -> None:
        raw = np.array([0.0, 0.0, 3.5, 0.3, 0.0, 0.0, 1.0, 2.0])
        physical = physical_state_from_mujoco(raw)
        self.assertLessEqual(abs(physical[2]), np.pi)
        self.assertAlmostEqual(physical[3], -0.3)
        self.assertAlmostEqual(physical[7], -2.0)

    def test_gp_fit_predict_and_policy_optimization_shapes(self) -> None:
        states, actions, next_states = self.make_dataset()
        model = GaussianProcessDynamicsModel(max_points=18).fit(states, actions, next_states)
        delta, variance = model.predict_delta(states[0], actions[0], return_variance=True)
        self.assertEqual(delta.shape, (8,))
        self.assertEqual(variance.shape, (8,))
        self.assertTrue(np.all(variance >= 0.0))

        policy = RBFPolicy.from_states(states, num_centers=3, rng=np.random.default_rng(4))
        optimized, result = optimize_policy_cem(
            model,
            policy,
            states[:2],
            horizon=3,
            iterations=1,
            population=4,
            rng=np.random.default_rng(5),
        )
        self.assertEqual(optimized.control_physical(states[0]).shape, (2,))
        np.testing.assert_allclose(
            optimized.control_features(policy_features(states[0])),
            optimized.control_physical(states[0]),
        )
        self.assertTrue(np.isfinite(result.best_cost))

    def test_artifact_round_trip_loads_policy_and_model(self) -> None:
        states, actions, next_states = self.make_dataset()
        model = GaussianProcessDynamicsModel(max_points=18).fit(states, actions, next_states)
        policy = RBFPolicy.from_states(states, num_centers=4, rng=np.random.default_rng(6))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/pilco.npz"
            save_pilco_artifact(path, policy, model=model, metadata={"test": True})
            artifact = load_pilco_artifact(path)

        self.assertTrue(artifact.metadata["test"])
        self.assertIsNotNone(artifact.model)
        np.testing.assert_allclose(
            artifact.policy.control_physical(states[0]),
            policy.control_physical(states[0]),
        )
        controller = PilcoController(artifact.policy)
        action = controller.control(states[0])
        self.assertEqual(action.shape, (2,))

    def test_state_delta_wraps_angle_difference(self) -> None:
        state = np.array([0.0, 0.0, 3.1, -3.1, 0.0, 0.0, 0.0, 0.0])
        nxt = np.array([0.0, 0.0, -3.1, 3.1, 0.0, 0.0, 0.0, 0.0])
        delta = state_delta(nxt, state)
        self.assertLess(abs(delta[2]), 0.2)
        self.assertLess(abs(delta[3]), 0.2)

    def test_policy_parameters_round_trip_through_artifact(self) -> None:
        states, _, _ = self.make_dataset()
        rng = np.random.default_rng(9)
        policy = RBFPolicy.from_states(states, num_centers=3, rng=rng)
        params = rng.normal(size=policy.num_parameters)
        candidate = policy.with_parameters(params)
        np.testing.assert_allclose(candidate.parameters(), params)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/policy.npz"
            save_pilco_artifact(path, candidate, metadata={})
            loaded = load_pilco_artifact(path)
        np.testing.assert_allclose(
            loaded.policy.control_physical(states[0]),
            candidate.control_physical(states[0]),
        )

    def test_population_rollout_matches_scalar_rollouts(self) -> None:
        states, actions, next_states = self.make_dataset()
        model = GaussianProcessDynamicsModel(max_points=18).fit(
            states, actions, next_states
        )
        rng = np.random.default_rng(12)
        policy = RBFPolicy.from_states(states, num_centers=3, rng=rng)
        population = rng.normal(scale=0.3, size=(3, policy.num_parameters))
        starts = states[:2]
        weights = SwingUpCostWeights()

        batched = _population_rollout_costs(
            model, policy, population, starts, 5, weights, 0.0
        )
        scalar = np.array(
            [
                np.mean(
                    [
                        simulate_model_rollout(
                            model, policy.with_parameters(params), start, horizon=5
                        )
                        for start in starts
                    ]
                )
                for params in population
            ]
        )
        np.testing.assert_allclose(batched, scalar, rtol=1e-10)

    def test_swingup_cost_batch_matches_scalar(self) -> None:
        states, actions, _ = self.make_dataset()
        batched = swingup_cost_batch(states, actions)
        scalar = [swingup_cost(state, action) for state, action in zip(states, actions)]
        np.testing.assert_allclose(batched, scalar)


if __name__ == "__main__":
    unittest.main()
