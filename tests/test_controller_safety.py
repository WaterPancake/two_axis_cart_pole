"""Regression tests for coupled recovery after sustained manual input."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from controllers import (
    CoupledEnergySwingUp,
    HybridSwingUpLQR,
    LinearQuadraticRegulator,
    RailAwareManualController,
)
from envs import TwoAxisInvertedPendulum


def physical_angles(state: np.ndarray) -> np.ndarray:
    canonical = CoupledEnergySwingUp.canonical_upright_state(state)
    return np.array([canonical[2], -canonical[3]])


class ControllerSafetyTest(unittest.TestCase):
    def make_controllers(self, env: TwoAxisInvertedPendulum):
        gain = float(abs(env.model.actuator_gear[0, 0])) or 1.0
        swing_up = CoupledEnergySwingUp(input_gain=gain)
        lqr = LinearQuadraticRegulator(
            dt=env.model.opt.timestep,
            input_gain=gain,
            mujoco_y_axis=True,
            control_limit=0.8,
        )
        return RailAwareManualController(gain), HybridSwingUpLQR(swing_up, lqr)

    def assert_no_warnings(self, env: TwoAxisInvertedPendulum) -> None:
        self.assertFalse(
            any(warning.number for warning in env.data.warning),
            "MuJoCo emitted a numerical warning",
        )

    def apply_manual_schedule(
        self,
        env: TwoAxisInvertedPendulum,
        manual: RailAwareManualController,
        schedule: list[tuple[float, list[float]]],
    ) -> float:
        dt = float(env.model.opt.timestep)
        max_position = 0.0
        for duration, direction in schedule:
            for _ in range(round(duration / dt)):
                env.control(manual.control(env.get_obs(), np.asarray(direction)))
                max_position = max(max_position, float(np.max(np.abs(env.data.qpos[0:2]))))
        return max_position

    def run_until_settled(
        self,
        env: TwoAxisInvertedPendulum,
        hybrid: HybridSwingUpLQR,
        timeout: float,
    ) -> tuple[bool, float]:
        dt = float(env.model.opt.timestep)
        max_position = 0.0
        for _ in range(round(timeout / dt)):
            env.control(hybrid.control(env.get_obs()))
            state = env.get_obs()
            max_position = max(max_position, float(np.max(np.abs(state[0:2]))))
            if (
                hybrid.mode == "lqr"
                and np.max(np.abs(physical_angles(state))) < 0.05
                and np.max(np.abs(state[4:8])) < 0.1
                and np.max(np.abs(state[0:2])) < 0.1
            ):
                return True, max_position
        return False, max_position

    def test_pole_mass_matrix_is_regularized_at_coordinate_singularity(self) -> None:
        env = TwoAxisInvertedPendulum()
        env.data.qpos[:] = [0.0, 0.0, 0.7, np.pi / 2.0]
        mujoco.mj_forward(env.model, env.data)
        mass_matrix = np.empty((env.model.nv, env.model.nv))
        mujoco.mj_fullM(env.model, env.data, mass_matrix)
        self.assertLess(np.linalg.cond(mass_matrix), 300.0)
        self.assertGreater(np.min(np.linalg.eigvalsh(mass_matrix)), 0.005)

    def test_coupled_kinematics_matches_mujoco_tip_sensor(self) -> None:
        env = TwoAxisInvertedPendulum()
        env.data.qpos[:] = [0.2, -0.3, 0.7, -1.1]
        env.data.qvel[:] = [0.4, -0.2, 1.3, -0.8]
        mujoco.mj_forward(env.model, env.data)
        direction, direction_rate = CoupledEnergySwingUp.pole_direction_and_rate(env.get_obs())

        site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "mass_site")
        sensor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, "mas_vel")
        sensor_address = env.model.sensor_adr[sensor_id]
        measured_direction = (
            env.data.site_xpos[site_id] - np.array([env.data.qpos[0], env.data.qpos[1], 0.0])
        ) / 0.6
        measured_rate = (
            env.data.sensordata[sensor_address : sensor_address + 3]
            - np.array([env.data.qvel[0], env.data.qvel[1], 0.0])
        ) / 0.6
        np.testing.assert_allclose(direction, measured_direction, atol=1e-10)
        np.testing.assert_allclose(direction_rate, measured_rate, atol=1e-10)

    def test_manual_force_is_normalized_and_brakes_before_rail(self) -> None:
        env = TwoAxisInvertedPendulum()
        manual, _ = self.make_controllers(env)
        centered = env.get_obs()
        diagonal_force = manual.physical_force(centered, np.array([1.0, 1.0]))
        self.assertAlmostEqual(float(np.linalg.norm(diagonal_force)), 3.0)

        near_rail = centered.copy()
        near_rail[0] = 4.5
        near_rail[4] = 2.0
        self.assertLess(manual.physical_force(near_rail, np.array([1.0, 0.0]))[0], 0.0)

    def test_environment_clamps_actions_and_rejects_non_finite_input(self) -> None:
        env = TwoAxisInvertedPendulum()
        applied = env.control(np.array([100.0, -100.0]))
        np.testing.assert_allclose(applied, [1.0, -1.0])
        with self.assertRaises(ValueError):
            env.control(np.array([np.nan, 0.0]))

    def test_handoff_rejects_high_rate_and_uses_hysteresis(self) -> None:
        env = TwoAxisInvertedPendulum()
        _, hybrid = self.make_controllers(env)
        high_rate = np.array([0.0, 0.0, 0.1, -0.1, 0.0, 0.0, 5.0, -5.0])
        hybrid.control(high_rate)
        self.assertEqual(hybrid.mode, "swing-up")

        captured = np.zeros(8)
        for _ in range(hybrid.capture_steps):
            hybrid.control(captured)
        self.assertEqual(hybrid.mode, "lqr")

        outside_entry_but_inside_exit = captured.copy()
        outside_entry_but_inside_exit[2] = 0.4
        hybrid.control(outside_entry_but_inside_exit)
        self.assertEqual(hybrid.mode, "lqr")

        outside_exit = captured.copy()
        outside_exit[2] = 0.6
        hybrid.control(outside_exit)
        self.assertEqual(hybrid.mode, "swing-up")

    def test_equivalent_pi_pi_upright_coordinates_are_canonicalized(self) -> None:
        equivalent_upright = np.array(
            [0.0, 0.0, np.pi, np.pi, 0.0, 0.0, 0.0, 0.0]
        )
        canonical = CoupledEnergySwingUp.canonical_upright_state(equivalent_upright)
        np.testing.assert_allclose(canonical[2:4], 0.0, atol=1e-12)

        env = TwoAxisInvertedPendulum()
        _, hybrid = self.make_controllers(env)
        for _ in range(hybrid.capture_steps):
            hybrid.control(equivalent_upright)
        self.assertEqual(hybrid.mode, "lqr")

    def test_sustained_and_coupled_manual_inputs_recover_without_runaway(self) -> None:
        scenarios = {
            "two-second-diagonal": [(2.0, [1.0, 1.0])],
            "eight-second-diagonal": [(8.0, [1.0, 1.0])],
            "diagonal-reversal": [(1.0, [1.0, 1.0]), (1.0, [-1.0, -1.0])],
            "aggressive-circle": [
                (2.0, [1.0, 0.0]),
                (2.0, [0.0, 1.0]),
                (2.0, [-1.0, 0.0]),
                (2.0, [0.0, -1.0]),
            ],
        }
        for name, schedule in scenarios.items():
            with self.subTest(name=name):
                env = TwoAxisInvertedPendulum()
                env.reset()
                manual, hybrid = self.make_controllers(env)
                hybrid.reset("lqr")
                manual_max = self.apply_manual_schedule(env, manual, schedule)
                settled, recovery_max = self.run_until_settled(env, hybrid, timeout=40.0)
                self.assertTrue(settled)
                self.assertLess(max(manual_max, recovery_max), 4.95)
                self.assert_no_warnings(env)

    def test_manual_autonomy_cycle_can_repeat(self) -> None:
        env = TwoAxisInvertedPendulum()
        env.reset()
        manual, hybrid = self.make_controllers(env)
        hybrid.reset("lqr")
        schedules = [
            [(0.5, [1.0, 1.0])],
            [(1.0, [-1.0, 0.0]), (1.0, [0.0, 1.0])],
            [(1.0, [1.0, -1.0])],
        ]
        for cycle, schedule in enumerate(schedules):
            with self.subTest(cycle=cycle):
                self.apply_manual_schedule(env, manual, schedule)
                settled, max_position = self.run_until_settled(env, hybrid, timeout=25.0)
                self.assertTrue(settled)
                self.assertLess(max_position, 4.95)
                self.assert_no_warnings(env)

    def test_seeded_manual_stress_cases_recover(self) -> None:
        rng = np.random.default_rng(20260629)
        directions = np.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [1.0, 1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ]
        )
        for case in range(12):
            with self.subTest(case=case):
                env = TwoAxisInvertedPendulum()
                env.reset()
                manual, hybrid = self.make_controllers(env)
                hybrid.reset("lqr")
                schedule = []
                for _ in range(int(rng.integers(1, 5))):
                    direction = directions[int(rng.integers(len(directions)))].tolist()
                    schedule.append((float(rng.uniform(0.2, 3.0)), direction))

                manual_max = self.apply_manual_schedule(env, manual, schedule)
                settled, recovery_max = self.run_until_settled(env, hybrid, timeout=30.0)
                self.assertTrue(settled)
                self.assertLess(max(manual_max, recovery_max), 4.95)
                self.assert_no_warnings(env)
