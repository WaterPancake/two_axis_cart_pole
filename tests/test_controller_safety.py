"""Regression tests for the algorithmic swing-up and LQR controllers."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from controllers import (
    CoupledEnergySwingUp,
    HybridSwingUpLQR,
    LinearQuadraticRegulator,
)
from envs import TwoAxisInvertedPendulum


class ControllerSafetyTest(unittest.TestCase):
    def make_hybrid(self, env: TwoAxisInvertedPendulum) -> HybridSwingUpLQR:
        gain = float(abs(env.model.actuator_gear[0, 0])) or 1.0
        swing_up = CoupledEnergySwingUp(input_gain=gain)
        lqr = LinearQuadraticRegulator(
            dt=env.model.opt.timestep,
            input_gain=gain,
            mujoco_y_axis=True,
            control_limit=0.8,
        )
        return HybridSwingUpLQR(swing_up, lqr)

    def assert_no_warnings(self, env: TwoAxisInvertedPendulum) -> None:
        self.assertFalse(
            any(warning.number for warning in env.data.warning),
            "MuJoCo emitted a numerical warning",
        )

    def test_dynamics_stay_regular_at_former_coordinate_singularity(self) -> None:
        """mk2's universal joint was singular at ty = +/- pi/2 (pole parallel
        to the ground): coordinate rates and the armature energy stored in
        them blew up there. The ball-joint model must sail through the same
        configuration with bounded, physical angular velocity and without
        numerical warnings."""
        env = TwoAxisInvertedPendulum()
        # Exactly on the former singular surface, with transverse motion that
        # the universal-joint chart could only represent with unbounded rates.
        env.set_state(
            np.array([0.0, 0.0, 0.7, np.pi / 2.0]),
            np.array([0.0, 0.0, 1.0, 1.0]),
        )
        for _ in range(2000):
            env.control(np.zeros(2))
        self.assert_no_warnings(env)
        # |omega| is bounded by total energy: E ~ mgl + KE0 gives |omega| < 10.
        pole_omega = env.data.qvel[2:]
        self.assertLess(float(np.max(np.abs(pole_omega))), 10.0)

    def test_coupled_kinematics_matches_mujoco_tip_sensor(self) -> None:
        env = TwoAxisInvertedPendulum()
        env.set_state(
            np.array([0.2, -0.3, 0.7, -1.1]),
            np.array([0.4, -0.2, 1.3, -0.8]),
        )
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

    def test_environment_clamps_actions_and_rejects_non_finite_input(self) -> None:
        env = TwoAxisInvertedPendulum()
        applied = env.control(np.array([100.0, -100.0]))
        np.testing.assert_allclose(applied, [1.0, -1.0])
        with self.assertRaises(ValueError):
            env.control(np.array([np.nan, 0.0]))

    def test_handoff_rejects_high_rate_and_uses_hysteresis(self) -> None:
        env = TwoAxisInvertedPendulum()
        hybrid = self.make_hybrid(env)
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
        hybrid = self.make_hybrid(env)
        for _ in range(hybrid.capture_steps):
            hybrid.control(equivalent_upright)
        self.assertEqual(hybrid.mode, "lqr")
