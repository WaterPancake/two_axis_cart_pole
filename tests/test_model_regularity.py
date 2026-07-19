"""Regression tests for the ball-joint model (mk4) and its chart conversion.

mk2 parameterized the pole with two stacked hinges, which is singular when
the pole is parallel to the ground (hinge_y = +/- pi/2). Near that surface
the hinge rates blow up and the armature that regularized the mass matrix
exchanged spurious energy with the physical coordinates. mk4 simulates a
ball joint and converts to the universal-joint chart only in the observation,
so these tests pin down both halves: the conversion is exact, and the
dynamics are physically well behaved where mk2 was not.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from envs import TwoAxisInvertedPendulum

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def physical_energy(env: TwoAxisInvertedPendulum) -> float:
    """Cart + tip-mass energy computed from world kinematics."""
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "mass_site")
    sensor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, "mas_vel")
    adr = env.model.sensor_adr[sensor_id]
    tip_velocity = env.data.sensordata[adr : adr + 3]
    cart_velocity = env.data.qvel[0:2]
    tip_height = env.data.site_xpos[site_id][2]
    return (
        0.5 * 1.0 * float(cart_velocity @ cart_velocity)
        + 0.5 * 0.25 * float(tip_velocity @ tip_velocity)
        + 0.25 * 9.81 * tip_height
    )


class ModelRegularityTest(unittest.TestCase):
    def test_set_state_get_obs_round_trip_is_exact(self) -> None:
        env = TwoAxisInvertedPendulum()
        rng = np.random.default_rng(11)
        for _ in range(50):
            qpos = np.array(
                [*rng.uniform(-3.0, 3.0, 2), rng.uniform(-3.1, 3.1), rng.uniform(-1.5, 1.5)]
            )
            qvel = rng.uniform(-4.0, 4.0, 4)
            env.set_state(qpos, qvel)
            np.testing.assert_allclose(
                env.get_obs(), np.concatenate([qpos, qvel]), atol=1e-12
            )

    def test_observation_semantics_match_mk2_chart(self) -> None:
        """For the same universal-joint state, mk2 and mk4 must report the
        same observation, so controllers tuned on mk2 transfer unchanged."""
        mk2 = TwoAxisInvertedPendulum(xml_path=ASSETS / "mk2.xml")
        mk4 = TwoAxisInvertedPendulum()
        rng = np.random.default_rng(13)
        for _ in range(25):
            qpos = np.array(
                [*rng.uniform(-3.0, 3.0, 2), rng.uniform(-3.1, 3.1), rng.uniform(-1.5, 1.5)]
            )
            qvel = rng.uniform(-4.0, 4.0, 4)
            mk2.set_state(qpos, qvel)
            mk4.set_state(qpos, qvel)
            np.testing.assert_allclose(mk2.get_obs(), mk4.get_obs(), atol=1e-10)

    def test_passive_energy_never_increases_near_former_singularity(self) -> None:
        """With damping and zero control, physical energy must be
        non-increasing. mk2 injected up to ~0.24 J per pass near the
        singular surface; the ball joint must not inject any."""
        env = TwoAxisInvertedPendulum()
        scenarios = [
            ([0.0, 0.0, 0.05, 1.2], [0.0, 0.0, 0.4, 2.5]),
            ([0.0, 0.0, 0.7, 1.3], [0.0, 0.0, 2.0, 3.0]),
            ([0.0, 0.0, 0.0, 1.5], [0.0, 0.0, 1.0, 1.5]),
        ]
        for qpos, qvel in scenarios:
            with self.subTest(qpos=qpos, qvel=qvel):
                env.reset()
                env.set_state(np.asarray(qpos), np.asarray(qvel))
                energy_previous = physical_energy(env)
                injected = 0.0
                for _ in range(3000):
                    env.control(np.zeros(2))
                    energy = physical_energy(env)
                    injected += max(energy - energy_previous, 0.0)
                    energy_previous = energy
                # Integrator rounding only; mk2 injected ~0.24 J here.
                self.assertLess(injected, 1e-4)
                self.assertFalse(any(w.number for w in env.data.warning))


if __name__ == "__main__":
    unittest.main()
