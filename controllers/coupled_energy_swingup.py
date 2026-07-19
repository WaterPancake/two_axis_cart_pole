"""Coupled energy shaping for the two-axis cart-pole."""

from __future__ import annotations

import numpy as np


class CoupledEnergySwingUp:
    """Shape the total 3D pendulum energy with planar cart forces.

    The pole direction for ``assets/mk2.xml`` is

    ``p = [sin(tx) cos(ty), -sin(ty), cos(tx) cos(ty)]``.

    Unlike two independent planar energy laws, this controller uses one total
    kinetic/potential energy and the horizontal tip velocity in both axes.  All
    gains and limits are expressed in physical force units; ``input_gain``
    converts Newtons to MuJoCo actuator controls.
    """

    def __init__(
        self,
        m_pole: float = 0.25,
        ell: float = 0.6,
        g: float = 9.81,
        energy_gain: float = 80.0,
        # 100 collapses the precession mode (pole circling the vertical at the
        # target energy) that the singular mk2 dynamics used to break up for
        # free; see tests/test_model_regularity.py for the model change.
        angular_momentum_gain: float = 100.0,
        position_gain: float = 2.0,
        velocity_gain: float = 4.0,
        input_gain: float = 1.0,
        force_limit: float = 10.0,
        control_limit: float = 1.0,
    ) -> None:
        if input_gain <= 0.0:
            raise ValueError("input_gain must be positive")
        if force_limit <= 0.0:
            raise ValueError("force_limit must be positive")

        self.m_pole = float(m_pole)
        self.ell = float(ell)
        self.g = float(g)
        self.energy_gain = float(energy_gain)
        self.angular_momentum_gain = float(angular_momentum_gain)
        self.position_gain = float(position_gain)
        self.velocity_gain = float(velocity_gain)
        self.input_gain = float(input_gain)
        self.force_limit = float(force_limit)
        self.control_limit = float(control_limit)
        self.J = self.m_pole * self.ell**2

    @staticmethod
    def pole_direction_and_rate(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the world-frame pole unit vector and its time derivative."""

        values = np.asarray(state, dtype=float)
        theta_x, theta_y = values[2:4]
        theta_x_dot, theta_y_dot = values[6:8]

        sin_x, cos_x = np.sin(theta_x), np.cos(theta_x)
        sin_y, cos_y = np.sin(theta_y), np.cos(theta_y)
        direction = np.array([sin_x * cos_y, -sin_y, cos_x * cos_y])
        direction_rate = np.array(
            [
                cos_x * cos_y * theta_x_dot - sin_x * sin_y * theta_y_dot,
                -cos_y * theta_y_dot,
                -sin_x * cos_y * theta_x_dot - cos_x * sin_y * theta_y_dot,
            ]
        )
        return direction, direction_rate

    @classmethod
    def canonical_upright_state(cls, state: np.ndarray) -> np.ndarray:
        """Map redundant hinge coordinates to the chart around upright.

        The universal-joint coordinates ``(0, 0)`` and ``(pi, pi)`` describe
        the same pole direction. LQR and capture checks need the equivalent
        representation closest to upright, derived from the actual 3D tip
        direction rather than independently wrapping both joint positions.
        """

        canonical = np.asarray(state, dtype=float).copy()
        direction, direction_rate = cls.pole_direction_and_rate(canonical)
        horizontal_norm = float(np.hypot(direction[0], direction[2]))
        canonical[2] = np.arctan2(direction[0], direction[2])
        canonical[3] = np.arctan2(-direction[1], horizontal_norm)

        if horizontal_norm > 1e-6:
            canonical[6] = (
                direction[2] * direction_rate[0] - direction[0] * direction_rate[2]
            ) / horizontal_norm**2
            canonical[7] = -direction_rate[1] / horizontal_norm
        else:
            # This upright chart is singular when the pole lies exactly along y.
            # Large rates ensure the capture gate rejects that configuration.
            canonical[6:8] = np.copysign(1e6, canonical[6:8] + 1e-12)
        return canonical

    def energy(self, state: np.ndarray) -> float:
        direction, direction_rate = self.pole_direction_and_rate(state)
        kinetic = 0.5 * self.J * float(direction_rate @ direction_rate)
        potential = self.m_pole * self.g * self.ell * direction[2]
        return kinetic + potential

    def physical_force(self, state: np.ndarray) -> np.ndarray:
        values = np.asarray(state, dtype=float)
        direction, direction_rate = self.pole_direction_and_rate(values)
        energy_error = self.energy(values) - self.m_pole * self.g * self.ell

        force = self.energy_gain * energy_error * direction_rate[:2]
        # Energy shaping alone permits persistent precession around the vertical
        # axis. Damp that coupled mode by opposing vertical angular momentum.
        momentum_z = self.J * (
            direction[0] * direction_rate[1] - direction[1] * direction_rate[0]
        )
        force += self.angular_momentum_gain * momentum_z * np.array(
            [-direction[1], direction[0]]
        )
        force += -self.position_gain * values[0:2] - self.velocity_gain * values[4:6]

        norm = float(np.linalg.norm(force))
        if norm > self.force_limit:
            force *= self.force_limit / norm
        return force

    def control(self, state: np.ndarray) -> np.ndarray:
        action = self.physical_force(state) / self.input_gain
        return np.clip(action, -self.control_limit, self.control_limit)
