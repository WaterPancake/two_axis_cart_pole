"""Manual cart control with physical-force and rail-safety limits."""

from __future__ import annotations

import numpy as np


class RailAwareManualController:
    """Convert a direction command to force while braking before cart limits."""

    def __init__(
        self,
        input_gain: float,
        drive_force: float = 3.0,
        braking_force: float = 4.0,
        boundary_start: float = 3.5,
        boundary_stiffness: float = 8.0,
        boundary_damping: float = 2.0,
        control_limit: float = 1.0,
    ) -> None:
        if input_gain <= 0.0:
            raise ValueError("input_gain must be positive")
        self.input_gain = float(input_gain)
        self.drive_force = float(drive_force)
        self.braking_force = float(braking_force)
        self.boundary_start = float(boundary_start)
        self.boundary_stiffness = float(boundary_stiffness)
        self.boundary_damping = float(boundary_damping)
        self.control_limit = float(control_limit)

    def physical_force(self, state: np.ndarray, direction: np.ndarray) -> np.ndarray:
        values = np.asarray(state, dtype=float)
        command = np.clip(np.asarray(direction, dtype=float), -1.0, 1.0)
        norm = float(np.linalg.norm(command))
        if norm > 1.0:
            command /= norm
        force = self.drive_force * command

        for axis in range(2):
            position = float(values[axis])
            distance = abs(position)
            if distance <= self.boundary_start:
                continue
            sign = np.sign(position)
            outward_speed = max(0.0, sign * float(values[axis + 4]))
            correction = self.boundary_stiffness * (distance - self.boundary_start)
            correction += self.boundary_damping * outward_speed
            force[axis] -= sign * correction

        norm = float(np.linalg.norm(force))
        if norm > self.braking_force:
            force *= self.braking_force / norm
        return force

    def control(self, state: np.ndarray, direction: np.ndarray) -> np.ndarray:
        action = self.physical_force(state, direction) / self.input_gain
        return np.clip(action, -self.control_limit, self.control_limit)
