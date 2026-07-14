"""State-aware handoff between swing-up and upright stabilization."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from controllers.coupled_energy_swingup import CoupledEnergySwingUp


class StateController(Protocol):
    def control(self, state: np.ndarray) -> np.ndarray: ...


class HybridSwingUpLQR:
    """Hysteretic swing-up/LQR handoff with a short capture dwell.

    Entering LQR uses a strict region in angle, angular velocity, cart
    position, and cart velocity.  Exiting uses wider thresholds so the hybrid
    controller cannot chatter at a single angle boundary.
    """

    def __init__(
        self,
        swing_up: StateController,
        lqr: StateController,
        enter_angle: float = 0.3,
        exit_angle: float = 0.5,
        assist_angle: float = 0.5,
        assist_exit_angle: float = 0.65,
        enter_angle_rate: float = 2.5,
        exit_angle_rate: float = 4.0,
        assist_angle_rate: float = 4.0,
        assist_exit_angle_rate: float = 5.0,
        enter_position: float = 3.5,
        exit_position: float = 4.5,
        assist_position: float = 4.5,
        assist_exit_position: float = 4.8,
        enter_velocity: float = 2.5,
        exit_velocity: float = 4.0,
        assist_velocity: float = 4.0,
        assist_exit_velocity: float = 5.0,
        capture_steps: int = 5,
    ) -> None:
        if capture_steps < 1:
            raise ValueError("capture_steps must be at least one")
        self.swing_up = swing_up
        self.lqr = lqr
        self.enter_angle = float(enter_angle)
        self.exit_angle = float(exit_angle)
        self.assist_angle = float(assist_angle)
        self.assist_exit_angle = float(assist_exit_angle)
        self.enter_angle_rate = float(enter_angle_rate)
        self.exit_angle_rate = float(exit_angle_rate)
        self.assist_angle_rate = float(assist_angle_rate)
        self.assist_exit_angle_rate = float(assist_exit_angle_rate)
        self.enter_position = float(enter_position)
        self.exit_position = float(exit_position)
        self.assist_position = float(assist_position)
        self.assist_exit_position = float(assist_exit_position)
        self.enter_velocity = float(enter_velocity)
        self.exit_velocity = float(exit_velocity)
        self.assist_velocity = float(assist_velocity)
        self.assist_exit_velocity = float(assist_exit_velocity)
        self.capture_steps = int(capture_steps)
        self.mode = "swing-up"
        self._capture_count = 0

    @staticmethod
    def physical_angles_and_rates(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        canonical = CoupledEnergySwingUp.canonical_upright_state(state)
        return (
            np.array([canonical[2], -canonical[3]]),
            np.array([canonical[6], -canonical[7]]),
        )

    def reset(self, mode: str = "swing-up") -> None:
        if mode not in {"swing-up", "lqr"}:
            raise ValueError("mode must be 'swing-up' or 'lqr'")
        self.mode = mode
        self._capture_count = 0

    def _inside_capture_region(self, state: np.ndarray) -> bool:
        angles, angle_rates = self.physical_angles_and_rates(state)
        values = np.asarray(state, dtype=float)
        return bool(
            np.max(np.abs(angles)) < self.enter_angle
            and np.max(np.abs(angle_rates)) < self.enter_angle_rate
            and np.max(np.abs(values[0:2])) < self.enter_position
            and np.max(np.abs(values[4:6])) < self.enter_velocity
        )

    def _outside_stabilization_region(self, state: np.ndarray) -> bool:
        angles, angle_rates = self.physical_angles_and_rates(state)
        values = np.asarray(state, dtype=float)
        return bool(
            np.max(np.abs(angles)) > self.exit_angle
            or np.max(np.abs(angle_rates)) > self.exit_angle_rate
            or np.max(np.abs(values[0:2])) > self.exit_position
            or np.max(np.abs(values[4:6])) > self.exit_velocity
        )

    def _inside_assist_region(self, state: np.ndarray) -> bool:
        angles, angle_rates = self.physical_angles_and_rates(state)
        values = np.asarray(state, dtype=float)
        return bool(
            np.max(np.abs(angles)) < self.assist_angle
            and np.max(np.abs(angle_rates)) < self.assist_angle_rate
            and np.max(np.abs(values[0:2])) < self.assist_position
            and np.max(np.abs(values[4:6])) < self.assist_velocity
        )

    def _outside_assist_region(self, state: np.ndarray) -> bool:
        angles, angle_rates = self.physical_angles_and_rates(state)
        values = np.asarray(state, dtype=float)
        return bool(
            np.max(np.abs(angles)) > self.assist_exit_angle
            or np.max(np.abs(angle_rates)) > self.assist_exit_angle_rate
            or np.max(np.abs(values[0:2])) > self.assist_exit_position
            or np.max(np.abs(values[4:6])) > self.assist_exit_velocity
        )

    def control(self, state: np.ndarray) -> np.ndarray:
        if self.mode == "lqr":
            if self._outside_stabilization_region(state):
                self.mode = "swing-up"
                self._capture_count = 0
        elif self.mode == "capture":
            if self._inside_capture_region(state):
                self._capture_count += 1
            else:
                self._capture_count = 0

            if self._capture_count >= self.capture_steps:
                self.mode = "lqr"
            elif self._outside_assist_region(state):
                self.mode = "swing-up"
        else:
            if self._inside_capture_region(state):
                self._capture_count += 1
            else:
                self._capture_count = 0

            if self._capture_count >= self.capture_steps:
                self.mode = "lqr"
            elif self._inside_assist_region(state):
                self.mode = "capture"

        if self.mode in {"capture", "lqr"}:
            return self.lqr.control(CoupledEnergySwingUp.canonical_upright_state(state))
        return self.swing_up.control(state)
