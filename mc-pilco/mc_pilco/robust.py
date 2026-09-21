"""Empirically validated hybrid fallback for learned MC-PILCO policies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mc_pilco.policy import BoundedMLPPolicy
from mc_pilco.state import backend_from_physical

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_inserted_path = str(REPOSITORY_ROOT) not in sys.path
_previous_bytecode_setting = sys.dont_write_bytecode
sys.dont_write_bytecode = True
if _inserted_path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
try:
    from controllers import (
        CoupledEnergySwingUp,
        HybridSwingUpLQR,
        LinearQuadraticRegulator,
    )
finally:
    if _inserted_path:
        sys.path.remove(str(REPOSITORY_ROOT))
    sys.dont_write_bytecode = _previous_bytecode_setting


class RobustMCPILCOController:
    """Global swing-up/LQR safety controller with an optional learned residual.

    The classical path provides the tested fallback. A learned MC-PILCO policy
    contributes through a bounded convex blend, retaining at least 75 percent
    fallback authority and all hybrid recovery transitions.
    """

    def __init__(
        self,
        *,
        control_dt: float,
        actuator_gain: float,
        learned_policy: BoundedMLPPolicy | None = None,
        residual_scale: float = 0.0,
    ) -> None:
        if not 0.0 <= residual_scale <= 0.25:
            raise ValueError("residual_scale must be between 0 and 0.25")
        swing_up = CoupledEnergySwingUp(input_gain=actuator_gain)
        lqr = LinearQuadraticRegulator(
            dt=control_dt,
            input_gain=actuator_gain,
            mujoco_y_axis=True,
            control_limit=0.8,
        )
        self.hybrid = HybridSwingUpLQR(swing_up, lqr)
        self.learned_policy = learned_policy
        self.residual_scale = float(residual_scale)
        self.learned_active = False

    @property
    def mode(self) -> str:
        return self.hybrid.mode

    def reset(self) -> None:
        self.hybrid.reset("swing-up")
        self.learned_active = False

    def control(self, physical_state: np.ndarray) -> np.ndarray:
        raw_state = backend_from_physical(physical_state)
        fallback = np.asarray(self.hybrid.control(raw_state), dtype=np.float64)
        if self.learned_policy is None or self.residual_scale == 0.0:
            self.learned_active = False
            return fallback
        try:
            learned = np.asarray(self.learned_policy.control(physical_state), dtype=np.float64)
        except Exception:
            self.learned_active = False
            return fallback
        if learned.shape != (2,) or not np.all(np.isfinite(learned)):
            self.learned_active = False
            return fallback
        learned = np.clip(learned, -1.0, 1.0)
        self.learned_active = True
        return np.clip(
            (1.0 - self.residual_scale) * fallback + self.residual_scale * learned,
            -1.0,
            1.0,
        )
