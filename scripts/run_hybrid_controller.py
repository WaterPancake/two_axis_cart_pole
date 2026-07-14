"""Run coupled energy swing-up with a guarded LQR handoff in MuJoCo."""

import time

import mujoco.viewer
import numpy as np
from numpy import random

from controllers import CoupledEnergySwingUp, HybridSwingUpLQR, LinearQuadraticRegulator
from envs import TwoAxisInvertedPendulum


def physical_angles(obs: np.ndarray) -> np.ndarray:
    canonical = CoupledEnergySwingUp.canonical_upright_state(obs)
    return np.array([canonical[2], -canonical[3]])


def main() -> None:
    sys_env = TwoAxisInvertedPendulum()

    actuator_gain = float(abs(sys_env.model.actuator_gear[0, 0])) or 1.0
    swing_up_controller = CoupledEnergySwingUp(input_gain=actuator_gain)
    lqr_controller = LinearQuadraticRegulator(
        dt=sys_env.model.opt.timestep,
        input_gain=actuator_gain,
        mujoco_y_axis=True,
        control_limit=0.8,
    )
    hybrid_controller = HybridSwingUpLQR(swing_up_controller, lqr_controller)

    rng = random.default_rng(42)
    sys_env.reset(rng)

    with mujoco.viewer.launch_passive(sys_env.model, sys_env.data) as viewer:
        while viewer.is_running():
            x = sys_env.get_obs()
            theta_x, theta_y = physical_angles(x)

            u = hybrid_controller.control(x)
            mode = hybrid_controller.mode

            print(
                f"[{mode:<8}] θx: {theta_x:+.2f}  θy: {theta_y:+.2f}  u = {np.round(u, 2)}"
            )

            sys_env.control(u)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
