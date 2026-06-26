"""Run energy swing-up with an LQR handoff in the MuJoCo viewer."""

import time

import mujoco.viewer
import numpy as np
from numpy import random

from controllers import EnergySwingUp, LinearQuadraticRegulator
from envs import TwoAxisInvertedPendulum


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def physical_angles(obs: np.ndarray) -> np.ndarray:
    wrapped = wrap_to_pi(obs[2:4])
    return np.array([wrapped[0], -wrapped[1]])


def main() -> None:
    sys_env = TwoAxisInvertedPendulum()

    lqr_threshold = 0.25  # radians (~15 degrees)
    swing_up_controller = EnergySwingUp(
        k=5.0,
        position_gain=0.2,
        velocity_gain=0.4,
        mujoco_y_axis=True,
    )
    actuator_gain = float(abs(sys_env.model.actuator_gear[0, 0])) or 1.0
    lqr_controller = LinearQuadraticRegulator(
        dt=sys_env.model.opt.timestep,
        input_gain=actuator_gain,
        mujoco_y_axis=True,
        control_limit=1.0,
    )

    rng = random.default_rng(42)
    sys_env.reset(rng)

    with mujoco.viewer.launch_passive(sys_env.model, sys_env.data) as viewer:
        while viewer.is_running():
            x = sys_env.get_obs()
            theta_x, theta_y = physical_angles(x)

            if abs(theta_x) < lqr_threshold and abs(theta_y) < lqr_threshold:
                u = lqr_controller.control(x)
                mode = "LQR"
            else:
                u = swing_up_controller.control(x)
                mode = "SwingUp"

            print(
                f"[{mode:<7}] θx: {theta_x:+.2f}  θy: {theta_y:+.2f}  u = {np.round(u, 2)}"
            )

            sys_env.control(u)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
