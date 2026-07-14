"""
Mujoco backend for the the two axis inverted pendulum
"""

import mujoco
import mujoco.viewer
import numpy as np
from numpy import random
from numpy.random import Generator
import time
from pathlib import Path
from typing import Callable, Optional

MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "mk4.xml"

# Universal-joint chart shared by every controller in this project
# (see assets/mk2.xml and CoupledEnergySwingUp):
#
#   pole direction p = [sin(tx)cos(ty), -sin(ty), cos(tx)cos(ty)]
#
# i.e. orientation R = Ry(tx) @ Rx(ty). mk2 realizes the chart directly with
# two hinges, which is singular at ty = +/- pi/2 (pole parallel to the
# ground). mk4 uses a ball joint, so the helpers below convert between the
# quaternion state and this chart at the interface boundary only; the
# simulated dynamics stay smooth for every pole direction.

_CHART_EPS = 1e-9


def pole_chart_to_quat_angvel(
    theta_x: float, theta_y: float, theta_x_dot: float, theta_y_dot: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map chart coordinates/rates to (quaternion, world angular velocity)."""

    quat_y = np.array([np.cos(theta_x / 2.0), 0.0, np.sin(theta_x / 2.0), 0.0])
    quat_x = np.array([np.cos(theta_y / 2.0), np.sin(theta_y / 2.0), 0.0, 0.0])
    quat = np.empty(4)
    mujoco.mju_mulQuat(quat, quat_y, quat_x)

    # Hinge axes in world frame: y for theta_x, then the rotated x for theta_y.
    omega_world = theta_x_dot * np.array([0.0, 1.0, 0.0]) + theta_y_dot * np.array(
        [np.cos(theta_x), 0.0, -np.sin(theta_x)]
    )
    return quat, omega_world


def pole_quat_angvel_to_chart(
    quat: np.ndarray, omega_world: np.ndarray
) -> tuple[float, float, float, float]:
    """Map (quaternion, world angular velocity) to chart coordinates/rates.

    The rates are chart derivatives, so reconstructing the tip velocity from
    them (as ``CoupledEnergySwingUp.pole_direction_and_rate`` does) recovers
    the true tip velocity exactly. Like every two-angle chart of the sphere,
    they grow unbounded when the pole points along +/- y; the dynamics no
    longer do.
    """

    direction = np.empty(3)
    mujoco.mju_rotVecQuat(direction, np.array([0.0, 0.0, 1.0]), np.asarray(quat))
    direction_rate = np.cross(omega_world, direction)

    horizontal_norm = float(np.hypot(direction[0], direction[2]))
    theta_x = float(np.arctan2(direction[0], direction[2]))
    theta_y = float(np.arctan2(-direction[1], horizontal_norm))

    safe_norm = max(horizontal_norm, _CHART_EPS)
    theta_x_dot = float(
        (direction[2] * direction_rate[0] - direction[0] * direction_rate[2])
        / safe_norm**2
    )
    theta_y_dot = float(-direction_rate[1] / safe_norm)
    return theta_x, theta_y, theta_x_dot, theta_y_dot


class ArrowKeyController:
    """Maps arrow key presses into x/y cart forces."""

    def __init__(
        self, force_step: float = 0.25, max_force: float = 1.0, decay: float = 0.9
    ):
        self.force_step = float(force_step)
        self.max_force = float(max_force)
        self.decay = float(decay)
        self._action = np.zeros(2, dtype=float)

    def on_key(self, keycode: int) -> None:
        if keycode == 265:
            self._action[1] += self.force_step
        elif keycode == 264:
            self._action[1] -= self.force_step
        elif keycode == 263:
            self._action[0] -= self.force_step
        elif keycode == 262:
            self._action[0] += self.force_step

        np.clip(self._action, -self.max_force, self.max_force, out=self._action)

    def action(self) -> np.ndarray:
        action = self._action.copy()
        self._action *= self.decay
        self._action[np.abs(self._action) < 1e-3] = 0.0
        return action


class TwoAxisInvertedPendulum:
    def __init__(self, xml_path: Path = MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        ball_joints = [
            joint_id
            for joint_id in range(self.model.njnt)
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL
        ]
        self._pole_is_ball = bool(ball_joints)
        if self._pole_is_ball:
            self._quat_adr = int(self.model.jnt_qposadr[ball_joints[0]])
            self._omega_adr = int(self.model.jnt_dofadr[ball_joints[0]])

    def reset(self, rng: Optional[Generator] = None):
        mujoco.mj_resetData(self.model, self.data)

        if rng is not None:
            pos = rng.uniform(-0.5, 0.5, 2)
            angle = rng.uniform(-0.5, 0.5, 2)
            cart_vel = rng.uniform(-0.5, 0.5, 2)
            pole_vel = rng.uniform(-0.5, 0.5, 2)
            self.set_state(np.concatenate([pos, angle]), np.concatenate([cart_vel, pole_vel]))
        else:
            mujoco.mj_forward(self.model, self.data)

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Set the state from universal-joint coordinates.

        ``qpos`` is [x, y, theta_x, theta_y] and ``qvel`` their rates,
        regardless of how the loaded model parameterizes the pole.
        """

        qpos = np.asarray(qpos, dtype=float)
        qvel = np.asarray(qvel, dtype=float)
        if qpos.shape != (4,) or qvel.shape != (4,):
            raise ValueError("qpos and qvel must each have shape (4,)")

        self.data.qpos[0:2] = qpos[0:2]
        self.data.qvel[0:2] = qvel[0:2]

        if self._pole_is_ball:
            quat, omega_world = pole_chart_to_quat_angvel(
                qpos[2], qpos[3], qvel[2], qvel[3]
            )
            # MuJoCo expresses ball-joint velocities in the child body frame.
            omega_local = np.empty(3)
            mujoco.mju_rotVecQuat(omega_local, omega_world, self._quat_conj(quat))
            self.data.qpos[self._quat_adr : self._quat_adr + 4] = quat
            self.data.qvel[self._omega_adr : self._omega_adr + 3] = omega_local
        else:
            self.data.qpos[2:4] = qpos[2:4]
            self.data.qvel[2:4] = qvel[2:4]

        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _quat_conj(quat: np.ndarray) -> np.ndarray:
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]])

    def get_obs(self):
        """
        | Idx | Observation                                                |
        |-----+------------------------------------------------------------|
        |  0  | x position of the cart                                     |
        |  1  | y position of the cart                                     |
        |  2  | angle of cart's pole from the x axis expressed in radians  |
        |  3  | angle of cart's pole from the y axis expressed in raidans  |
        |  4  | x velocity of the cart                                     |
        |  5  | y velocity of the cart                                     |
        |  6  | angular velocity of cart along the x axis                  |
        |  7  | angular velocity of cart along the y axis                  |
        """

        if not self._pole_is_ball:
            return np.concatenate([self.data.qpos, self.data.qvel])

        quat = self.data.qpos[self._quat_adr : self._quat_adr + 4]
        omega_local = self.data.qvel[self._omega_adr : self._omega_adr + 3]
        omega_world = np.empty(3)
        mujoco.mju_rotVecQuat(omega_world, omega_local, quat)
        theta_x, theta_y, theta_x_dot, theta_y_dot = pole_quat_angvel_to_chart(
            quat, omega_world
        )
        return np.array(
            [
                self.data.qpos[0],
                self.data.qpos[1],
                theta_x,
                theta_y,
                self.data.qvel[0],
                self.data.qvel[1],
                theta_x_dot,
                theta_y_dot,
            ]
        )

    def get_obs_2p(self):
        x = self.get_obs()

        x = [float(round(a, 2)) for a in x]

        return x

    def control(self, action: np.ndarray) -> np.ndarray:
        applied = np.asarray(action, dtype=float)
        if applied.shape != self.data.ctrl.shape:
            raise ValueError(f"action must have shape {self.data.ctrl.shape}, got {applied.shape}")
        if not np.all(np.isfinite(applied)):
            raise ValueError("action must contain only finite values")

        applied = applied.copy()
        limited = self.model.actuator_ctrllimited.astype(bool)
        applied[limited] = np.clip(
            applied[limited],
            self.model.actuator_ctrlrange[limited, 0],
            self.model.actuator_ctrlrange[limited, 1],
        )
        self.data.ctrl[:] = applied
        mujoco.mj_step(self.model, self.data)
        return applied

    def run_passive_viewer(
        self,
        step_callback: Optional[
            Callable[["TwoAxisInvertedPendulum"], np.ndarray]
        ] = None,
        timestep: float = 0.01,
    ) -> None:
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                action = random.uniform(-1, 1, size=2)
                if step_callback is not None:
                    action = np.asarray(step_callback(self), dtype=float)

                self.control(action)
                viewer.sync()
                time.sleep(timestep)

    def run_interactive_viewer(
        self,
        timestep: float = 0.01,
        force_step: float = 0.25,
        max_force: float = 1.0,
        decay: float = 0.9,
    ) -> None:
        controller = ArrowKeyController(
            force_step=force_step,
            max_force=max_force,
            decay=decay,
        )

        print("Interactive controls: arrow keys for x/y control.")

        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            key_callback=controller.on_key,
        ) as viewer:
            while viewer.is_running():
                self.control(controller.action())
                viewer.sync()
                time.sleep(timestep)
