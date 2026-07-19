"""
Linear Quadratic Regulator for the two-axis cart-pole.
The implementation follows the small-angle linearization derived in
docs/writeup.qmd.
"""

from typing import Optional

import numpy as np
import scipy.linalg


class LinearQuadraticRegulator:
    def __init__(
        self,
        m_cart: float = 1.0,
        m_pole: float = 0.25,
        ell: float = 0.6,
        g: float = 9.81,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        input_gain: float = 1.0,
        mujoco_y_axis: bool = False,
        control_limit: Optional[float] = None,
        wrap_angles: bool = True,
    ):
        self.m_cart = m_cart
        self.m_pole = m_pole
        self.ell = ell
        self.g = g
        self.dt = dt
        self.input_gain = input_gain
        self.mujoco_y_axis = mujoco_y_axis
        self.control_limit = control_limit
        self.wrap_angles = wrap_angles

        # The writeup models a massless rod with a point mass at the end.
        # The point-mass inertia is included in J below, so there is no extra
        # pole inertia term to add here.
        self.Inertia = 0.0

        self.A, self.B_force = self.linearize()
        self.B = self.B_force * self.input_gain

        if Q is not None:
            self.Q = Q
        else:
            self.Q = np.diag([1.0, 1.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0])

        if R is not None:
            self.R = R
        else:
            self.R = np.diag([1.0, 1.0])

        self.K = self.solve()

    def linearize(self) -> tuple[np.ndarray, np.ndarray]:
        """State: [x, y, θ_x, θ_y, ẋ, ẏ, θ̇_x, θ̇_y]."""

        J = self.Inertia + self.m_pole * self.ell**2
        D = (self.m_cart + self.m_pole) * J - (self.m_pole * self.ell) ** 2

        p = J / D
        q = (self.g * (self.ell**2) * (self.m_pole**2)) / D
        r = (self.m_pole * self.ell) / D
        s = ((self.m_cart + self.m_pole) * (self.m_pole * self.ell * self.g)) / D

        A = np.zeros((8, 8))

        A[0, 4] = 1
        A[1, 5] = 1
        A[2, 6] = 1
        A[3, 7] = 1
        A[4, 2] = -q
        A[5, 3] = -q
        A[6, 2] = s
        A[7, 3] = s

        B = np.zeros((8, 2))

        B[4, 0] = p
        B[5, 1] = p
        B[6, 0] = -r
        B[7, 1] = -r

        return A, B

    def discretize(self) -> tuple[np.ndarray, np.ndarray]:
        """Discretize the continuous-time linear model with zero-order hold."""
        if self.dt is None:
            raise ValueError("dt is required to discretize the LQR dynamics")

        n_states = self.A.shape[0]
        n_inputs = self.B.shape[1]
        augmented = np.zeros((n_states + n_inputs, n_states + n_inputs))
        augmented[:n_states, :n_states] = self.A
        augmented[:n_states, n_states:] = self.B

        discretized = scipy.linalg.expm(augmented * self.dt)
        A_d = discretized[:n_states, :n_states]
        B_d = discretized[:n_states, n_states:]
        return A_d, B_d

    def solve(self):
        if self.dt is None:
            P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
            K = np.linalg.solve(self.R, self.B.T @ P)
            return K

        A_d, B_d = self.discretize()
        P = scipy.linalg.solve_discrete_are(A_d, B_d, self.Q, self.R)
        K = np.linalg.solve(self.R + B_d.T @ P @ B_d, B_d.T @ P @ A_d)
        return K

    def control(self, cur_state: np.ndarray):
        """Returns the LQR action for the current state.

        If ``mujoco_y_axis`` is true, the y-axis pole angle/velocity are
        negated before feedback.  The mk-series models measure ``theta_y``
        about the ``1 0 0`` axis, so a positive angle leans the pole in the
        negative-y direction, opposite the convention used in the writeup.
        """
        state = np.asarray(cur_state, dtype=float).copy()

        if self.wrap_angles:
            state[2:4] = (state[2:4] + np.pi) % (2.0 * np.pi) - np.pi

        if self.mujoco_y_axis:
            state[3] *= -1.0
            state[7] *= -1.0

        action = -self.K @ state

        if self.control_limit is not None:
            action = np.clip(action, -self.control_limit, self.control_limit)

        return action
