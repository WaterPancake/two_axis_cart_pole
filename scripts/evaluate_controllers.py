"""Headless controller checks for the two-axis cart-pole.

These are lightweight simulation checks rather than a full test suite.  They are
useful while iterating on controllers because they run MuJoCo without opening a
viewer and report simple pass/fail metrics.
"""

import argparse
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from controllers import CoupledEnergySwingUp, HybridSwingUpLQR, LinearQuadraticRegulator
from envs import TwoAxisInvertedPendulum


@dataclass
class ControllerResult:
    name: str
    success: bool
    final_obs: np.ndarray
    max_abs_cart_position: float
    handoff_step: int | None = None

    @property
    def final_physical_angles(self) -> np.ndarray:
        return physical_angles(self.final_obs)

    @property
    def final_physical_angle_rates(self) -> np.ndarray:
        canonical = CoupledEnergySwingUp.canonical_upright_state(self.final_obs)
        return np.array([canonical[6], -canonical[7]])


def physical_angles(obs: np.ndarray) -> np.ndarray:
    canonical = CoupledEnergySwingUp.canonical_upright_state(obs)
    return np.array([canonical[2], -canonical[3]])


def actuator_gain(env: TwoAxisInvertedPendulum) -> float:
    return float(abs(env.model.actuator_gear[0, 0])) or 1.0


def make_lqr(env: TwoAxisInvertedPendulum) -> LinearQuadraticRegulator:
    return LinearQuadraticRegulator(
        dt=env.model.opt.timestep,
        input_gain=actuator_gain(env),
        mujoco_y_axis=True,
        control_limit=0.8,
    )


def set_state(
    env: TwoAxisInvertedPendulum,
    qpos: list[float],
    qvel: list[float],
) -> None:
    env.reset()
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    mujoco.mj_forward(env.model, env.data)


def run_steps(
    env: TwoAxisInvertedPendulum,
    steps: int,
    policy: Callable[[np.ndarray, int], np.ndarray],
) -> float:
    max_abs_cart_position = 0.0

    for step in range(steps):
        obs = env.get_obs()
        max_abs_cart_position = max(
            max_abs_cart_position,
            float(np.max(np.abs(obs[0:2]))),
        )
        env.control(policy(obs, step))

    return max_abs_cart_position


def result_passed(
    obs: np.ndarray,
    max_abs_cart_position: float,
    angle_tol: float = 0.05,
    rate_tol: float = 0.10,
    pos_tol: float = 0.10,
    max_pos_tol: float = 4.5,
) -> bool:
    return bool(
        np.max(np.abs(physical_angles(obs))) < angle_tol
        and np.max(np.abs([obs[4], obs[5], obs[6], obs[7]])) < rate_tol
        and np.max(np.abs(obs[0:2])) < pos_tol
        and max_abs_cart_position < max_pos_tol
    )


def evaluate_lqr_stabilization(steps: int = 2_000) -> ControllerResult:
    """Stabilize from a small perturbation around the upright equilibrium."""
    env = TwoAxisInvertedPendulum()
    set_state(env, qpos=[0.0, 0.0, 0.05, 0.05], qvel=[0.0, 0.0, 0.0, 0.0])

    lqr = make_lqr(env)
    max_pos = run_steps(env, steps, lambda obs, _: lqr.control(obs))
    obs = env.get_obs()

    return ControllerResult(
        name="lqr-stabilization",
        success=result_passed(obs, max_pos, angle_tol=0.02, rate_tol=0.05),
        final_obs=obs,
        max_abs_cart_position=max_pos,
    )


def evaluate_energy_swingup_handoff(
    axis: str = "x",
    steps: int = 8_000,
) -> ControllerResult:
    """Swing up one axis with coupled energy shaping, then hand off to LQR.

    The two-axis simultaneous swing-up is not robust yet, so this check isolates
    one axis while keeping the other upright.  That makes it useful for testing
    the controller sign convention and the LQR handoff.
    """
    env = TwoAxisInvertedPendulum()

    if axis == "x":
        set_state(env, qpos=[0.0, 0.0, 2.8, 0.0], qvel=[0.0, 0.0, 0.5, 0.0])
    elif axis == "y":
        # qpos[3] has the opposite sign from the physical y lean in mk2.xml.
        set_state(env, qpos=[0.0, 0.0, 0.0, -2.8], qvel=[0.0, 0.0, 0.0, -0.5])
    else:
        raise ValueError("axis must be 'x' or 'y'")

    lqr = make_lqr(env)
    swing_up = CoupledEnergySwingUp(input_gain=actuator_gain(env))
    hybrid = HybridSwingUpLQR(swing_up, lqr)
    handoff_step: int | None = None

    def policy(obs: np.ndarray, step: int) -> np.ndarray:
        nonlocal handoff_step

        action = hybrid.control(obs)
        if hybrid.mode == "lqr" and handoff_step is None:
            handoff_step = step
        return action

    max_pos = run_steps(env, steps, policy)
    obs = env.get_obs()

    return ControllerResult(
        name=f"energy-swingup-{axis}-handoff",
        success=handoff_step is not None and result_passed(obs, max_pos),
        final_obs=obs,
        max_abs_cart_position=max_pos,
        handoff_step=handoff_step,
    )


def print_result(result: ControllerResult) -> None:
    print(f"{result.name}: {'PASS' if result.success else 'FAIL'}")
    print(f"  final obs: {np.round(result.final_obs, 4)}")
    print(f"  final physical angles [x, y]: {np.round(result.final_physical_angles, 4)}")
    print(
        "  final physical angle rates [x, y]: "
        f"{np.round(result.final_physical_angle_rates, 4)}"
    )
    print(f"  max |cart position|: {result.max_abs_cart_position:.4f}")
    if result.handoff_step is not None:
        print(f"  LQR handoff step: {result.handoff_step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless controller checks.")
    parser.add_argument(
        "--case",
        choices=("lqr", "swingup-x", "swingup-y", "all"),
        default="all",
    )
    parser.add_argument(
        "--assert-pass",
        action="store_true",
        help="Exit non-zero if any selected check fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checks = []
    if args.case in {"lqr", "all"}:
        checks.append(evaluate_lqr_stabilization())
    if args.case in {"swingup-x", "all"}:
        checks.append(evaluate_energy_swingup_handoff(axis="x"))
    if args.case in {"swingup-y", "all"}:
        checks.append(evaluate_energy_swingup_handoff(axis="y"))

    for result in checks:
        print_result(result)

    if args.assert_pass and not all(result.success for result in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
