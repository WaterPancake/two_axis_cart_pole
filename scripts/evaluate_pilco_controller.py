"""Evaluate a saved PILCO-style policy in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from controllers.pilco import (
    PilcoController,
    load_pilco_artifact,
    physical_state_from_mujoco,
    swingup_cost,
)
from envs import TwoAxisInvertedPendulum
from pilco_workflow import PILCO_SCENARIOS, pilco_initial_state, set_physical_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PILCO-style controller.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument(
        "--scenario", choices=(*PILCO_SCENARIOS.keys(), "random"), default="upright"
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--assert-finite", action="store_true")
    parser.add_argument("--assert-upright", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.frame_skip < 1:
        raise SystemExit("--frame-skip must be positive")

    artifact_path = Path(args.artifact)
    artifact = load_pilco_artifact(artifact_path)
    controller = PilcoController(artifact.policy)
    env = TwoAxisInvertedPendulum()
    rng = np.random.default_rng(args.seed)
    set_physical_state(env, pilco_initial_state(args.scenario, rng))

    total_cost = 0.0
    max_abs_cart = 0.0
    max_abs_angle = 0.0
    for _ in range(args.steps):
        raw_state = env.get_obs()
        physical_state = physical_state_from_mujoco(raw_state)
        action = controller.control(raw_state)
        total_cost += swingup_cost(physical_state, action)
        max_abs_cart = max(max_abs_cart, float(np.max(np.abs(physical_state[0:2]))))
        max_abs_angle = max(max_abs_angle, float(np.max(np.abs(physical_state[2:4]))))
        for _ in range(args.frame_skip):
            env.control(action)

    final_state = physical_state_from_mujoco(env.get_obs())
    finite = bool(np.all(np.isfinite(final_state)))
    final_angle_norm = float(np.linalg.norm(final_state[2:4]))
    final_rate_norm = float(np.linalg.norm(final_state[6:8]))
    warnings = int(sum(warning.number for warning in env.data.warning))

    print(f"artifact: {artifact_path}")
    print(f"scenario: {args.scenario}")
    print(f"mean cost: {total_cost / args.steps:.3f}")
    print(f"final physical state: {np.round(final_state, 4)}")
    print(f"final angle norm: {final_angle_norm:.4f}")
    print(f"final angular-rate norm: {final_rate_norm:.4f}")
    print(f"max |cart|: {max_abs_cart:.4f}")
    print(f"max |angle|: {max_abs_angle:.4f}")
    print(f"MuJoCo warning count: {warnings}")

    if args.assert_finite and (not finite or warnings > 0):
        raise SystemExit("PILCO evaluation produced non-finite state or MuJoCo warnings")
    if args.assert_upright and not (final_angle_norm < 0.25 and final_rate_norm < 1.0):
        raise SystemExit("PILCO evaluation did not end near the upright")


if __name__ == "__main__":
    main()
