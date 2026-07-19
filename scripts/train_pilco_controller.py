"""Train the repository's basic PILCO-inspired controller.

The loop intentionally stays small and inspectable:

1. collect random transitions from MuJoCo,
2. fit a Gaussian-process model of one-step state deltas,
3. optimize an RBF policy with CEM inside that model,
4. collect transitions from the new policy, and
5. save the policy and final dynamics model.

This is PILCO-inspired rather than a reproduction of the original algorithm:
it uses sampled mean rollouts and CEM instead of analytic moment matching and
policy gradients.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from controllers.pilco import (
    GaussianProcessDynamicsModel,
    RBFPolicy,
    SwingUpCostWeights,
    optimize_policy_cem,
    save_pilco_artifact,
)
from envs import TwoAxisInvertedPendulum
from pilco_workflow import (
    PILCO_SCENARIOS,
    collect_rollouts,
    evaluate_policy_scenarios,
    random_piecewise_policy,
    sample_optimization_starts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basic PILCO-style controller.")
    parser.add_argument("--output", default="artifacts/pilco_controller.npz")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--random-rollouts", type=int, default=8)
    parser.add_argument("--policy-rollouts", type=int, default=4)
    parser.add_argument("--steps-per-rollout", type=int, default=250)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--exploration-hold", type=int, default=8)
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument("--policy-centers", type=int, default=12)
    parser.add_argument("--model-max-points", type=int, default=700)
    parser.add_argument("--model-length-scale", type=float, default=1.5)
    parser.add_argument("--model-noise", type=float, default=1e-5)
    parser.add_argument("--cem-iterations", type=int, default=8)
    parser.add_argument("--cem-population", type=int, default=48)
    parser.add_argument("--cem-elite-fraction", type=float, default=0.2)
    parser.add_argument("--cem-initial-std", type=float, default=0.45)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--optimization-starts", type=int, default=8)
    parser.add_argument("--cart-range", type=float, default=1.0)
    parser.add_argument("--angle-range", type=float, default=float(np.pi))
    parser.add_argument("--cart-velocity-range", type=float, default=1.0)
    parser.add_argument("--pole-velocity-range", type=float, default=3.0)
    parser.add_argument("--uncertainty-weight", type=float, default=0.0)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny train/save path for tests and documentation checks.",
    )
    return parser.parse_args()


def apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    args.iterations = 1
    args.random_rollouts = 2
    args.policy_rollouts = 1
    args.steps_per_rollout = 24
    args.frame_skip = 4
    args.exploration_hold = 4
    args.policy_centers = 4
    args.model_max_points = 96
    args.cem_iterations = 2
    args.cem_population = 8
    args.cem_elite_fraction = 0.25
    args.horizon = 12
    args.optimization_starts = 3
    args.cart_range = 0.4
    args.angle_range = 0.7
    args.cart_velocity_range = 0.5
    args.pole_velocity_range = 1.0
    args.eval_steps = 24
    return args


def mean_cost(summaries: list[dict[str, float]]) -> float:
    return float(np.mean([summary["mean_cost"] for summary in summaries]))


def optimization_starts(
    args: argparse.Namespace, rng: np.random.Generator
) -> np.ndarray:
    starts = [state.copy() for state in PILCO_SCENARIOS.values()]
    random_count = max(args.optimization_starts - len(starts), 0)
    if random_count:
        starts.extend(
            sample_optimization_starts(
                rng,
                count=random_count,
                cart_range=args.cart_range,
                angle_range=args.angle_range,
                cart_velocity_range=args.cart_velocity_range,
                pole_velocity_range=args.pole_velocity_range,
            )
        )
    return np.vstack(starts)


def main() -> None:
    args = apply_smoke_overrides(parse_args())
    rng = np.random.default_rng(args.seed)
    env = TwoAxisInvertedPendulum()
    started = time.perf_counter()
    cost_weights = SwingUpCostWeights()

    print("collecting random exploration transitions...")
    exploration = random_piecewise_policy(
        rng, action_limit=args.action_limit, hold_steps=args.exploration_hold
    )
    dataset, summaries = collect_rollouts(
        env,
        rng,
        exploration,
        rollouts=args.random_rollouts,
        steps=args.steps_per_rollout,
        frame_skip=args.frame_skip,
        cart_range=args.cart_range,
        angle_range=args.angle_range,
        cart_velocity_range=args.cart_velocity_range,
        pole_velocity_range=args.pole_velocity_range,
    )
    print(f"  {len(dataset.states)} transitions, mean cost {mean_cost(summaries):.3f}")

    policy = RBFPolicy.from_states(
        dataset.states,
        num_centers=args.policy_centers,
        rng=rng,
        action_limit=args.action_limit,
    )
    history: list[dict[str, float]] = []

    for iteration in range(args.iterations):
        print(f"iteration {iteration + 1}/{args.iterations}: fit GP and optimize policy")
        model = GaussianProcessDynamicsModel(
            length_scale=args.model_length_scale,
            noise=args.model_noise,
            max_points=args.model_max_points,
        ).fit(dataset.states, dataset.actions, dataset.next_states)
        policy, result = optimize_policy_cem(
            model,
            policy,
            optimization_starts(args, rng),
            horizon=args.horizon,
            iterations=args.cem_iterations,
            population=args.cem_population,
            elite_fraction=args.cem_elite_fraction,
            initial_std=args.cem_initial_std,
            uncertainty_weight=args.uncertainty_weight,
            rng=rng,
            cost_weights=cost_weights,
        )
        scenario_results = evaluate_policy_scenarios(
            env, policy, steps=args.eval_steps, frame_skip=args.frame_skip
        )
        upright = int(sum(item["upright"] for item in scenario_results.values()))
        scenario_cost = float(
            np.mean([item["mean_cost"] for item in scenario_results.values()])
        )
        print(
            f"  model cost {result.best_cost:.3f}; "
            f"real named scenarios {upright}/4 upright, mean cost {scenario_cost:.3f}"
        )

        policy_batch, policy_summaries = collect_rollouts(
            env,
            rng,
            lambda state, _step: policy.control_physical(state),
            rollouts=args.policy_rollouts,
            steps=args.steps_per_rollout,
            frame_skip=args.frame_skip,
            cart_range=args.cart_range,
            angle_range=args.angle_range,
            cart_velocity_range=args.cart_velocity_range,
            pole_velocity_range=args.pole_velocity_range,
        )
        dataset = dataset.append(policy_batch)
        history.append(
            {
                "iteration": float(iteration + 1),
                "model_cost": result.best_cost,
                "real_scenario_cost": scenario_cost,
                "real_scenario_upright": float(upright),
                "policy_rollout_cost": mean_cost(policy_summaries),
                "transitions": float(len(dataset.states)),
            }
        )

    final_model = GaussianProcessDynamicsModel(
        length_scale=args.model_length_scale,
        noise=args.model_noise,
        max_points=args.model_max_points,
    ).fit(dataset.states, dataset.actions, dataset.next_states)
    final_results = evaluate_policy_scenarios(
        env, policy, steps=args.eval_steps, frame_skip=args.frame_skip
    )
    metadata = {
        "algorithm": "PILCO-inspired GP dynamics plus CEM policy search",
        "seed": args.seed,
        "iterations": args.iterations,
        "transitions": len(dataset.states),
        "frame_skip": args.frame_skip,
        "control_dt": float(env.model.opt.timestep * args.frame_skip),
        "smoke": bool(args.smoke),
        "cost_weights": asdict(cost_weights),
        "history": history,
        "final_scenario_results": final_results,
    }
    output = Path(args.output)
    save_pilco_artifact(output, policy, model=final_model, metadata=metadata)
    print(f"saved artifact: {output}")
    print(f"elapsed: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
