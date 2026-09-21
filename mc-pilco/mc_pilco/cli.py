"""Command-line interface for training and evaluating MC-PILCO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mc_pilco.acceptance import print_acceptance, run_acceptance_suite
from mc_pilco.experiment import (
    DEFAULT_OUTPUT_DIR,
    ExperimentConfig,
    checkpoint_frame_skip,
    evaluate_policy,
    load_policy,
    train,
)
from mc_pilco.interactive import run_interactive
from mc_pilco.simulator import SCENARIOS, CartPoleSimulator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train", help="run iterative MC-PILCO training")
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--frame-skip", type=int, default=5)
    training.add_argument("--initial-rollouts", type=int, default=5)
    training.add_argument("--trial-rollouts", type=int, default=2)
    training.add_argument("--rollout-steps", type=int, default=100)
    training.add_argument("--iterations", type=int, default=3)
    training.add_argument("--max-gp-points", type=int, default=350)
    training.add_argument("--gp-steps", type=int, default=75)
    training.add_argument("--gp-learning-rate", type=float, default=0.03)
    training.add_argument("--policy-updates", type=int, default=100)
    training.add_argument("--policy-learning-rate", type=float, default=0.01)
    training.add_argument("--particles", type=int, default=96)
    training.add_argument("--imagination-horizon", type=int, default=100)
    training.add_argument("--gradient-clip", type=float, default=10.0)
    training.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    evaluation = subparsers.add_parser("evaluate", help="evaluate a saved direct policy")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--frame-skip", type=int)
    evaluation.add_argument("--steps", type=int, default=400)

    acceptance = subparsers.add_parser(
        "robust-evaluate", help="verify swing-up, centering, and disturbance recovery"
    )
    acceptance.add_argument("--seed", type=int, default=17)
    acceptance.add_argument("--random-trials", type=int, default=4)
    acceptance.add_argument("--duration", type=float, default=30.0)
    acceptance.add_argument("--frame-skip", type=int)
    acceptance.add_argument("--checkpoint", type=Path)
    acceptance.add_argument("--residual-scale", type=float, default=0.0)
    acceptance.add_argument("--no-disturbances", action="store_true")
    acceptance.add_argument("--assert-pass", action="store_true")
    acceptance.add_argument(
        "--output", type=Path, default=Path("artifacts/robust-acceptance.json")
    )

    interactive = subparsers.add_parser(
        "interactive", help="run robust control with live WASD disturbances"
    )
    interactive.add_argument(
        "--scenario", choices=("random", *SCENARIOS.keys()), default="random"
    )
    interactive.add_argument("--seed", type=int, default=17)
    interactive.add_argument("--frame-skip", type=int)
    interactive.add_argument("--checkpoint", type=Path)
    interactive.add_argument("--residual-scale", type=float, default=0.0)

    website = subparsers.add_parser("web", help="launch the browser control lab")
    website.add_argument("--host", default="127.0.0.1")
    website.add_argument("--port", type=int, default=8765)
    website.add_argument("--checkpoint", type=Path)
    website.add_argument("--frame-skip", type=int)
    website.add_argument("--residual-scale", type=float, default=0.1)
    website.add_argument("--seed", type=int, default=17)
    website.add_argument("--no-open", action="store_true")

    mc_benchmark = subparsers.add_parser(
        "train-mc-benchmark", help="train standalone long-horizon MC-PILCO"
    )
    mc_benchmark.add_argument("--seed", type=int, default=29)
    mc_benchmark.add_argument("--output-dir", type=Path)

    ppo_benchmark = subparsers.add_parser(
        "train-ppo-benchmark", help="train the standalone PPO baseline"
    )
    ppo_benchmark.add_argument("--seed", type=int, default=54)
    ppo_benchmark.add_argument("--transitions", type=int, default=100_000)
    ppo_benchmark.add_argument("--output-dir", type=Path)

    comparison = subparsers.add_parser(
        "benchmark", help="evaluate frozen MC-PILCO and PPO checkpoints"
    )
    comparison.add_argument("--mc-checkpoint", type=Path)
    comparison.add_argument("--ppo-checkpoint", type=Path)
    comparison.add_argument("--episodes", type=int, default=50)
    comparison.add_argument("--output", type=Path)

    single_mc = subparsers.add_parser(
        "train-single-axis-mc", help="train end-to-end single-axis MC-PILCO"
    )
    single_mc.add_argument("--seed", type=int, default=79)
    single_mc.add_argument("--output-dir", type=Path)

    single_ppo = subparsers.add_parser(
        "train-single-axis-ppo", help="train end-to-end single-axis PPO"
    )
    single_ppo.add_argument("--seed", type=int, default=83)
    single_ppo.add_argument("--transitions", type=int, default=500_000)
    single_ppo.add_argument("--output-dir", type=Path)

    single_benchmark = subparsers.add_parser(
        "single-axis-benchmark", help="compare frozen single-axis controllers"
    )
    single_benchmark.add_argument("--mc-checkpoint", type=Path)
    single_benchmark.add_argument("--ppo-checkpoint", type=Path)
    single_benchmark.add_argument("--episodes", type=int, default=50)
    single_benchmark.add_argument("--output", type=Path)

    single_view = subparsers.add_parser(
        "single-axis-view", help="view a frozen single-axis swing-up controller"
    )
    single_view.add_argument("--method", choices=("mc-pilco", "ppo"), required=True)
    single_view.add_argument("--checkpoint", type=Path)
    single_view.add_argument("--scenario", choices=("stabilize", "swingup"), default="swingup")
    single_view.add_argument("--seed", type=int, default=7001)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "train-single-axis-mc":
        from mc_pilco.single_axis_mc import SingleAxisMCConfig, train_single_axis_mc

        values = {"seed": args.seed}
        if args.output_dir is not None:
            values["output_dir"] = args.output_dir
        print(
            json.dumps(
                train_single_axis_mc(SingleAxisMCConfig(**values)), indent=2, default=str
            )
        )
        return
    if args.command == "train-single-axis-ppo":
        from mc_pilco.single_axis_ppo import SingleAxisPPOConfig, train_single_axis_ppo

        values = {"seed": args.seed, "total_transitions": args.transitions}
        if args.output_dir is not None:
            values["output_dir"] = args.output_dir
        print(
            json.dumps(
                train_single_axis_ppo(SingleAxisPPOConfig(**values)),
                indent=2,
                default=str,
            )
        )
        return
    if args.command == "single-axis-benchmark":
        from mc_pilco.single_axis_report import (
            DEFAULT_MC,
            DEFAULT_PPO,
            generate_single_axis_report,
        )

        report = generate_single_axis_report(
            mc_checkpoint=args.mc_checkpoint or DEFAULT_MC,
            ppo_checkpoint=args.ppo_checkpoint or DEFAULT_PPO,
            episodes_per_stratum=args.episodes,
            output=args.output,
        )
        print(json.dumps(report, indent=2))
        return
    if args.command == "single-axis-view":
        from mc_pilco.single_axis_view import run_single_axis_view

        run_single_axis_view(
            method=args.method,
            checkpoint=args.checkpoint,
            scenario=args.scenario,
            seed=args.seed,
        )
        return
    if args.command == "train-mc-benchmark":
        from mc_pilco.standalone_mc import MCPILCOConfig, train_mc_pilco

        values = {"seed": args.seed}
        if args.output_dir is not None:
            values["output_dir"] = args.output_dir
        print(json.dumps(train_mc_pilco(MCPILCOConfig(**values)), indent=2, default=str))
        return
    if args.command == "train-ppo-benchmark":
        from mc_pilco.ppo_baseline import PPOBaselineConfig, train_ppo_baseline

        values = {"seed": args.seed, "total_transitions": args.transitions}
        if args.output_dir is not None:
            values["output_dir"] = args.output_dir
        print(
            json.dumps(
                train_ppo_baseline(PPOBaselineConfig(**values)), indent=2, default=str
            )
        )
        return
    if args.command == "benchmark":
        from mc_pilco.benchmark_report import (
            DEFAULT_MC_CHECKPOINT,
            DEFAULT_PPO_CHECKPOINT,
            PROJECT_DIR,
            generate_report,
        )

        report = generate_report(
            mc_checkpoint=args.mc_checkpoint or DEFAULT_MC_CHECKPOINT,
            ppo_checkpoint=args.ppo_checkpoint or DEFAULT_PPO_CHECKPOINT,
            episodes=args.episodes,
            output=args.output
            or PROJECT_DIR / "artifacts" / "benchmark" / "comparison.json",
        )
        print(json.dumps(report, indent=2))
        return
    if args.command == "web":
        from mc_pilco.web_server import serve

        serve(
            host=args.host,
            port=args.port,
            checkpoint=args.checkpoint,
            frame_skip=args.frame_skip,
            residual_scale=args.residual_scale,
            seed=args.seed,
            open_browser=not args.no_open,
        )
        return
    if args.command == "robust-evaluate":
        policy = load_policy(args.checkpoint) if args.checkpoint else None
        frame_skip = (
            args.frame_skip
            if args.frame_skip is not None
            else checkpoint_frame_skip(args.checkpoint)
            if args.checkpoint
            else 5
        )
        result = run_acceptance_suite(
            seed=args.seed,
            random_trials=args.random_trials,
            duration_seconds=args.duration,
            disturbances=not args.no_disturbances,
            frame_skip=frame_skip,
            learned_policy=policy,
            residual_scale=args.residual_scale,
        )
        print_acceptance(result, args.output)
        if args.assert_pass and not result["passed"]:
            raise SystemExit(1)
        return
    if args.command == "interactive":
        policy = load_policy(args.checkpoint) if args.checkpoint else None
        frame_skip = (
            args.frame_skip
            if args.frame_skip is not None
            else checkpoint_frame_skip(args.checkpoint)
            if args.checkpoint
            else 5
        )
        run_interactive(
            scenario=args.scenario,
            seed=args.seed,
            frame_skip=frame_skip,
            learned_policy=policy,
            residual_scale=args.residual_scale,
        )
        return
    if args.command == "evaluate":
        policy = load_policy(args.checkpoint)
        frame_skip = args.frame_skip or checkpoint_frame_skip(args.checkpoint)
        result = evaluate_policy(
            CartPoleSimulator(frame_skip), policy, steps=args.steps
        )
    else:
        values = vars(args).copy()
        values.pop("command")
        result = train(ExperimentConfig(**values))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
