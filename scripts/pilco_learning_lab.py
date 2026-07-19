"""Small, inspectable experiments for the PILCO foundations lab.

This is deliberately a diagnostic tool, not a trainer.  It exposes the three
ideas a learner needs before running policy search: continuous angle features,
real transition data, and the difference between a one-step model prediction
and a recursively rolled-out prediction.

Examples
--------
python scripts/pilco_learning_lab.py features
python scripts/pilco_learning_lab.py collect --output artifacts/pilco_foundations_data.npz
python scripts/pilco_learning_lab.py model --dataset artifacts/pilco_foundations_data.npz
python scripts/pilco_learning_lab.py compare --dataset artifacts/pilco_foundations_data.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from controllers.pilco import (
    GaussianProcessDynamicsModel,
    apply_state_delta,
    policy_features,
    state_delta,
)
from envs import TwoAxisInvertedPendulum
from pilco_workflow import (
    TransitionBatch,
    collect_rollouts,
    random_piecewise_policy,
    rollout_env,
    sample_physical_state,
)


@dataclass(frozen=True)
class DatasetSplit:
    """Training and held-out partitions of a transition dataset."""

    train: TransitionBatch
    test: TransitionBatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "section",
        choices=("features", "collect", "model", "compare"),
        help="Lab section to run.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="An .npz dataset written by the 'collect' section.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pilco_foundations_data.npz"),
        help="Where the 'collect' section writes transitions.",
    )
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--action-hold", type=int, default=6)
    parser.add_argument("--cart-range", type=float, default=0.5)
    parser.add_argument("--angle-range", type=float, default=0.9)
    parser.add_argument("--cart-velocity-range", type=float, default=0.5)
    parser.add_argument("--pole-velocity-range", type=float, default=1.5)
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--max-points", type=int, default=250)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rollouts < 1 or args.steps < 2:
        raise ValueError("--rollouts must be positive and --steps must be at least two")
    if args.frame_skip < 1 or args.action_hold < 1:
        raise ValueError("--frame-skip and --action-hold must be positive")
    if not 0.0 < args.holdout < 0.5:
        raise ValueError("--holdout must be in (0, 0.5)")
    if args.max_points < 2:
        raise ValueError("--max-points must be at least two")


def save_dataset(path: Path, batch: TransitionBatch, frame_skip: int) -> None:
    """Save only plain arrays, so the data can be inspected in a REPL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=batch.states,
        actions=batch.actions,
        next_states=batch.next_states,
        frame_skip=np.array(frame_skip, dtype=int),
    )


def load_dataset(path: Path) -> tuple[TransitionBatch, int | None]:
    """Load a dataset created by :func:`save_dataset` with shape checks."""

    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        required = {"states", "actions", "next_states"}
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(f"dataset is missing arrays: {', '.join(sorted(missing))}")
        batch = TransitionBatch(
            states=np.asarray(arrays["states"], dtype=float),
            actions=np.asarray(arrays["actions"], dtype=float),
            next_states=np.asarray(arrays["next_states"], dtype=float),
        )
        frame_skip = int(arrays["frame_skip"]) if "frame_skip" in arrays.files else None

    if (
        batch.states.ndim != 2
        or batch.states.shape[1] != 8
        or batch.actions.shape != (len(batch.states), 2)
        or batch.next_states.shape != batch.states.shape
    ):
        raise ValueError("dataset must contain states (N, 8), actions (N, 2), next_states (N, 8)")
    if len(batch.states) < 4:
        raise ValueError("dataset needs at least four transitions")
    return batch, frame_skip


def split_dataset(
    batch: TransitionBatch, holdout: float, rng: np.random.Generator
) -> DatasetSplit:
    """Make a random held-out split without changing the samples themselves."""

    order = rng.permutation(len(batch.states))
    test_count = max(2, int(round(holdout * len(order))))
    test_indices = order[:test_count]
    train_indices = order[test_count:]
    if len(train_indices) < 2:
        raise ValueError("not enough training transitions after the hold-out split")

    def take(indices: np.ndarray) -> TransitionBatch:
        return TransitionBatch(
            states=batch.states[indices],
            actions=batch.actions[indices],
            next_states=batch.next_states[indices],
        )

    return DatasetSplit(train=take(train_indices), test=take(test_indices))


def collect_dataset(args: argparse.Namespace) -> tuple[TransitionBatch, int]:
    """Interact with MuJoCo under held random actions and print a compact report."""

    env = TwoAxisInvertedPendulum()
    rng = np.random.default_rng(args.seed)
    policy = random_piecewise_policy(rng, hold_steps=args.action_hold)
    batch, summaries = collect_rollouts(
        env,
        rng,
        policy,
        rollouts=args.rollouts,
        steps=args.steps,
        frame_skip=args.frame_skip,
        cart_range=args.cart_range,
        angle_range=args.angle_range,
        cart_velocity_range=args.cart_velocity_range,
        pole_velocity_range=args.pole_velocity_range,
    )
    save_dataset(args.output, batch, args.frame_skip)
    control_dt = env.model.opt.timestep * args.frame_skip
    print("Collected real MuJoCo transitions")
    print(f"  dataset: {args.output}")
    print(f"  shapes: S {batch.states.shape}, A {batch.actions.shape}, S' {batch.next_states.shape}")
    print(f"  control interval: {control_dt:.4f} s ({args.frame_skip} physics steps)")
    print(f"  action min/max: {batch.actions.min():+.3f} / {batch.actions.max():+.3f}")
    print(
        "  mean rollout cost: "
        f"{np.mean([summary['mean_cost'] for summary in summaries]):.3f} "
        "(lower is better for this task cost)"
    )
    print("  first transition:")
    print(f"    s  = {np.array2string(batch.states[0], precision=3)}")
    print(f"    a  = {np.array2string(batch.actions[0], precision=3)}")
    print(f"    s' = {np.array2string(batch.next_states[0], precision=3)}")
    return batch, args.frame_skip


def get_or_collect_dataset(args: argparse.Namespace) -> tuple[TransitionBatch, int]:
    """Use supplied data, or make a fresh small dataset for a standalone run."""

    if args.dataset is not None:
        batch, saved_frame_skip = load_dataset(args.dataset)
        frame_skip = saved_frame_skip if saved_frame_skip is not None else args.frame_skip
        print(f"Loaded {len(batch.states)} transitions from {args.dataset}")
        return batch, frame_skip
    print("No --dataset supplied; collecting a fresh dataset first.")
    return collect_dataset(args)


def print_feature_report() -> None:
    """Show why a periodic state should not be supplied as a wrapped scalar."""

    left = np.zeros(8)
    right = np.zeros(8)
    left[2] = np.pi - 0.01
    right[2] = -np.pi + 0.01
    raw_gap = float(np.linalg.norm(left - right))
    trig_gap = float(np.linalg.norm(policy_features(left) - policy_features(right)))
    print("Angle-boundary experiment")
    print(f"  physical angles: {left[2]:+.4f} rad and {right[2]:+.4f} rad")
    print(f"  raw-state distance: {raw_gap:.4f}")
    print(f"  trig-feature distance: {trig_gap:.4f}")
    print("  These poses are physically close. [sin(theta), cos(theta)] preserves that fact.")
    print(f"  feature order: {policy_features(np.zeros(8)).shape[0]} values =")
    print("    [x, y, sin(tx), cos(tx), sin(ty), cos(ty), x_dot, y_dot, tx_dot, ty_dot]")


def fit_model_report(args: argparse.Namespace) -> None:
    """Fit the one-step GP and measure predictions on transitions it did not fit."""

    batch, _ = get_or_collect_dataset(args)
    split = split_dataset(batch, args.holdout, np.random.default_rng(args.seed + 1))
    model = GaussianProcessDynamicsModel(max_points=min(args.max_points, len(split.train.states)))
    model.fit(split.train.states, split.train.actions, split.train.next_states, rng=np.random.default_rng(args.seed + 2))

    predictions = []
    variances = []
    targets = []
    for state, action, next_state in zip(
        split.test.states, split.test.actions, split.test.next_states
    ):
        prediction, variance = model.predict_delta(state, action, return_variance=True)
        predictions.append(prediction)
        variances.append(variance)
        targets.append(state_delta(next_state, state))
    errors = np.asarray(predictions) - np.asarray(targets)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mean_variance = np.mean(np.asarray(variances), axis=0)

    in_distribution_state = split.train.states[0]
    in_distribution_action = split.train.actions[0]
    _, in_distribution_variance = model.predict_delta(
        in_distribution_state, in_distribution_action, return_variance=True
    )
    out_of_distribution_state = np.array([4.5, -4.5, 2.7, -2.7, 4.0, -4.0, 8.0, -8.0])
    out_of_distribution_action = np.array([1.0, -1.0])
    _, out_of_distribution_variance = model.predict_delta(
        out_of_distribution_state, out_of_distribution_action, return_variance=True
    )

    labels = ("x", "y", "tx", "ty", "x_dot", "y_dot", "tx_dot", "ty_dot")
    print("Held-out one-step GP model report")
    print(f"  train/test transitions: {len(split.train.states)} / {len(split.test.states)}")
    print("  delta RMSE by state component:")
    for label, value, variance in zip(labels, rmse, mean_variance):
        print(f"    {label:>6}: rmse={value:.5f}, mean predictive variance={variance:.5f}")
    print("  total predictive variance (lower means more familiar to this GP):")
    print(f"    recorded training transition: {np.sum(in_distribution_variance):.6f}")
    print(f"    deliberately far-away query: {np.sum(out_of_distribution_variance):.6f}")
    print("  A low one-step error is useful, but it does not prove long-horizon rollouts are accurate.")


def wrap_angle_errors(errors: np.ndarray) -> np.ndarray:
    """Measure angular errors on the shortest arc rather than across +/- pi."""

    wrapped = np.asarray(errors, dtype=float).copy()
    wrapped[..., 2:4] = (wrapped[..., 2:4] + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped


def compare_rollout_report(args: argparse.Namespace) -> None:
    """Compare one real open-loop trajectory against recursively applied GP means."""

    batch, frame_skip = get_or_collect_dataset(args)
    model = GaussianProcessDynamicsModel(max_points=min(args.max_points, len(batch.states)))
    model.fit(batch.states, batch.actions, batch.next_states, rng=np.random.default_rng(args.seed + 3))

    env = TwoAxisInvertedPendulum()
    rng = np.random.default_rng(args.seed + 4)
    initial_state = sample_physical_state(
        rng,
        cart_range=args.cart_range,
        angle_range=args.angle_range,
        cart_velocity_range=args.cart_velocity_range,
        pole_velocity_range=args.pole_velocity_range,
    )
    action_source = random_piecewise_policy(rng, hold_steps=args.action_hold)
    actual, _ = rollout_env(
        env, initial_state, action_source, steps=args.steps, frame_skip=frame_skip
    )

    predicted_states = []
    predicted_state = actual.states[0].copy()
    for action in actual.actions:
        predicted_state = apply_state_delta(
            predicted_state, model.predict_delta(predicted_state, action)
        )
        predicted_states.append(predicted_state.copy())
    predicted = np.asarray(predicted_states)
    errors = wrap_angle_errors(predicted - actual.next_states)

    print("Recursive model-rollout comparison")
    print(f"  starting state: {np.array2string(initial_state, precision=3)}")
    print("  Both systems receive exactly the recorded action sequence; only the dynamics differ.")
    print("  RMS prediction error over time (cart position / pole angle / all components):")
    control_dt = env.model.opt.timestep * frame_skip
    horizons = sorted({1, min(5, args.steps), min(10, args.steps), args.steps})
    for horizon in horizons:
        prefix = errors[:horizon]
        cart_error = float(np.sqrt(np.mean(prefix[:, 0:2] ** 2)))
        angle_error = float(np.sqrt(np.mean(prefix[:, 2:4] ** 2)))
        total_error = float(np.sqrt(np.mean(prefix**2)))
        print(
            f"    {horizon:>3} steps ({horizon * control_dt:>5.2f} s): "
            f"{cart_error:.4f} / {angle_error:.4f} / {total_error:.4f}"
        )
    print("  Watch the error as the horizon grows: a policy optimizer can exploit that gap.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.section == "features":
        print_feature_report()
    elif args.section == "collect":
        collect_dataset(args)
    elif args.section == "model":
        fit_model_report(args)
    else:
        compare_rollout_report(args)


if __name__ == "__main__":
    main()
