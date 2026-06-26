"""Launch the MuJoCo viewer for the two-axis cart-pole."""

import argparse

from numpy import random

from envs import TwoAxisInvertedPendulum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the MuJoCo viewer.")
    parser.add_argument(
        "--viewer",
        choices=("passive", "interactive"),
        default="passive",
        help="Passive applies random controls; interactive maps arrow keys to x/y force.",
    )
    parser.add_argument(
        "--random-reset",
        action="store_true",
        help="Randomize the initial state before launching the viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys_env = TwoAxisInvertedPendulum()

    rng = random.default_rng(42)
    sys_env.reset(rng if args.random_reset else None)
    print(sys_env.get_obs_2p())

    if args.viewer == "interactive":
        sys_env.run_interactive_viewer()
    else:
        sys_env.run_passive_viewer()


if __name__ == "__main__":
    main()
