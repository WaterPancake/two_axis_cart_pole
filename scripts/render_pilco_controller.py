"""Render a saved PILCO-style controller artifact.

Examples
--------
Open an interactive MuJoCo viewer::

    mjpython scripts/render_pilco_controller.py \
      --artifact artifacts/pilco_controller.npz \
      --mode viewer \
      --scenario diagonal

Record an off-screen GIF::

    python scripts/render_pilco_controller.py \
      --artifact artifacts/pilco_controller.npz \
      --mode gif \
      --out media/pilco_diagonal.gif \
      --scenario diagonal

On Linux without a display, set ``MUJOCO_GL=egl`` or ``MUJOCO_GL=osmesa`` for
GIF rendering.  On macOS, the interactive viewer usually requires ``mjpython``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np

from controllers.pilco import (
    PilcoController,
    load_pilco_artifact,
    physical_state_from_mujoco,
)
from envs import TwoAxisInvertedPendulum
from pilco_workflow import (
    PILCO_SCENARIOS,
    pilco_initial_state,
    set_physical_state,
)

CameraMode = Literal["track", "fixed"]

CAMERAS = {
    "upright": {
        "track_cart": True,
        "azimuth": 90.0,
        "elevation": -15.0,
        "distance": 3.0,
        "lookat": [0.0, 0.0, 0.3],
        "seconds": 4.0,
    },
    "swing-x": {
        "track_cart": True,
        "azimuth": 90.0,
        "elevation": -15.0,
        "distance": 3.2,
        "lookat": [0.0, 0.0, 0.3],
        "seconds": 7.0,
    },
    "swing-y": {
        "track_cart": True,
        "azimuth": 0.0,
        "elevation": -15.0,
        "distance": 3.2,
        "lookat": [0.0, 0.0, 0.3],
        "seconds": 7.0,
    },
    "diagonal": {
        "track_cart": False,
        "azimuth": 45.0,
        "elevation": -22.0,
        "distance": 4.8,
        "lookat": [0.0, 0.4, 0.4],
        "seconds": 7.0,
    },
    "random": {
        "track_cart": False,
        "azimuth": 45.0,
        "elevation": -22.0,
        "distance": 4.8,
        "lookat": [0.0, 0.4, 0.4],
        "seconds": 6.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a saved PILCO-style controller.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--mode", choices=("viewer", "gif"), default="viewer")
    parser.add_argument(
        "--scenario",
        choices=(*PILCO_SCENARIOS.keys(), "random"),
        default="diagonal",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="GIF path; defaults to media/pilco_<scenario>.gif",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.25,
        help="Telemetry interval in seconds.",
    )
    parser.add_argument("--fps", type=int, default=24, help="GIF frame rate.")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--colors", type=int, default=96, help="GIF palette size.")
    parser.add_argument(
        "--camera",
        choices=("track", "fixed"),
        default=None,
        help="Override the scenario camera tracking behavior.",
    )
    return parser.parse_args()


def make_controller(
    artifact_path: Path,
) -> PilcoController:
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"PILCO artifact not found: {artifact_path}. "
            "Train one first with scripts/train_pilco_controller.py."
        )
    artifact = load_pilco_artifact(artifact_path)
    return PilcoController(artifact.policy)


def configure_camera(scenario: str) -> mujoco.MjvCamera:
    cfg = CAMERAS[scenario]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = cfg["azimuth"]
    camera.elevation = cfg["elevation"]
    camera.distance = cfg["distance"]
    camera.lookat[:] = cfg["lookat"]
    return camera


def should_track_cart(scenario: str, camera_mode: CameraMode | None) -> bool:
    if camera_mode == "track":
        return True
    if camera_mode == "fixed":
        return False
    return bool(CAMERAS[scenario]["track_cart"])


def reset_scenario(
    env: TwoAxisInvertedPendulum,
    scenario: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    set_physical_state(env, pilco_initial_state(scenario, rng))


class ActionRepeater:
    """Hold each controller action for ``frame_skip`` MuJoCo steps."""

    def __init__(self, frame_skip: int) -> None:
        if frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        self.frame_skip = int(frame_skip)
        self.action = np.zeros(2, dtype=float)

    def step(
        self,
        env: TwoAxisInvertedPendulum,
        controller: PilcoController,
        step: int,
    ) -> np.ndarray:
        if step % self.frame_skip == 0:
            self.action = controller.control(env.get_obs())
        env.control(self.action)
        return self.action


def telemetry(
    env: TwoAxisInvertedPendulum,
    action: np.ndarray,
) -> str:
    physical = physical_state_from_mujoco(env.get_obs())
    return (
        f"t={env.data.time:5.2f}s "
        f"θ=({physical[2]:+.2f}, {physical[3]:+.2f}) "
        f"cart=({physical[0]:+.2f}, {physical[1]:+.2f}) "
        f"u={np.round(action, 2)}"
    )


def run_viewer(args: argparse.Namespace) -> None:
    env = TwoAxisInvertedPendulum()
    reset_scenario(env, args.scenario, args.seed)
    controller = make_controller(Path(args.artifact))

    timestep = float(env.model.opt.timestep)
    max_steps = None if args.seconds is None else int(args.seconds / timestep)
    print_every_steps = max(1, int(args.print_every / timestep))
    step = 0
    repeater = ActionRepeater(args.frame_skip)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and (max_steps is None or step < max_steps):
            action = repeater.step(env, controller, step)
            if step % print_every_steps == 0:
                print(telemetry(env, action))
            viewer.sync()
            time.sleep(timestep)
            step += 1


def render_gif(args: argparse.Namespace) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional demo extra
        raise SystemExit(
            "GIF rendering requires Pillow. Install it with `pip install -e '.[demo]'`."
        ) from exc

    env = TwoAxisInvertedPendulum()
    reset_scenario(env, args.scenario, args.seed)
    controller = make_controller(Path(args.artifact))

    scenario_seconds = float(CAMERAS[args.scenario]["seconds"])
    seconds = scenario_seconds if args.seconds is None else float(args.seconds)
    out_path = Path(args.out) if args.out else Path("media") / f"pilco_{args.scenario}.gif"

    timestep = float(env.model.opt.timestep)
    total_steps = int(seconds / timestep)
    render_every = max(1, round(1.0 / (args.fps * timestep)))
    print_every_steps = max(1, int(args.print_every / timestep))
    track_cart = should_track_cart(args.scenario, args.camera)
    camera = configure_camera(args.scenario)

    frames: list[Image.Image] = []
    repeater = ActionRepeater(args.frame_skip)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    try:
        for step in range(total_steps):
            action = repeater.step(env, controller, step)
            if step % print_every_steps == 0:
                print(telemetry(env, action))

            if step % render_every == 0:
                if track_cart:
                    camera.lookat[:] = [env.data.qpos[0], env.data.qpos[1], 0.3]
                renderer.update_scene(env.data, camera=camera)
                frames.append(Image.fromarray(renderer.render()))
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("no frames rendered; increase --seconds or --fps")

    palette = frames[0].quantize(colors=args.colors, method=Image.FASTOCTREE)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    print(f"wrote {len(frames)} frames -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    args = parse_args()
    if args.frame_skip < 1:
        raise SystemExit("--frame-skip must be positive")
    if args.mode == "viewer":
        run_viewer(args)
    else:
        render_gif(args)


if __name__ == "__main__":
    main()
