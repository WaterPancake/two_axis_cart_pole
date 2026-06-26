"""Render a controller demo to a GIF (headless).

Unlike the interactive viewer, this renders off-screen, so it needs no display
and no ``mjpython`` -- a single command produces a GIF:

    uv run --extra demo scripts/record_demo.py                    # swing-up hero
    uv run --extra demo scripts/record_demo.py --scenario twoaxis  # two-axis

Scenarios:
  * ``swingup`` -- the x-axis pole starts near hanging; energy swing-up pumps it
    upright and hands off to LQR. The camera tracks the sliding cart.
  * ``twoaxis`` -- the cart starts offset in both x and y with the x-pole near
    hanging; it swings the pole up and LQR drives the cart back to the origin.
    A fixed camera shows the cart translating in two axes during the recovery.

On Linux without a display, set ``MUJOCO_GL=egl`` (or ``osmesa``). macOS works
out of the box.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from controllers import EnergySwingUp, LinearQuadraticRegulator
from envs import TwoAxisInvertedPendulum

SCENARIOS = {
    "swingup": {
        "out": "media/demo.gif",
        "qpos": [0.0, 0.0, 2.8, 0.0],
        "qvel": [0.0, 0.0, 0.5, 0.0],
        "track_cart": True,
        "azimuth": 90.0,
        "elevation": -15.0,
        "distance": 3.0,
        "lookat": [0.0, 0.0, 0.3],
        "seconds": 7.0,
    },
    "twoaxis": {
        "out": "media/demo_2axis.gif",
        "qpos": [-1.0, 1.2, 0.3, 0.0],
        "qvel": [1.4, -1.4, 9.0, 0.0],
        "track_cart": False,
        "azimuth": 45.0,
        "elevation": -22.0,
        "distance": 4.8,
        "lookat": [0.0, 0.4, 0.4],
        "seconds": 6.0,
    },
    "extreme": {
        "out": "media/demo_extreme.gif",
        "qpos": [-1.0, 1.5, 0.3, 0.0],
        "qvel": [1.8, -1.8, 10.5, 0.0],
        "track_cart": False,
        "azimuth": 45.0,
        "elevation": -24.0,
        "distance": 7.5,
        "lookat": [1.2, 0.4, 0.3],
        "seconds": 7.0,
    },
}


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def physical_angles(obs: np.ndarray) -> np.ndarray:
    wrapped = wrap_to_pi(obs[2:4])
    return np.array([wrapped[0], -wrapped[1]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a controller demo GIF.")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), default="swingup")
    parser.add_argument("--out", default=None, help="Output path (defaults per scenario).")
    parser.add_argument("--seconds", type=float, default=None, help="Simulated duration.")
    parser.add_argument("--fps", type=int, default=24, help="GIF frame rate.")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--colors", type=int, default=96, help="GIF palette size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SCENARIOS[args.scenario]
    out_path = Path(args.out) if args.out else Path(cfg["out"])
    seconds = args.seconds if args.seconds is not None else cfg["seconds"]

    env = TwoAxisInvertedPendulum()
    env.reset()
    env.data.qpos[:] = cfg["qpos"]
    env.data.qvel[:] = cfg["qvel"]
    mujoco.mj_forward(env.model, env.data)

    lqr_threshold = 0.25  # radians (~15 degrees)
    swing_up = EnergySwingUp(
        k=5.0, position_gain=0.2, velocity_gain=0.4, mujoco_y_axis=True, control_limit=1.0
    )
    lqr = LinearQuadraticRegulator(
        dt=env.model.opt.timestep,
        input_gain=float(abs(env.model.actuator_gear[0, 0])) or 1.0,
        mujoco_y_axis=True,
        control_limit=1.0,
    )

    timestep = env.model.opt.timestep
    total_steps = int(seconds / timestep)
    render_every = max(1, round(1.0 / (args.fps * timestep)))

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = cfg["azimuth"]
    camera.elevation = cfg["elevation"]
    camera.distance = cfg["distance"]
    camera.lookat[:] = cfg["lookat"]

    frames: list[Image.Image] = []
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    try:
        for step in range(total_steps):
            obs = env.get_obs()
            if np.max(np.abs(physical_angles(obs))) < lqr_threshold:
                u = lqr.control(obs)
            else:
                u = swing_up.control(obs)
            env.control(u)

            if step % render_every == 0:
                if cfg["track_cart"]:
                    camera.lookat[:] = [env.data.qpos[0], env.data.qpos[1], 0.3]
                renderer.update_scene(env.data, camera=camera)
                frames.append(Image.fromarray(renderer.render()))
    finally:
        renderer.close()

    # Quantize every frame to one shared palette with dithering off. The busy
    # checkerboard floor dithers into huge GIFs otherwise; a fixed palette keeps
    # the file small and avoids inter-frame flicker.
    palette = frames[0].quantize(colors=args.colors, method=Image.FASTOCTREE)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    print(f"wrote {len(frames)} frames -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
