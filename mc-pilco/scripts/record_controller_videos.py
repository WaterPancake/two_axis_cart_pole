"""Record one demo video per autonomous controller.

Each video shows the same story:

    initial state -> swing-up -> balance -> random push -> balance -> push -> balance

Run from the ``mc-pilco`` directory:

    ../.venv/bin/python scripts/record_controller_videos.py --probe
    ../.venv/bin/python scripts/record_controller_videos.py

The script uses the same controller objects, checkpoints, and capture
thresholds as the web demo and the robustness acceptance suite.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from mc_pilco.acceptance import AcceptanceThresholds, physical_metrics  # noqa: E402
from mc_pilco.experiment import load_policy  # noqa: E402
from mc_pilco.ppo_baseline import load_ppo_baseline  # noqa: E402
from mc_pilco.robust import RobustMCPILCOController  # noqa: E402
from mc_pilco.simulator import SCENARIOS, CartPoleSimulator  # noqa: E402
from mc_pilco.standalone_mc import load_standalone_mc  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent

DEFAULT_STANDALONE_MC = (
    PROJECT_DIR / "artifacts" / "benchmark" / "mc-candidate" / "checkpoint-8828.pt"
)
DEFAULT_PPO = PROJECT_DIR / "artifacts" / "benchmark" / "ppo-fixed" / "checkpoint-100500.zip"
DEFAULT_CHECKPOINT = PROJECT_DIR / "artifacts" / "latest" / "checkpoint.pt"

CONTROLLERS = ("hybrid", "mc-pilco", "robust", "mc-standalone", "ppo")

LABELS = {
    "hybrid": "Energy swing-up + LQR",
    "mc-pilco": "MC-PILCO (legacy direct)",
    "robust": "MC-PILCO + fallback (90/10)",
    "mc-standalone": "MC-PILCO standalone",
    "ppo": "PPO standalone",
}

# Push pulses: 0.25 s of 14 N on the cart sliders, in two random directions.
PUSH_FORCE = 14.0
PUSH_SECONDS = 0.25


def build_controller(name: str, simulator: CartPoleSimulator):
    """Return the controller object that the web demo uses for ``name``."""

    gain = float(abs(simulator.backend.model.actuator_gear[0, 0]))
    if name == "hybrid":
        return RobustMCPILCOController(control_dt=simulator.control_dt, actuator_gain=gain)
    if name == "robust":
        return RobustMCPILCOController(
            control_dt=simulator.control_dt,
            actuator_gain=gain,
            learned_policy=load_policy(DEFAULT_CHECKPOINT),
            residual_scale=0.1,
        )
    if name == "mc-pilco":
        return load_policy(DEFAULT_CHECKPOINT)
    if name == "mc-standalone":
        return load_standalone_mc(DEFAULT_STANDALONE_MC)
    if name == "ppo":
        return load_ppo_baseline(DEFAULT_PPO, require_prior=True)
    raise ValueError(f"unknown controller {name!r}")


@dataclass
class Episode:
    controller: str
    frame_size: tuple[int, int]
    thresholds: AcceptanceThresholds = field(default_factory=AcceptanceThresholds)
    capture_dwell: float = 1.0
    pre_push_delay: float = 1.5
    recover_timeout: float = 5.0
    hold_seconds: float = 2.5
    max_seconds: float = 26.0
    render: bool = False
    canvas: tuple[int, int] = (960, 540)
    frame_skip: int = 10
    fps: int = 25

    def run(self, initial_state: np.ndarray, pushes: list[np.ndarray]) -> dict:
        simulator = CartPoleSimulator(frame_skip=self.frame_skip)
        controller = build_controller(self.controller, simulator)
        if hasattr(controller, "reset"):
            controller.reset()
        simulator.set_state(initial_state)

        dt = simulator.control_dt
        data = simulator.backend.data
        renderer = None
        if self.render:
            model = simulator.backend.model
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.canvas[0])
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.canvas[1])
            renderer = mujoco.Renderer(model, height=self.canvas[1], width=self.canvas[0])
            # The low-resolution shadow map leaves blocky artifacts at this zoom.
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = 45.0
        camera.elevation = -18.0
        camera.distance = 3.2

        frames: list[Image.Image] = []
        history: deque[tuple[float, float, float, float]] = deque(maxlen=int(10.0 / dt))
        render_every = max(1, round(1.0 / (self.fps * dt)))

        phase = "swing"
        phase_start = 0.0
        push_index = 0
        capture_streak = 0.0
        time_to_capture: float | None = None
        recoveries: list[bool] = []
        max_cart = 0.0
        terminated = False
        steps = int(round(self.max_seconds / dt))

        state = simulator.state()
        for step in range(steps):
            time_s = step * dt
            angle, rate, cart_position, cart_velocity = physical_metrics(state)

            if phase == "swing" and capture_streak >= self.capture_dwell:
                if time_to_capture is None:
                    time_to_capture = time_s
                phase, phase_start = "settle", time_s
            elif phase == "settle" and time_s - phase_start >= self.pre_push_delay:
                phase, phase_start = "push", time_s
                capture_streak = 0.0
            elif phase == "push":
                if time_s - phase_start >= PUSH_SECONDS:
                    phase, phase_start = "recover", time_s
                    capture_streak = 0.0
            elif phase == "recover":
                if capture_streak >= self.capture_dwell:
                    recoveries.append(True)
                    push_index += 1
                    if push_index < len(pushes):
                        phase, phase_start = "push", time_s
                        capture_streak = 0.0
                    else:
                        phase, phase_start = "hold", time_s
                elif time_s - phase_start >= self.recover_timeout:
                    recoveries.append(False)
                    push_index += 1
                    if push_index < len(pushes):
                        phase, phase_start = "push", time_s
                        capture_streak = 0.0
                    else:
                        phase, phase_start = "hold", time_s
            elif phase == "hold" and time_s - phase_start >= self.hold_seconds:
                break

            action = np.asarray(controller.control(state), dtype=np.float64)
            pushing = phase == "push"
            if pushing:
                data.qfrc_applied[0:2] = pushes[min(push_index, len(pushes) - 1)]
            state, _, terminated = simulator.step(action)
            if pushing:
                data.qfrc_applied[0:2] = 0.0

            angle, rate, cart_position, cart_velocity = physical_metrics(state)
            history.append((time_s, angle, cart_position, cart_velocity))
            max_cart = max(max_cart, float(np.max(np.abs(state[0:2]))))

            if (
                angle < self.thresholds.upright_angle
                and rate < self.thresholds.tangent_rate
                and cart_position < self.thresholds.cart_position
                and cart_velocity < self.thresholds.cart_velocity
            ):
                capture_streak += dt
            else:
                capture_streak = 0.0

            if renderer is not None and step % render_every == 0:
                camera.lookat[:] = [float(state[0]), float(state[1]), 0.35]
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                frames.append(
                    self._annotate(
                        frame,
                        time_s=time_s,
                        phase=phase,
                        push_index=push_index,
                        angle=angle,
                        rate=rate,
                        cart_position=cart_position,
                        state=state,
                        history=history,
                        pushing=pushing,
                    )
                )
            if terminated:
                break

        if renderer is not None:
            renderer.close()

        return {
            "controller": self.controller,
            "captured": time_to_capture is not None,
            "time_to_capture": time_to_capture,
            "recoveries": recoveries,
            "max_cart_excursion": max_cart,
            "terminated": terminated,
            "frames": frames,
            "duration": (len(frames) * render_every * dt) if frames else 0.0,
        }

    def _annotate(self, frame, *, time_s, phase, push_index, angle, rate, cart_position,
                  state, history, pushing) -> Image.Image:
        draw = ImageDraw.Draw(frame)
        title = LABELS.get(self.controller, self.controller)
        big = self._font(30)
        small = self._font(19)
        draw.rectangle([0, 0, frame.width, 46], fill=(16, 22, 34))
        draw.text((14, 8), title, fill=(255, 255, 255), font=big)
        draw.text((frame.width - 190, 15), f"t = {time_s:5.1f} s", fill=(180, 200, 230), font=small)

        if phase == "swing":
            phase_text, colour = "SWING-UP", (255, 196, 0)
        elif phase == "settle":
            phase_text, colour = "BALANCED", (80, 220, 120)
        elif phase == "push":
            phase_text, colour = f"PUSH {push_index + 1}", (255, 70, 70)
        elif phase == "recover":
            phase_text, colour = "RECOVERING", (255, 140, 60)
        else:
            phase_text, colour = "BALANCED", (80, 220, 120)
        draw.rectangle([14, 56, 14 + 190, 88], fill=(16, 22, 34))
        draw.text((24, 61), phase_text, fill=colour, font=small)

        draw.rectangle([0, frame.height - 40, frame.width, frame.height], fill=(16, 22, 34))
        draw.text(
            (14, frame.height - 32),
            f"upright error {angle:5.3f} rad   rate {rate:5.2f} rad/s   "
            f"cart ({state[0]:6.2f}, {state[1]:6.2f})",
            fill=(210, 220, 235),
            font=small,
        )
        self._draw_trace(draw, history, frame.width, frame.height)
        return frame

    @staticmethod
    def _font(size: int):
        for path in (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _draw_trace(draw, history, width, height):
        if len(history) < 2:
            return
        box_w, box_h = 300, 84
        x0 = width - box_w - 16
        y0 = height - box_h - 46
        draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(12, 17, 26), outline=(60, 70, 90))
        times = np.array([h[0] for h in history])
        span = max(times[-1] - times[0], 1e-6)
        for index, (key, colour) in enumerate(((1, (90, 170, 255)), (2, (255, 170, 70)))):
            values = np.array([h[key] for h in history])
            scale = 1.6 if key == 1 else 6.0
            xs = x0 + 6 + (times - times[0]) / span * (box_w - 12)
            ys = y0 + box_h - 6 - np.clip(values / scale, 0.0, 1.0) * (box_h - 14)
            points = list(zip(xs.tolist(), ys.tolist()))
            draw.line(points, fill=colour, width=2)


def write_mp4(frames: list[Image.Image], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("no frames to write")
    width, height = frames[0].size
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.asarray(frame.convert("RGB")).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "media" / "controller_videos")
    parser.add_argument("--scenario", default="diagonal", choices=tuple(SCENARIOS))
    parser.add_argument("--initial-state", default=None, help="comma separated 8 values")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--frame-skip", type=int, default=10, help="physics steps per control step")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--push-force", type=float, default=PUSH_FORCE)
    parser.add_argument("--controllers", default=",".join(CONTROLLERS))
    parser.add_argument("--probe", action="store_true", help="run without rendering")
    parser.add_argument("--seeds", default="7", help="seeds for --probe")
    parser.add_argument("--showcase", action="store_true", help="concat the successful videos")
    args = parser.parse_args()

    push_force = args.push_force
    if args.initial_state:
        initial_state = np.array([float(v) for v in args.initial_state.split(",")])
        assert initial_state.shape == (8,), "initial state must have 8 values"
    else:
        initial_state = SCENARIOS[args.scenario]

    names = [name.strip() for name in args.controllers.split(",") if name.strip()]

    if args.probe:
        for seed in [int(s) for s in args.seeds.split(",")]:
            rng = np.random.default_rng(seed)
            angles = rng.uniform(0.0, 2.0 * np.pi, size=2)
            pushes = [push_force * np.array([np.cos(a), np.sin(a)]) for a in angles]
            for name in names:
                episode = Episode(
                    controller=name,
                    frame_size=(args.width, args.height),
                    render=False,
                    frame_skip=args.frame_skip,
                    fps=args.fps,
                )
                result = episode.run(initial_state, pushes)
                print(
                    f"seed {seed:>3} {name:<14} captured={result['captured']!s:<5} "
                    f"t_capture={result['time_to_capture']} "
                    f"recoveries={result['recoveries']} "
                    f"max|cart|={result['max_cart_excursion']:.2f} "
                    f"terminated={result['terminated']}"
                )
        return

    rng = np.random.default_rng(args.seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=2)
    pushes = [push_force * np.array([np.cos(a), np.sin(a)]) for a in angles]
    print(f"initial state: {np.round(initial_state, 3).tolist()}")
    print(f"pushes: {[np.round(p, 2).tolist() for p in pushes]}")

    results = []
    for name in names:
        episode = Episode(
            controller=name,
            frame_size=(args.width, args.height),
            render=True,
            canvas=(args.width, args.height),
            frame_skip=args.frame_skip,
            fps=args.fps,
        )
        result = episode.run(initial_state, pushes)
        path = (args.out / f"{name}.mp4").resolve()
        write_mp4(result.pop("frames"), path, args.fps)
        result["video"] = str(path.relative_to(REPO_ROOT))
        results.append(result)
        print(
            f"{name:<14} captured={result['captured']!s:<5} "
            f"t_capture={result['time_to_capture']} recoveries={result['recoveries']} "
            f"max|cart|={result['max_cart_excursion']:.2f} -> {path}"
        )

    summary = {
        "scenario": {
            "initial_state": initial_state.tolist(),
            "frame_skip": args.frame_skip,
            "control_dt": args.frame_skip * 0.002,
            "disturbances": {
                "type": "cart force pulse",
                "force_n": push_force,
                "seconds": PUSH_SECONDS,
                "vectors": [np.round(p, 4).tolist() for p in pushes],
            },
            "seed": args.seed,
            "fps": args.fps,
            "resolution": f"{args.width}x{args.height}",
            "capture_thresholds": {
                "upright_angle_rad": AcceptanceThresholds().upright_angle,
                "tangent_rate": AcceptanceThresholds().tangent_rate,
                "cart_position_m": AcceptanceThresholds().cart_position,
                "cart_velocity": AcceptanceThresholds().cart_velocity,
            },
        },
        "controllers": results,
    }

    if args.showcase:
        good = [
            REPO_ROOT / entry["video"]
            for entry in results
            if entry["captured"] and all(entry["recoveries"])
        ]
        if good:
            list_file = args.out / "concat.txt"
            list_file.write_text("".join(f"file '{path.resolve()}'\n" for path in good))
            showcase = (args.out / "controller_showcase.mp4").resolve()
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(list_file), "-c", "copy", "-y", str(showcase),
                ],
                check=True,
            )
            summary["showcase"] = str(showcase.relative_to(REPO_ROOT))
            print(f"showcase -> {showcase}")

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary -> {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
