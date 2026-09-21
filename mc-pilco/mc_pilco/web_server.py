"""Dependency-free HTTP server for the interactive cart-pole website."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
import time
from typing import Any
import webbrowser

import numpy as np

from mc_pilco.acceptance import physical_metrics
from mc_pilco.experiment import checkpoint_frame_skip, load_policy
from mc_pilco.policy import BoundedMLPPolicy
from mc_pilco.robust import RobustMCPILCOController
from mc_pilco.simulator import SCENARIOS, CartPoleSimulator, sample_initial_state
from mc_pilco.standalone_mc import load_standalone_mc

try:
    from mc_pilco.ppo_baseline import PPOController, load_ppo_baseline
except ImportError:  # pragma: no cover - PPO is an optional benchmark dependency
    PPOController = Any  # type: ignore[misc,assignment]
    load_ppo_baseline = None  # type: ignore[assignment]

PROJECT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_CHECKPOINTS = (
    PROJECT_DIR / "artifacts" / "latest" / "checkpoint.pt",
    PROJECT_DIR / "artifacts" / "smoke" / "checkpoint.pt",
)
DEFAULT_STANDALONE_MC = (
    PROJECT_DIR / "artifacts" / "benchmark" / "mc-candidate" / "checkpoint-8828.pt"
)
DEFAULT_PPO = (
    PROJECT_DIR / "artifacts" / "benchmark" / "ppo-fixed" / "checkpoint-100500.zip"
)
MOVEMENT_KEYS = frozenset(("w", "a", "s", "d", "arrowup", "arrowleft", "arrowdown", "arrowright"))


def _default_checkpoint() -> Path | None:
    return next((path for path in DEFAULT_CHECKPOINTS if path.is_file()), None)


class SimulationSession:
    """Own one MuJoCo plant and arbitrate controller versus manual authority."""

    def __init__(
        self,
        *,
        frame_skip: int | None = None,
        checkpoint: Path | None = None,
        residual_scale: float = 0.1,
        seed: int = 17,
        load_benchmarks: bool = True,
    ) -> None:
        selected_checkpoint = checkpoint or _default_checkpoint()
        if frame_skip is None and (
            DEFAULT_STANDALONE_MC.is_file() or DEFAULT_PPO.is_file()
        ):
            frame_skip = 10
        elif frame_skip is None and selected_checkpoint is not None:
            try:
                frame_skip = checkpoint_frame_skip(selected_checkpoint)
            except Exception:
                frame_skip = 5
        self.lock = threading.RLock()
        self.simulator = CartPoleSimulator(frame_skip or 5)
        self.checkpoint = selected_checkpoint
        self.policy: BoundedMLPPolicy | None = None
        self.standalone_mc = (
            load_standalone_mc(DEFAULT_STANDALONE_MC)
            if load_benchmarks and DEFAULT_STANDALONE_MC.is_file()
            else None
        )
        self.ppo: PPOController | None = (
            load_ppo_baseline(DEFAULT_PPO, require_prior=True)
            if load_benchmarks and DEFAULT_PPO.is_file() and load_ppo_baseline is not None
            else None
        )
        self.policy_error: str | None = None
        self.residual_scale = float(residual_scale)
        if selected_checkpoint is not None:
            try:
                self.policy = load_policy(selected_checkpoint)
            except Exception as error:
                self.policy_error = str(error)
        gain = float(abs(self.simulator.backend.model.actuator_gear[0, 0]))
        self.fallback = RobustMCPILCOController(
            control_dt=self.simulator.control_dt,
            actuator_gain=gain,
        )
        self.robust = RobustMCPILCOController(
            control_dt=self.simulator.control_dt,
            actuator_gain=gain,
            learned_policy=self.policy,
            residual_scale=residual_scale if self.policy is not None else 0.0,
        )
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.controller = (
            "mc-standalone"
            if self.standalone_mc is not None
            else "robust"
            if self.policy is not None
            else "hybrid"
        )
        self.scenario = "swing-x" if self.standalone_mc is not None else "random"
        self.paused = False
        self.terminated = False
        self.keys: set[str] = set()
        self.client_inputs: dict[str, tuple[int, set[str], float]] = {}
        self.last_input_at = 0.0
        self.input_timeout = 0.30
        self.authority = "controller"
        self.requested_action = np.zeros(2)
        self.applied_action = np.zeros(2)
        self.step_count = 0
        self.last_finite_state = np.zeros(8)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.reset(self.scenario)

    def available_controllers(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "mc-standalone",
                "label": "MC-PILCO standalone",
                "enabled": self.standalone_mc is not None,
                "description": "Frozen benchmark policy; no runtime fallback",
            },
            {
                "id": "ppo",
                "label": "PPO standalone (fixed)",
                "enabled": self.ppo is not None,
                "description": "100.5k-transition expert-initialized PPO policy",
            },
            {
                "id": "robust",
                "label": "MC-PILCO + fallback",
                "enabled": self.policy is not None,
                "description": (
                    f"{100 * (1 - self.residual_scale):.0f}% hybrid fallback, "
                    f"{100 * self.residual_scale:.0f}% learned policy"
                ),
            },
            {
                "id": "mc-pilco",
                "label": "Legacy MC-PILCO direct",
                "enabled": self.policy is not None,
                "description": "Learned neural policy without fallback",
            },
            {
                "id": "hybrid",
                "label": "Energy + LQR",
                "enabled": True,
                "description": "Global energy shaping with local stabilization",
            },
            {
                "id": "manual",
                "label": "Manual only",
                "enabled": True,
                "description": "Zero command whenever no movement key is held",
            },
        ]

    def config(self) -> dict[str, Any]:
        return {
            "controllers": self.available_controllers(),
            "scenarios": [
                {"id": "random", "label": "Random sphere"},
                {"id": "upright", "label": "Upright perturbation"},
                {"id": "swing-x", "label": "X-axis swing-up"},
                {"id": "swing-y", "label": "Y-axis swing-up"},
                {"id": "diagonal", "label": "Diagonal swing-up"},
                {"id": "downward", "label": "Exactly downward"},
            ],
            "selected_controller": self.controller,
            "selected_scenario": self.scenario,
            "control_dt": self.simulator.control_dt,
            "rail_limit": self.simulator.rail_limit,
            "checkpoint": None if self.checkpoint is None else str(self.checkpoint),
            "policy_checkpoints": {
                "mc-standalone": (
                    str(DEFAULT_STANDALONE_MC) if self.standalone_mc is not None else None
                ),
                "ppo": str(DEFAULT_PPO) if self.ppo is not None else None,
                "legacy": None if self.checkpoint is None else str(self.checkpoint),
            },
            "policy_error": self.policy_error,
        }

    def _initial_state(self, scenario: str) -> np.ndarray:
        if scenario == "random":
            return sample_initial_state(self.rng)
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        return SCENARIOS[scenario].copy()

    def reset(self, scenario: str | None = None) -> None:
        with self.lock:
            if scenario is not None:
                if scenario != "random" and scenario not in SCENARIOS:
                    raise ValueError(f"unknown scenario: {scenario}")
                self.scenario = scenario
            self.simulator.set_state(self._initial_state(self.scenario))
            self.fallback.reset()
            self.robust.reset()
            self.keys.clear()
            self.client_inputs.clear()
            self.authority = "controller"
            self.requested_action.fill(0.0)
            self.applied_action.fill(0.0)
            self.terminated = False
            self.step_count = 0
            self.last_finite_state = self.simulator.state()

    def select_controller(self, controller: str) -> None:
        with self.lock:
            available = {item["id"]: item for item in self.available_controllers()}
            if controller not in available:
                raise ValueError(f"unknown controller: {controller}")
            if not available[controller]["enabled"]:
                raise ValueError(f"controller {controller!r} requires a checkpoint")
            self.controller = controller
            self.fallback.reset()
            self.robust.reset()

    def _refresh_keys(self) -> None:
        self.keys = set().union(*(entry[1] for entry in self.client_inputs.values()))
        self.last_input_at = max(
            (entry[2] for entry in self.client_inputs.values()), default=self.last_input_at
        )

    def set_keys(
        self,
        keys: list[str],
        *,
        client_id: str = "default",
        revision: int | None = None,
    ) -> bool:
        normalized = {str(key).lower() for key in keys}
        invalid = normalized - MOVEMENT_KEYS
        if invalid:
            raise ValueError(f"unsupported movement keys: {sorted(invalid)}")
        if not client_id or len(client_id) > 128:
            raise ValueError("client_id must contain between 1 and 128 characters")
        with self.lock:
            previous_revision = self.client_inputs.get(client_id, (-1, set(), 0.0))[0]
            next_revision = previous_revision + 1 if revision is None else int(revision)
            if next_revision <= previous_revision:
                return False
            now = time.monotonic()
            self.client_inputs[client_id] = (next_revision, normalized, now)
            self._refresh_keys()
            return True

    def _expire_inputs(self, now: float) -> None:
        expired = [
            client_id
            for client_id, (_, _, updated_at) in self.client_inputs.items()
            if now - updated_at > self.input_timeout
        ]
        for client_id in expired:
            del self.client_inputs[client_id]
        if expired:
            self._refresh_keys()
            if not self.keys:
                self.authority = (
                    "manual-idle" if self.controller == "manual" else "controller"
                )

    def set_paused(self, paused: bool) -> None:
        with self.lock:
            self.paused = bool(paused)

    @staticmethod
    def _manual_action(keys: set[str]) -> np.ndarray:
        x = float("d" in keys or "arrowright" in keys) - float(
            "a" in keys or "arrowleft" in keys
        )
        y = float("w" in keys or "arrowup" in keys) - float(
            "s" in keys or "arrowdown" in keys
        )
        action = np.array([x, y], dtype=np.float64)
        norm = float(np.linalg.norm(action))
        return action / max(norm, 1.0)

    def _controller_action(self, state: np.ndarray) -> np.ndarray:
        if self.controller == "mc-standalone" and self.standalone_mc is not None:
            return self.standalone_mc.control(state)
        if self.controller == "ppo" and self.ppo is not None:
            return self.ppo.control(state)
        if self.controller == "robust":
            return self.robust.control(state)
        if self.controller == "mc-pilco" and self.policy is not None:
            return self.policy.control(state)
        if self.controller == "hybrid":
            return self.fallback.control(state)
        return np.zeros(2)

    def step_once(self, now: float | None = None) -> None:
        with self.lock:
            current_time = time.monotonic() if now is None else now
            self._expire_inputs(current_time)
            if self.paused or self.terminated:
                return
            state = self.simulator.state()
            if self.keys:
                action = self._manual_action(self.keys)
                self.authority = "manual"
            else:
                action = self._controller_action(state)
                self.authority = "manual-idle" if self.controller == "manual" else "controller"
            self.requested_action = np.asarray(action, dtype=np.float64)
            next_state, applied, terminated = self.simulator.step(self.requested_action)
            self.applied_action = applied
            self.terminated = terminated
            if np.all(np.isfinite(next_state)):
                self.last_finite_state = next_state
            self.step_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            observed_state = self.simulator.state()
            state_valid = bool(np.all(np.isfinite(observed_state)))
            state = observed_state if state_valid else self.last_finite_state
            angle, tangent_rate, cart_position, cart_velocity = physical_metrics(state)
            theta_x, theta_y = state[2:4]
            direction = [
                float(np.sin(theta_x) * np.cos(theta_y)),
                float(np.sin(theta_y)),
                float(np.cos(theta_x) * np.cos(theta_y)),
            ]
            mode = None
            if self.controller == "robust":
                mode = self.robust.mode
            elif self.controller == "hybrid":
                mode = self.fallback.mode
            control_sources = self._control_sources(mode)
            return {
                "state": state.tolist(),
                "direction": direction,
                "requested_action": self.requested_action.tolist(),
                "applied_action": self.applied_action.tolist(),
                "controller": self.controller,
                "controller_mode": mode,
                "control_sources": control_sources,
                "scenario": self.scenario,
                "authority": self.authority,
                "keys": sorted(self.keys),
                "paused": self.paused,
                "terminated": self.terminated,
                "state_valid": state_valid,
                "simulation_time": float(self.simulator.backend.data.time),
                "step_count": self.step_count,
                "metrics": {
                    "upright_angle": angle,
                    "tangent_rate": tangent_rate,
                    "cart_position": cart_position,
                    "cart_velocity": cart_velocity,
                },
            }

    def _control_sources(self, mode: str | None) -> list[dict[str, Any]]:
        if self.authority == "manual":
            return [{"label": "Manual override", "weight": 1.0}]
        classical_label = {
            "swing-up": "Energy swing-up",
            "capture": "LQR capture",
            "lqr": "LQR stabilization",
        }.get(mode, "Energy + LQR")
        if self.controller == "robust":
            if self.robust.learned_active:
                return [
                    {
                        "label": classical_label,
                        "weight": 1.0 - self.robust.residual_scale,
                    },
                    {"label": "MC-PILCO", "weight": self.robust.residual_scale},
                ]
            return [{"label": classical_label, "weight": 1.0}]
        labels = {
            "mc-standalone": "MC-PILCO standalone",
            "ppo": "PPO standalone",
            "mc-pilco": "Legacy MC-PILCO",
            "hybrid": classical_label,
            "manual": "Manual idle",
        }
        return [{"label": labels.get(self.controller, self.controller), "weight": 1.0}]

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="mujoco-web-loop", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            self.step_once(started)
            delay = max(self.simulator.control_dt - (time.monotonic() - started), 0.0)
            self._stop_event.wait(delay)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class WebsiteHandler(BaseHTTPRequestHandler):
    session: SimulationSession

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} {format % args}")

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 16_384:
            raise ValueError("request body is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/config":
            self._json(self.session.config())
            return
        if path == "/api/state":
            self._json(self.session.snapshot())
            return
        route = "/index.html" if path == "/" else path
        candidate = (WEB_DIR / route.lstrip("/")).resolve()
        try:
            candidate.relative_to(WEB_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self._body()
            if self.path == "/api/input":
                accepted = self.session.set_keys(
                    body.get("keys", []),
                    client_id=str(body.get("client_id", "default")),
                    revision=body.get("revision"),
                )
                if not accepted:
                    self._json({"accepted": False, "state": self.session.snapshot()})
                    return
            elif self.path == "/api/reset":
                self.session.reset(body.get("scenario"))
            elif self.path == "/api/settings":
                if "controller" in body:
                    self.session.select_controller(str(body["controller"]))
                if "paused" in body:
                    self.session.set_paused(bool(body["paused"]))
            else:
                self._json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
                return
            self._json(self.session.snapshot())
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    checkpoint: Path | None = None,
    frame_skip: int | None = None,
    residual_scale: float = 0.1,
    seed: int = 17,
    open_browser: bool = True,
) -> None:
    session = SimulationSession(
        frame_skip=frame_skip,
        checkpoint=checkpoint,
        residual_scale=residual_scale,
        seed=seed,
    )
    WebsiteHandler.session = session
    server = ThreadingHTTPServer((host, port), WebsiteHandler)
    session.start()
    url = f"http://{host}:{port}"
    print(f"NumPy control lab running at {url}")
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        session.stop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--frame-skip", type=int)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)
    serve(
        host=args.host,
        port=args.port,
        checkpoint=args.checkpoint,
        frame_skip=args.frame_skip,
        residual_scale=args.residual_scale,
        seed=args.seed,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    main()
