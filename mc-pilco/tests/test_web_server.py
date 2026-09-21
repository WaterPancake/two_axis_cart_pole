import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.request import urlopen

import numpy as np

from mc_pilco.web_server import PROJECT_DIR, SimulationSession, WebsiteHandler


def session_without_checkpoint() -> SimulationSession:
    return SimulationSession(
        checkpoint=PROJECT_DIR / "artifacts" / "does-not-exist.pt",
        load_benchmarks=False,
    )


def test_manual_keys_take_exclusive_authority_until_release() -> None:
    session = session_without_checkpoint()
    session.select_controller("hybrid")
    session.set_keys(["w", "ArrowRight"])
    session.step_once(now=session.last_input_at)
    manual = session.snapshot()
    assert manual["authority"] == "manual"
    np.testing.assert_allclose(manual["requested_action"], [2**-0.5, 2**-0.5])

    session.set_keys([])
    session.step_once(now=session.last_input_at)
    resumed = session.snapshot()
    assert resumed["authority"] == "controller"
    assert resumed["keys"] == []


def test_stale_manual_input_expires_to_controller() -> None:
    session = session_without_checkpoint()
    session.set_keys(["a"])
    session.step_once(now=session.last_input_at + session.input_timeout + 0.01)
    assert session.snapshot()["authority"] == "controller"


def test_stale_input_revision_cannot_reactivate_released_key() -> None:
    session = session_without_checkpoint()
    assert session.set_keys([], client_id="browser", revision=2)
    assert not session.set_keys(["w"], client_id="browser", revision=1)
    session.step_once(now=session.last_input_at)
    assert session.snapshot()["authority"] == "controller"


def test_input_timeout_expires_while_paused() -> None:
    session = session_without_checkpoint()
    session.set_keys(["d"])
    session.set_paused(True)
    session.step_once(now=session.last_input_at + session.input_timeout + 0.01)
    assert session.snapshot()["keys"] == []
    assert session.snapshot()["authority"] == "controller"


def test_snapshot_reports_live_robust_controller_blend() -> None:
    class ConstantPolicy:
        @staticmethod
        def control(_state: np.ndarray) -> np.ndarray:
            return np.array([0.25, -0.25])

    session = session_without_checkpoint()
    policy = ConstantPolicy()
    session.policy = policy  # type: ignore[assignment]
    session.robust.learned_policy = policy  # type: ignore[assignment]
    session.robust.residual_scale = 0.1
    session.select_controller("robust")
    session.step_once()

    state = session.snapshot()
    assert state["control_sources"] == [
        {"label": "Energy swing-up", "weight": 0.9},
        {"label": "MC-PILCO", "weight": 0.1},
    ]

    session.set_keys(["d"])
    session.step_once(now=session.last_input_at)
    assert session.snapshot()["control_sources"] == [
        {"label": "Manual override", "weight": 1.0}
    ]


def test_static_site_and_json_api_are_served() -> None:
    WebsiteHandler.session = session_without_checkpoint()
    server = ThreadingHTTPServer(("127.0.0.1", 0), WebsiteHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with urlopen(f"http://{host}:{port}/api/config", timeout=2) as response:
            config = json.load(response)
        assert config["selected_controller"] == "hybrid"
        assert {item["id"] for item in config["scenarios"]} >= {
            "random",
            "swing-x",
            "swing-y",
            "diagonal",
        }
        with urlopen(f"http://{host}:{port}/", timeout=2) as response:
            page = response.read().decode("utf-8")
        assert "NumPy Cart-Pole Control Lab" in page
        assert (Path(PROJECT_DIR) / "mc_pilco" / "web" / "app.js").is_file()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
