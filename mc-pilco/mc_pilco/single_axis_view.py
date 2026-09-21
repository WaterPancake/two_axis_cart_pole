"""MuJoCo viewer for frozen single-axis policies."""

from __future__ import annotations

from pathlib import Path
import time

import mujoco.viewer
import numpy as np

from mc_pilco.single_axis_mc import load_single_axis_mc
from mc_pilco.single_axis_ppo import load_single_axis_ppo
from mc_pilco.single_axis_report import DEFAULT_MC, DEFAULT_PPO
from mc_pilco.single_axis_task import HARD_CART_LIMIT, SingleAxisSimulator, sample_state


def run_single_axis_view(
    *,
    method: str,
    checkpoint: Path | None = None,
    scenario: str = "swingup",
    seed: int = 7001,
) -> None:
    if method == "mc-pilco":
        controller = load_single_axis_mc(checkpoint or DEFAULT_MC)
    elif method == "ppo":
        controller = load_single_axis_ppo(checkpoint or DEFAULT_PPO)
    else:
        raise ValueError("method must be 'mc-pilco' or 'ppo'")
    if scenario not in {"stabilize", "swingup"}:
        raise ValueError("scenario must be 'stabilize' or 'swingup'")
    simulator = SingleAxisSimulator()
    rng = np.random.default_rng(seed)
    initial, _ = sample_state(rng, scenario)
    simulator.set_state(initial)
    print(f"Running standalone {method} from {scenario}; close the viewer to stop.")
    with mujoco.viewer.launch_passive(
        simulator.backend.model,
        simulator.backend.data,
    ) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            state = simulator.state()
            _, _, terminated = simulator.step(controller.control(state))
            viewer.sync()
            if terminated or abs(simulator.state()[0]) >= HARD_CART_LIMIT:
                print("Cart limit reached; stopping the viewer.")
                break
            remaining = simulator.control_dt - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)
