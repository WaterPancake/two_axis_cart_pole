"""Interactive robust controller with WASD command disturbances."""

from __future__ import annotations

import time

import mujoco.viewer
import numpy as np

from mc_pilco.policy import BoundedMLPPolicy
from mc_pilco.robust import RobustMCPILCOController
from mc_pilco.simulator import SCENARIOS, CartPoleSimulator, sample_initial_state


class WASDDisturbance:
    """Turn key presses into decaying commands added after policy control."""

    def __init__(self, step: float = 0.45, decay: float = 0.92) -> None:
        self.step = float(step)
        self.decay = float(decay)
        self.value = np.zeros(2, dtype=np.float64)

    def on_key(self, keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        increments = {
            "w": np.array([0.0, self.step]),
            "a": np.array([-self.step, 0.0]),
            "s": np.array([0.0, -self.step]),
            "d": np.array([self.step, 0.0]),
        }
        if key in increments:
            self.value = np.clip(self.value + increments[key], -1.0, 1.0)
        elif key == " ":
            self.value.fill(0.0)

    def command(self) -> np.ndarray:
        command = self.value.copy()
        self.value *= self.decay
        self.value[np.abs(self.value) < 1e-3] = 0.0
        return command


def run_interactive(
    *,
    scenario: str = "random",
    seed: int = 17,
    frame_skip: int = 5,
    learned_policy: BoundedMLPPolicy | None = None,
    residual_scale: float = 0.0,
) -> None:
    simulator = CartPoleSimulator(frame_skip)
    controller = RobustMCPILCOController(
        control_dt=simulator.control_dt,
        actuator_gain=float(abs(simulator.backend.model.actuator_gear[0, 0])),
        learned_policy=learned_policy,
        residual_scale=residual_scale,
    )
    rng = np.random.default_rng(seed)
    initial_state = sample_initial_state(rng) if scenario == "random" else SCENARIOS[scenario]
    simulator.set_state(initial_state)
    controller.reset()
    disturbance = WASDDisturbance()
    print("WASD adds disturbances; space clears them. The controller remains active.")
    with mujoco.viewer.launch_passive(
        simulator.backend.model,
        simulator.backend.data,
        key_callback=disturbance.on_key,
    ) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            state = simulator.state()
            action = np.clip(controller.control(state) + disturbance.command(), -1.0, 1.0)
            _, _, terminated = simulator.step(action)
            viewer.sync()
            if terminated:
                print("Rail or finite-state limit reached; stopping the viewer.")
                break
            remaining = simulator.control_dt - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)
