# Two-Axis Cart-Pole

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Physics: MuJoCo](https://img.shields.io/badge/physics-MuJoCo%203.7-orange.svg)](https://mujoco.org/)

A two-degree-of-freedom extension of the canonical cart-pole: a cart that moves
in the **x–y plane** balancing an inverted pendulum that is free to fall in
**any direction**. The repo implements and compares three controller approaches
— algorithmic energy swing-up, LQR, and a PILCO-inspired learner — on the full
8-state nonlinear system, simulated in [MuJoCo](https://mujoco.org/).

<p align="center">
  <img src="demo_extreme.gif" alt="Energy swing-up handing off to LQR balance" width="600">
</p>


---

The dynamics are linearized from scratch; the full derivation lives in the
[writeup](https://waterpancake.github.io/two_axis_cart_pole/).

## The system

State vector:

$$\mathbf{x} = \left[\, x,\; y,\; \theta_x,\; \theta_y,\; \dot{x},\; \dot{y},\; \dot{\theta}_x,\; \dot{\theta}_y \,\right]$$

| Variable | Description | Range |
| --- | --- | --- |
| $x,\ y$ | cart position along the x / y axis | $[-5, 5]$  |
| $\theta_x,\ \theta_y$ | pole lean angle about the x / y axis | $[0, 2\pi)$|
| $\dot x,\ \dot y$ | cart linear velocity | unbounded |
| $\dot\theta_x,\ \dot\theta_y$ | pole angular velocity | unbounded |

Control input is the planar force applied to the cart:

$$\mathbf{u} = \left[\, F_x,\; F_y \,\right]^\top$$

## Controllers

| Controller | File | Idea |
| --- | --- | --- |
| **LQR** | `controllers/lqr.py` | Stabilizes the upright equilibrium using the small-angle linearization, solving the continuous- or discrete-time algebraic Riccati equation. |
| **Algorithmic swing-up + LQR** | `controllers/coupled_energy_swingup.py`, `controllers/hybrid.py` | Shapes the pole's 3D energy, then hands off to LQR inside a guarded capture region. |
| **PILCO-style learned policy** | `controllers/pilco.py` | GP dynamics model + RBF policy optimized by model rollouts; see the [PILCO foundations lab](docs/pilco_foundations_lab.md). |

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/WaterPancake/two_axis_cart_pole.git
cd two_axis_cart_pole
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quickstart

```bash
# Watch the hybrid swing-up -> LQR controller balance the pole
python scripts/run_hybrid_controller.py

# Open the interactive viewer and drive the cart with the arrow keys
python scripts/run_viewer.py --viewer interactive

# Run the headless controller checks (no viewer)
python scripts/evaluate_controllers.py --case all

# Smoke-train and evaluate the PILCO-style learned controller
python scripts/train_pilco_controller.py --smoke --output artifacts/pilco_smoke.npz
python scripts/evaluate_pilco_controller.py --artifact artifacts/pilco_smoke.npz --assert-finite
```

> **macOS:** the MuJoCo viewer must be launched with `mjpython` instead of
> `python` (e.g. `mjpython scripts/run_hybrid_controller.py`). Headless scripts
> such as `evaluate_controllers.py` run under plain `python`.

## PILCO-style learned controller

The learned-controller workflow is intentionally dependency-light: it fits a
Gaussian-process model of one-step state deltas, optimizes a squashed RBF policy
with many cheap rollouts inside that model, then saves the policy/model to an
`.npz` artifact. Evaluation runs that learned policy directly, without LQR or
algorithmic recovery hidden inside it.

Read the [PILCO foundations lab](docs/pilco_foundations_lab.md) for theory and
first principles, then use the [PILCO implementation lab](docs/pilco_code_lab.md)
for runnable training, evaluation, and debugging exercises. Training outputs
live under the ignored `artifacts/` directory and should be passed explicitly
to evaluation and rendering commands.

## Results

Reproduce with `python scripts/evaluate_controllers.py --case all`.

| Scenario | Result | Notes |
| --- | --- | --- |
| **LQR stabilization** from a 0.05 rad perturbation | ✅ converges | settles to < 0.001 rad; max cart excursion 0.06 m |
| **Coupled swing-up + LQR handoff (x-axis)** from near-hanging (~160°) | ✅ swings up & balances | hands off to LQR at ~1.70 s; max cart excursion 2.15 m |
| **Coupled swing-up + LQR handoff (y-axis)** from near-hanging (~160°) | ✅ swings up & balances | symmetric to the x-axis case |

## Writeup

A full derivation of the dynamics and the small-angle linearization used by the
LQR is published [here](https://waterpancake.github.io/two_axis_cart_pole/)
(source in `docs/writeup.qmd`).

## Limitations & future work

- The coupled controller is an energy-shaping heuristic rather than a formally
  verified region-of-attraction controller.
- Fixed learned-controller scenarios do not establish broad random-start
  robustness.
- Possible extensions include a formal region-of-attraction study and broader
  learned-controller evaluation.

## Acknowledgments

The MuJoCo model is adapted from the
[inverted pendulum in Brax](https://github.com/google/brax/blob/main/brax/envs/assets/inverted_pendulum.xml)
by Google.

## License

[MIT](LICENSE)
