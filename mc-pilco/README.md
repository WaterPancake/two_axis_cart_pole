# MC-PILCO for the Two-Axis Cart-Pole

This directory is a standalone PyTorch implementation of MC-PILCO for the
repository's MuJoCo two-axis cart-pole. It reads the existing plant through
`envs.TwoAxisInvertedPendulum`; source, tests, caches, datasets, metrics, and
checkpoints produced by this implementation remain under `mc-pilco/`.

Robust deployment convexly blends the learned policy with the repository's
energy-shaping/LQR hybrid. This empirically tested fallback is intentional: the
learned policy can contribute without being solely responsible for global
capture or perturbation recovery. Setting the residual scale to zero uses the
fallback alone; scales up to `0.25` retain at least 75% fallback authority.

## Method

The implementation uses the defining MC-PILCO policy-search path:

1. Collect aligned `(state, applied action, next state)` transitions from MuJoCo.
2. Fit five independent exact Gaussian processes to cart and pole tangent-velocity increments.
3. Propagate state particles through reparameterized GP posterior samples.
4. Backpropagate expected discounted swing-up cost through the sampled dynamics.
5. Execute the optimized bounded policy in MuJoCo and aggregate new transitions.
6. Repeat model fitting and policy search, then save a direct-deployment policy.

States use the physical convention
`[x, y, theta_x, theta_y, x_dot, y_dot, theta_x_dot, theta_y_dot]`. GP inputs use
manifold features and normalized motor commands in `[-1, 1]^2`. The pole is
represented internally by its 3D unit direction and tangent velocity, avoiding
the public ball-joint chart fold. Actions are held for five 0.002 s MuJoCo steps
by default, giving a 0.01 s model interval. The model predicts two cart-velocity
and three pole tangent-velocity changes, then integrates on the unit sphere.

This is MC-PILCO, not the sibling deterministic GP-mean/CEM baseline. GP posterior
samples are formed as `mean + sqrt(variance) * epsilon`, and policy parameters are
updated with gradients through every imagined transition.

The frozen standalone benchmark below is a practical tempered-uncertainty variant:
its first 5.76 million imagined transitions use 15% of posterior standard
deviation and its final 0.96 million use 10%. The legacy implementation above
uses the full posterior scale.

## Setup

From this directory, use the repository environment, which already provides the
sibling simulator package:

```bash
../.venv/bin/python -m pip install -e .
```

Alternatively, run directly with `PYTHONPATH=.` and the repository virtual
environment. The adapter locates the sibling `envs` source without modifying it.

## Train

```bash
../.venv/bin/python -m mc_pilco train
```

The default output is `artifacts/latest/`:

- `checkpoint.pt`: policy, GP state, and experiment configuration;
- `transitions.npz`: all collected physical transitions and controller interval;
- `metrics.json`: model RMSE, imagined costs, real rollout costs, and scenario results.

For a quick pipeline check:

```bash
../.venv/bin/python -m mc_pilco train \
  --initial-rollouts 2 --trial-rollouts 1 --rollout-steps 10 \
  --iterations 1 --max-gp-points 20 --gp-steps 2 \
  --policy-updates 2 --particles 8 --imagination-horizon 5 \
  --output-dir artifacts/smoke
```

Longer horizons, more particles, and more policy updates improve the optimization
signal but make exact-GP rollouts substantially more expensive.

## Evaluate

Evaluation runs only the saved neural policy in MuJoCo; it does not need GP
sampling or online trajectory optimization.

```bash
../.venv/bin/python -m mc_pilco evaluate artifacts/latest/checkpoint.pt --steps 400
```

Direct policy evaluation does not apply the safety envelope. Use the robust
acceptance command before deployment:

```bash
../.venv/bin/python -m mc_pilco robust-evaluate \
  --checkpoint artifacts/latest/checkpoint.pt \
  --residual-scale 0.1 --assert-pass
```

The protocol runs 30-second x-axis, y-axis, diagonal, and uniformly random
starts. A pass requires capture and centering for two continuous seconds, rail
survival, at least 80% of every full 0.2-second pulse to affect the applied action,
then a new two-second recovery dwell after the final pulse. Pulses correspond
to W, A, S, D, and two seeded random diagonal disturbances. Thresholds are:

- geodesic upright error below `0.20 rad`;
- pole tangent speed below `1.0 rad/s`;
- each cart coordinate below `0.50 m`;
- cart speed below `0.75 m/s`;
- maximum cart excursion below `9.0 m`.

The machine-readable report is saved to `artifacts/robust-acceptance.json`.
The checked composite controller passed x, y, diagonal, and four random starts.
This validates the fallback-plus-learned blend, not the learned policy in
isolation; `evaluate` remains the standalone-policy measurement.

## Interactive Disturbances

On macOS, launch the MuJoCo viewer with `mjpython`:

```bash
mjpython -m mc_pilco interactive \
  --scenario random \
  --checkpoint artifacts/latest/checkpoint.pt \
  --residual-scale 0.1
```

WASD adds a decaying disturbance command while the robust controller remains
active. Space clears the current disturbance. Without `--checkpoint`, the same
command runs the empirically tested global swing-up/recovery fallback.

## Browser Control Lab

Launch the NumPy-inspired real-time website from this directory:

```bash
../.venv/bin/python -m mc_pilco web
```

The server opens `http://127.0.0.1:8765` and automatically loads
`artifacts/latest/checkpoint.pt`, falling back to `artifacts/smoke/checkpoint.pt`
when available. The interface provides:

- MC-PILCO direct, MC-PILCO plus fallback, energy/LQR, and manual controllers;
- frozen standalone MC-PILCO and PPO benchmark controllers;
- upright, x-axis, y-axis, diagonal, downward, and random-sphere scenarios;
- a live isometric plant view, response trace, state arrays, and control telemetry;
- keyboard and touch controls using WASD or arrow keys.

Any held movement key gives manual input exclusive authority. The selected
controller does not contribute while a key is held and resumes immediately once
all movement keys are released. A 300 ms input timeout also releases manual
authority if the browser disconnects while sending a command.

To use an explicit artifact or avoid opening a browser automatically:

```bash
../.venv/bin/python -m mc_pilco web \
  --checkpoint artifacts/latest/checkpoint.pt \
  --port 9000 --no-open
```

## Standalone MC-PILCO vs PPO

The benchmark controllers do not call the classical fallback at runtime. Both
receive the same zero-interaction local stabilizing prior by cloning LQR actions
on synthetic states. MC-PILCO then trains a shared axis-equivariant residual
through stochastic GP particles. PPO uses a `64x64` residual actor and `64x64`
critic from Stable-Baselines3, initialized with 20 expert trials, five DAgger
rounds, and demonstration rehearsal during PPO updates.

The primary task mixes local two-axis stabilization with a basic x-axis swing-up.
Training trials last 10 seconds at 50 Hz. Frozen policies are evaluated for 30
seconds and must remain upright and centered for the final two seconds.

| Metric | Standalone MC-PILCO | PPO |
|---|---:|---:|
| Real transitions | 8,828 | 100,500 |
| Equivalent 10 s trials | 17.7 | 201 |
| Completed training trials | 20 | 236 |
| Deployment parameters | 2,755 | 6,468 |
| Stabilization success | 100% | 100% |
| Basic swing-up success | 84% | 96% |
| Median successful capture | 4.91 s | 7.43 s |
| Y-axis transfer | 80% | 0% |
| Diagonal transfer | 100% | 20% |
| Two-step-delay swing-up | 65% | 30% |
| Observation-noise swing-up | 75% | 95% |
| Joint pulse capture + recovery | 70% | 100% |
| Recovery conditional on capture | 100% | 100% |
| Measured optimization phase | about 352 s | 67 s on-policy/rehearsal only |

The observed model-based benefit in this selected run is plant-sample efficiency:
the structured MC-PILCO variant used about 11 times fewer real transitions. Its
shared axis residual also transferred to rotated starts. This is not an
algorithm-only causal comparison because that symmetry is an MC policy inductive
bias. PPO now has higher nominal and noise/pulse robustness, while MC-PILCO has
faster capture and much stronger rotated-axis transfer. MC's cost is substantially
more computation per real transition: 6.72 million imagined particle transitions.

Important limitations: this is a selected single-seed development result, not a
confidence interval. Both methods start from the same learned local neural prior,
but PPO additionally uses counted expert trajectories, DAgger, and rehearsal. MC has
a structured actor, uses an 8-second model objective, and tempers GP uncertainty;
PPO uses a separately shaped 10-second reward and a larger gated logit-residual
actor. MC training terminates at 4.8 m, PPO on-policy training at 9 m, and PPO
expert collection at the simulator's approximately 9.95 m limit. Final evaluation
uses 9 m for both. The reported PPO wall time excludes expert collection, DAgger,
and supervised initialization. Excluded
pilot tuning consumed another 37,775 MC and 2,030,533 PPO interactions. These facts,
checkpoint hashes, and lineage metadata are recorded in
`artifacts/benchmark/comparison.json`.

Reproduce the evaluations:

```bash
../.venv/bin/python -m mc_pilco benchmark
```

Train fresh checkpoints with the current frozen defaults. These commands do not
recreate the published MC checkpoint's historical two-stage resume lineage:

```bash
../.venv/bin/python -m mc_pilco train-mc-benchmark --seed 29
../.venv/bin/python -m mc_pilco train-ppo-benchmark --seed 54 --transitions 100000
```

Install the optional PPO dependencies with `pip install -e '.[benchmark]'`.

## Single-Axis Aside

An isolated planar cart-pole benchmark uses the existing MuJoCo plant with the
y-axis fixed at zero. Both controllers map `[x, theta, x_dot, theta_dot]`
directly to one normalized x-axis action. They were trained end to end from
environment interaction, without expert trajectories, LQR initialization, or a
runtime fallback.

The frozen common-seed evaluation uses 50 stabilization and 50 swing-up trials,
15 seconds per trial, with a two-second final capture dwell:

| Metric | MC-PILCO | PPO |
| --- | ---: | ---: |
| Real training transitions | 14,826 | 501,760 |
| Imagined transitions | 8,304,000 | 0 |
| Stabilization success | 100% | 100% |
| Swing-up success | 96% | 98% |
| Median successful capture | 2.55 s | 5.60 s |

Train and reproduce the comparison from this directory:

```bash
../.venv/bin/python -m mc_pilco train-single-axis-mc
../.venv/bin/python -m mc_pilco train-single-axis-ppo
../.venv/bin/python -m mc_pilco single-axis-benchmark --episodes 50
```

On macOS, view either frozen standalone policy with MuJoCo:

```bash
mjpython -m mc_pilco single-axis-view --method mc-pilco --scenario swingup
mjpython -m mc_pilco single-axis-view --method ppo --scenario swingup
```

The report is saved to `artifacts/single-axis/comparison.json`.

## Tests

```bash
../.venv/bin/python -m pytest
../.venv/bin/ruff check .
```

The tests cover state conventions, structured integration, GP uncertainty and
query gradients, reparameterized rollout gradients, MuJoCo timing/action clipping,
a complete tiny train/save/load cycle, website/API serving, exclusive manual
authority, WASD input, and sustained swing-up, centering, and recovery across
every required scenario.

## Coordinates

MuJoCo uses a nonsingular ball joint internally, while the public repository API
returns a two-angle chart. The policy, cost, GP inputs, GP targets, and imagined
integration all use pole direction and tangent velocity, so equivalent chart
branches have the same representation and trajectories can cross the chart fold.
Conversion back to the 8D chart is retained only at the simulator and public API
boundaries.

## License

MIT. See [LICENSE](LICENSE).
