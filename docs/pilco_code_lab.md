---
title: "PILCO Implementation Lab: From Transitions to a Learned Controller"
subtitle: "A code-first walkthrough of the two-axis cart-pole learner"
format:
  html:
    toc: true
    number-sections: false
    html-math-method: mathjax
execute:
  enabled: false
---

This is the **technical and code** half of the repository's two-part PILCO
lesson. Start with the companion [foundations lab](pilco_foundations_lab.md) if
Gaussian processes, model rollouts, or CEM are unfamiliar.

The implementation follows one path:

```text
random MuJoCo data → GP delta model → RBF policy → model CEM
                  → new MuJoCo data → saved .npz artifact
```

There is no expert imitation, recovery supervisor, real-rollout policy search,
or hidden LQR handoff. LQR and algorithmic swing-up remain separate classical
controllers.

## 1. Setup

Run from the repository root:

```bash
uv sync
uv run python scripts/pilco_learning_lab.py features
uv run python -m pytest -q tests/test_pilco_components.py
```

Preview this lesson with:

```bash
quarto preview docs/pilco_code_lab.md
```

Use a fresh artifact path for each experiment. Training outputs belong under
the ignored `artifacts/` directory.

## 2. Source map

| File | Responsibility |
| --- | --- |
| `controllers/pilco.py` | representations, cost, GP model, RBF policy, CEM, artifact I/O |
| `scripts/pilco_workflow.py` | resets, rollout collection, named evaluation, start sampling |
| `scripts/pilco_learning_lab.py` | small feature/model/rollout experiments |
| `scripts/train_pilco_controller.py` | complete iterative training loop |
| `scripts/evaluate_pilco_controller.py` | one reproducible headless rollout |
| `scripts/render_pilco_controller.py` | interactive viewer or off-screen GIF |
| `tests/test_pilco_components.py` | numerical and serialization regression tests |

The main call graph is:

```text
train_pilco_controller.py
├─ collect_rollouts(random_piecewise_policy)
├─ RBFPolicy.from_states
├─ repeat
│  ├─ GaussianProcessDynamicsModel.fit
│  ├─ optimize_policy_cem
│  │  └─ _population_rollout_costs
│  ├─ evaluate_policy_scenarios
│  └─ collect_rollouts(current policy)
├─ fit final GP model
└─ save_pilco_artifact
```

## 3. Shape and convention ledger

The public controller observation is

```text
[x, y, theta_x, theta_y, x_dot, y_dot, theta_x_dot, theta_y_dot]
```

| Quantity | Shape | Notes |
| --- | --- | --- |
| raw controller state | `(8,)` | public MuJoCo convention |
| physical state | `(8,)` | wrapped angles, symmetric y convention |
| policy features | `(10,)` | positions, trig angles, velocities |
| dynamics input | `(12,)` | policy features plus two actions |
| delta target | `(8,)` | wrapped `next_state - state` |
| action | `(2,)` | squashed by policy, clipped by environment |

All learning arrays use the physical convention. Only the controller boundary
accepts the raw observation.

```python
from controllers.pilco import physical_state_from_mujoco, policy_features

physical = physical_state_from_mujoco(raw_state)
features = policy_features(physical)
assert physical.shape == (8,)
assert features.shape == (10,)
```

## 4. Collecting transitions

`rollout_env` stores aligned arrays:

```text
states[i]       state before applying the action
actions[i]      action actually applied by the environment
next_states[i]  state after frame_skip physics steps
```

The same action is repeated for `frame_skip` physics ticks. The learned model
therefore predicts one **controller interval**, not one MuJoCo integration tick.
Training, evaluation, and rendering must use the same frame skip.

Create a small dataset:

```bash
uv run python scripts/pilco_learning_lab.py collect \
  --rollouts 4 --steps 80 --frame-skip 5 \
  --output artifacts/pilco_code_data.npz
```

Inspect it:

```bash
uv run python - <<'PY'
import numpy as np

with np.load("artifacts/pilco_code_data.npz") as data:
    for key in data.files:
        print(key, data[key].shape)
PY
```

Checkpoint: verify that the stored action is returned by `env.control`, not
merely the action requested by the policy.

## 5. Fitting the GP delta model

The model call is intentionally small:

```python
model = GaussianProcessDynamicsModel(
    length_scale=1.5,
    noise=1e-5,
    max_points=700,
)
model.fit(states, actions, next_states)

mean_delta, variance = model.predict_delta(
    states[0], actions[0], return_variance=True
)
```

Internally, fitting performs:

1. continuous state feature construction;
2. wrapped delta-target construction;
3. input/output standardization;
4. RBF-kernel matrix construction;
5. Cholesky factorization;
6. one coefficient solve for all eight outputs.

Run a held-out check:

```bash
uv run python scripts/pilco_learning_lab.py model \
  --dataset artifacts/pilco_code_data.npz --holdout 0.25
```

The reported RMSE is a one-step metric. It does not measure accumulated model
rollout error.

## 6. Comparing model and simulator rollouts

```bash
uv run python scripts/pilco_learning_lab.py compare \
  --dataset artifacts/pilco_code_data.npz --steps 60
```

The comparison begins from the same initial state and applies the same action
sequence to both the fitted GP and MuJoCo. Watch how angle and velocity errors
grow with horizon.

If the first few steps are wrong, check representation, coordinate convention,
frame skip, and action storage. If only long rollouts drift, collect more data
along the current policy's trajectories or shorten the optimization horizon.

## 7. The RBF policy in code

`RBFPolicy.from_states` samples normalized feature vectors as centers. With
$M$ centers and two actions, the trainable vector contains $2M+2$ values:

```python
policy = RBFPolicy.from_states(
    states, num_centers=12, rng=rng, action_limit=1.0
)
print(policy.parameters().shape)  # (26,)
```

For one state:

```text
physical state
  → 10D trig features
  → feature scaling
  → distance to every center
  → RBF activations
  → weighted sum plus bias
  → tanh action squash
```

`control_physical_batch` performs the same computation over a batch of states
and is used during policy optimization.

## 8. CEM inside the learned model

The optimizer receives a fitted model, an RBF policy template, and initial
states:

```python
optimized, result = optimize_policy_cem(
    model,
    policy,
    start_states,
    horizon=200,
    iterations=8,
    population=48,
    elite_fraction=0.2,
    rng=rng,
)
print(result.best_cost)
```

`_population_rollout_costs` evaluates every candidate and every start in one
batched array. Candidate parameters differ, while centers, feature scales, and
kernel hyperparameters are shared.

Keep two facts separate:

- `result.best_cost` is measured inside the learned model;
- `evaluate_policy_scenarios` measures the policy in MuJoCo.

A low model cost can coexist with poor MuJoCo behavior when the policy exploits
model error.

## 9. Run the full training path

First verify wiring with a smoke run:

```bash
uv run python scripts/train_pilco_controller.py --smoke \
  --output artifacts/pilco_smoke.npz

uv run python scripts/evaluate_pilco_controller.py \
  --artifact artifacts/pilco_smoke.npz \
  --scenario upright --steps 100 --assert-finite
```

The smoke budget is deliberately too small to establish controller quality.

A normal experiment is:

```bash
uv run python scripts/train_pilco_controller.py \
  --output artifacts/pilco_controller.npz \
  --iterations 3 \
  --random-rollouts 8 \
  --policy-rollouts 4 \
  --steps-per-rollout 250
```

Each iteration prints both model cost and real named-scenario metrics. The
policy is updated from model optimization, then its real transitions are added
to the next GP fit.

## 10. Artifact format

`save_pilco_artifact` writes a compressed NumPy archive containing:

```text
policy_centers
policy_weights
policy_bias
policy_length_scale
policy_action_limit
policy_feature_scale
model_* arrays
metadata_json
```

Load and inspect it:

```bash
uv run python - <<'PY'
from controllers.pilco import load_pilco_artifact

artifact = load_pilco_artifact("artifacts/pilco_controller.npz")
print(artifact.policy.centers.shape)
print(artifact.metadata)
PY
```

The artifact stores the frame skip and effective control interval in metadata.
Pass the same frame skip to evaluation and rendering.

## 11. Evaluate the learned policy

```bash
for scenario in upright swing-x swing-y diagonal; do
  uv run python scripts/evaluate_pilco_controller.py \
    --artifact artifacts/pilco_controller.npz \
    --scenario "$scenario" --steps 800
done
```

Report at least final angle/rate, mean cost, maximum cart excursion, and MuJoCo
warnings. Use `--assert-upright` only after the artifact actually meets the
documented final-state criterion.

The evaluator runs only the learned RBF policy. Success belongs to that policy;
failure cannot be masked by the classical controllers.

## 12. Render the same controller

Interactive viewer on macOS:

```bash
mjpython scripts/render_pilco_controller.py \
  --artifact artifacts/pilco_controller.npz \
  --mode viewer --scenario diagonal
```

Off-screen GIF:

```bash
uv run python scripts/render_pilco_controller.py \
  --artifact artifacts/pilco_controller.npz \
  --mode gif --scenario diagonal \
  --out media/pilco_diagonal.gif
```

On headless Linux, MuJoCo rendering may require `MUJOCO_GL=egl` or
`MUJOCO_GL=osmesa`.

## 13. Debugging map

| Symptom | First checks |
| --- | --- |
| discontinuity near $\pm\pi$ | angle wrapping and sine/cosine features |
| poor one-step GP error | feature scaling, delta targets, data coverage |
| good one-step error but drifting rollout | horizon length and on-policy data |
| action is nearly constant | RBF center coverage and parameter spread |
| action saturates | cost weights, CEM variance, action limit |
| low model cost but poor MuJoCo result | model exploitation and uncertainty penalty |
| artifact behaves at wrong speed | training/evaluation frame-skip mismatch |

## 14. Exercises

1. Double the number of RBF centers. Record parameter count, runtime, model
   cost, and real scenario cost.
2. Sweep GP length scale over three values while holding the dataset fixed.
3. Compare 40-step and 200-step model rollouts from the same transition.
4. Set a positive uncertainty weight and check whether selected policies visit
   lower-variance state-action regions.
5. Add held-out random starts to the evaluator without changing training.
6. Repeat one experiment across three seeds and keep every artifact separately.

Keep model metrics, imagined policy cost, and real MuJoCo behavior as three
separate measurements.
