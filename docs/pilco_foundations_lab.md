---
title: "PILCO Foundations: Learning a Controller from a Probabilistic Model"
subtitle: "Theory and first principles for the two-axis cart-pole"
format:
  html:
    toc: true
    number-sections: false
    html-math-method: mathjax
execute:
  enabled: false
---

This is the **theory and fundamentals** half of the repository's two-part
PILCO lesson. The companion [implementation lab](pilco_code_lab.md) follows the
actual Python pipeline from transition collection to a saved controller.

The repository intentionally contains three control ideas:

1. **LQR** for local stabilization near the upright equilibrium;
2. **algorithmic energy swing-up** followed by LQR; and
3. a **PILCO-inspired learned controller** built from data.

The PILCO controller is evaluated as its own policy. It does not silently hand
off to LQR or the algorithmic swing-up controller.

## 1. What problem does PILCO solve?

Suppose the system state is

$$
s_t = [x, y, \theta_x, \theta_y,
       \dot{x}, \dot{y}, \dot{\theta}_x, \dot{\theta}_y]^\top
$$

and the two control inputs are

$$
u_t = [F_x, F_y]^\top.
$$

We want a policy $u_t = \pi_\psi(s_t)$ with parameters $\psi$ that minimizes

$$
J(\psi) = \mathbb{E}\left[
  \sum_{t=0}^{H-1} \gamma^t c(s_t, u_t)
  + \gamma^H c_T(s_H)
\right].
$$

The true transition function $s_{t+1}=f(s_t,u_t)$ is not given to the learner.
MuJoCo can generate transitions, but the policy should improve from a limited
dataset rather than search entirely through expensive simulator rollouts.

## 2. The model-based policy-search loop

PILCO's central idea is:

```text
interact with system
        ↓
collect (state, action, next state)
        ↓
fit a probabilistic dynamics model
        ↓
optimize a policy inside that model
        ↓
run the improved policy on the system
        ↺
```

The probabilistic model matters because a policy optimizer will otherwise
exploit unsupported predictions. A useful model should say both what it
expects and how uncertain that prediction is.

## 3. Representing angles continuously

Raw angles wrap at $-\pi$ and $\pi$. Numerically, $\pi-\epsilon$ and
$-\pi+\epsilon$ appear far apart even though they represent almost the same
pose. The learner therefore uses

$$
\phi(s) = [x, y,
\sin\theta_x, \cos\theta_x,
\sin\theta_y, \cos\theta_y,
\dot{x}, \dot{y}, \dot{\theta}_x, \dot{\theta}_y]^\top.
$$

Run the boundary experiment:

```bash
uv run python scripts/pilco_learning_lab.py features
```

It compares raw-angle distance with trigonometric-feature distance across the
wrap boundary.

## 4. Learn state changes, not absolute next states

The model predicts

$$
\Delta s_t = s_{t+1} - s_t
$$

from $z_t=[\phi(s_t),u_t]$. Then

$$
\hat{s}_{t+1} = s_t + \widehat{\Delta s_t}.
$$

Angle differences are wrapped back into $[-\pi,\pi]$. Delta prediction is
usually easier because the dominant identity mapping is handled exactly.

## 5. Gaussian-process dynamics

The implementation fits one independent Gaussian process for each of the eight
state-delta outputs. For normalized training inputs $Z$, targets $Y$, kernel
matrix $K$, and noise $\sigma_n^2$, the predictive mean at $z_*$ is

$$
\mu(z_*) = k_*^\top (K + \sigma_n^2 I)^{-1}Y,
$$

and the predictive variance is

$$
\sigma^2(z_*) = k(z_*, z_*)
 - k_*^\top (K + \sigma_n^2 I)^{-1}k_*.
$$

The radial-basis kernel is

$$
k(z_i,z_j) = \exp\left(
  -\frac{1}{2}\left\|\frac{z_i-z_j}{\ell}\right\|^2
\right).
$$

Nearby state-action pairs strongly influence one another; distant queries have
less support and usually higher uncertainty.

Positions, angles, velocities, and actions have different scales, so inputs
and outputs are standardized before forming the kernel. Exact GP fitting grows
cubically with the number of points; `model_max_points` caps the retained data.

## 6. The RBF policy

The policy uses radial basis functions over continuous state features:

$$
b_j(s) = \exp\left(
  -\frac{1}{2}\left\|\frac{\tilde\phi(s)-c_j}{\ell_\pi}\right\|^2
\right).
$$

The actions are

$$
\pi_\psi(s) = u_{\max}\tanh\left(B(s)W+b\right).
$$

The hyperbolic tangent enforces the action limit smoothly. The trainable
parameters are only the RBF weights $W$ and bias $b$.

## 7. The swing-up objective

The running cost combines several goals:

$$
c(s,u) =
w_p\lVert[x,y]\rVert^2
+ w_\theta\sum_i (1-\cos\theta_i)
+ w_v\lVert[\dot{x},\dot{y}]\rVert^2
+ w_\omega\lVert[\dot\theta_x,\dot\theta_y]\rVert^2
+ w_u\lVert u\rVert^2
+ c_{\text{rail}}.
$$

The angle term is periodic and equals zero at upright. The rail penalty
discourages imagined solutions that send the cart outside its useful workspace.
A terminal multiplier makes the final predicted state especially important.

## 8. Imagined rollouts and uncertainty

Starting from $s_0$, the learner applies

$$
u_t = \pi_\psi(s_t), \qquad
s_{t+1} = s_t + \mu_f(s_t,u_t).
$$

The implementation uses GP mean rollouts and can add a variance penalty:

$$
\tilde c_t = c(s_t,u_t) + \lambda_\sigma
\sum_d \sigma_d^2(s_t,u_t).
$$

This is simpler than original PILCO's analytic moment matching. The accurate
name for this repository is therefore **PILCO-inspired**. Small one-step errors
also compound, so good held-out one-step error does not guarantee a good
multi-step rollout.

## 9. Cross-entropy policy search

The cross-entropy method (CEM) maintains a Gaussian search distribution over
policy parameters:

1. sample a population of parameter vectors;
2. evaluate each vector on model rollouts;
3. retain the lowest-cost elite set;
4. update the search mean and standard deviation from the elites;
5. repeat.

CEM is derivative-free and inspectable. It replaces original PILCO's
gradient-based policy optimization while preserving model-based policy search.

## 10. Data aggregation

After every policy update, the new policy runs in MuJoCo and its transitions are
appended. This corrects the model in regions the current policy visits. Random
exploration is necessary at the beginning because the zero-initialized policy
cannot create a useful first dataset by itself.

## 11. What the evidence means

The learned policy is reported separately from the two classical controllers:

| Result | Defensible interpretation |
| --- | --- |
| LQR balances a small perturbation | the local linear controller works near upright |
| algorithmic swing-up reaches LQR | the hand-designed hybrid controller works |
| PILCO policy lowers imagined cost | optimization succeeded inside the learned model |
| PILCO policy succeeds in MuJoCo | the learned policy transferred to the simulator |

A smoke run verifies wiring, serialization, and finite simulation. Its tiny
budget is not expected to solve swing-up.

## 12. Experiments

Collect a small dataset:

```bash
uv run python scripts/pilco_learning_lab.py collect \
  --rollouts 4 --steps 80 \
  --output artifacts/pilco_foundations_data.npz
```

Measure one-step model error:

```bash
uv run python scripts/pilco_learning_lab.py model \
  --dataset artifacts/pilco_foundations_data.npz --holdout 0.25
```

Compare model and MuJoCo rollouts:

```bash
uv run python scripts/pilco_learning_lab.py compare \
  --dataset artifacts/pilco_foundations_data.npz --steps 60
```

Run the complete path quickly:

```bash
uv run python scripts/train_pilco_controller.py --smoke \
  --output artifacts/pilco_smoke.npz

uv run python scripts/evaluate_pilco_controller.py \
  --artifact artifacts/pilco_smoke.npz --assert-finite
```

## 13. Checkpoint questions

1. Why do sine/cosine features help both the GP and policy?
2. Why is low one-step error insufficient evidence for control?
3. What failure can an uncertainty penalty discourage?
4. Why append policy-generated transitions after every iteration?
5. What does a successful smoke run establish—and what does it not establish?
6. Which parts of this implementation differ from original PILCO?

The implementation lab answers these questions with the exact arrays and
functions used by the repository.
