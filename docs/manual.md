# Manual

## Overview

The `sarsa` module provides a tabular SARSA implementation designed to be independent
of the interpretation of state and action.

It uses a **composite state** and **flat action** representation:
- each state is represented as a length-`k` integer NumPy array of discrete
  state-factor indices, one entry per state factor
- each action is a single integer index into the action axis

Equivalently, `q0` has shape `(*state_dims, n_actions)`.
For example, if `q0.shape == (3, 4, 4, 3)`, then a state like `s = [2, 1, 3]`
and action `a = 0` index the value `q0[2, 1, 3, 0]`.

---

## Parameter vector

The flat optimizer vector is structured as the concatenation of two blocks:

| Block | Meaning |
|-------|---------|
| `sarsa_params` | Parameters interpreted by the vanilla SARSA kernel |
| `user_params` | User-defined extension parameters consumed by callbacks |

The canonical SARSA block is currently:

| Index | Name | Role |
|-------|------|------|
| `sarsa_params[0]` | `alpha` | Learning rate |
| `sarsa_params[1]` | `beta` | Inverse temperature (softmax policy) |
| `sarsa_params[2]` | `gamma` | Discount / decay factor |

Use `ParamIndex` to access canonical SARSA indices by name:

```python
from sarsa import sarsa

sarsa.ParamIndex.alpha   # 0
sarsa.ParamIndex.beta    # 1
sarsa.ParamIndex.gamma   # 2
```

Preferred bounds constant:

```python
sarsa.SARSA_PARAM_BOUNDS
```

Backward-compatible alias:

```python
sarsa.PARAM_BOUNDS
```

Helper functions for packing and unpacking the flat vector:

```python
params = sarsa.concat_params(sarsa_params, user_params)
sarsa_params, user_params = sarsa.split_params(params)
```

**Important:** this refactor isolates user-defined extension state from the
vanilla SARSA kernel, but the kernel itself still canonically interprets
`alpha`, `beta`, and `gamma`.

Current `fit()` behavior: by default, `beta` is treated as a fixed policy
hyperparameter rather than an optimized coordinate. Use `fit_beta=True` to
estimate `beta` explicitly, or pass an explicit beta value through
`static_params` to override the default fixed value.

The default is `sarsa.DEFAULT_POLICY_BETA = 5.0`. That value was chosen as a
conservative large-yet-stable setting from an internal sensitivity sweep:
fits remained well behaved around `beta ≈ 5–8`, while `beta >= 12` showed
seed-dependent instability and occasional numerical warnings. Treat this as a
modeling default, not a universal constant, and rerun a sensitivity sweep for
materially different tasks or reward scales.

Canonical parameter domains now follow edge-safe box constraints:
- `alpha ∈ [0, 1]`
- `beta ∈ [0, ∞)`
- `gamma ∈ [0, 1)`

This means `alpha = 0`, `alpha = 1`, `beta = 0`, and `gamma = 0` are valid edge
cases, while exact `gamma = 1` remains excluded through the optimizer bound
`1 - EPS`. `fit()` warns when trainable canonical SARSA parameters land on
active bounds, since that often signals weak identifiability or conditioning.

---

## Data structures

### `Quintuple`

A single SARSA transition `(s1, a1, r2, s2, a2)`:

```python
@dataclass
class Quintuple:
    s1: NDArray   # state at time t (length-k integer vector of discrete state-factor indices)
    a1: int       # action taken at time t
    r2: float     # reward received at time t+1
    s2: NDArray   # state at time t+1 (length-k integer vector of discrete state-factor indices)
    a2: int       # action taken at time t+1
```

**Constraints:**
- `s1` and `s2` must be length-`k` integer NumPy arrays (composite state vectors)
- `a1` and `a2` are flat integer action indices into the last axis of `q0`
- Indices must be non-negative and within the bounds of `q0`
- In vanilla mode, `r2` must be finite
- In extended mode, `r2` may be `np.nan` (reward is recomputed by callback)

---

## Modes of operation

### Vanilla SARSA

Use stored `r2` directly from the data. No reward callback needed.

```python
params, loss, q_trajectory, action_prob = sarsa.fit(
    quintuples,
    q0=q0,
    p0=np.array([0.5, sarsa.DEFAULT_POLICY_BETA, 0.9]),
)
```

By default, this optimizes `alpha` and `gamma` while keeping `beta` fixed at
`policy_beta=sarsa.DEFAULT_POLICY_BETA`.

### Extended SARSA

Learn user-defined extension parameters jointly with the canonical SARSA block.
Provide a `transition_reward_func` and `user_param_bounds`.

```python
params, loss, q_trajectory, action_prob = sarsa.fit(
    quintuples,
    q0=q0,
    p0=np.array([0.5, sarsa.DEFAULT_POLICY_BETA, 0.9, 1.5, 0.5]),
    transition_reward_func=my_reward_func,
    user_param_bounds=[(1.0, None), (0.0, None)],
    fit_beta=False,
)
```

By default, this optimizes `alpha`, `gamma`, and `user_params`; set
`fit_beta=True` if you also want to fit `beta`.

Preferred reward callback signature:

```python
def my_reward_func(user_params, s1, a1, s2) -> tuple[NDArray, float]:
    ...
    return s2, reward
```

The callback must return `s2` unchanged.

Compatibility note:
- `custom_param_bounds` is still accepted as a deprecated alias in `fit()`
- before `v0.3.0`, the reward callback received the full flat parameter vector

---

## API Reference

### `fit`

```python
sarsa.fit(
    quintuples,
    q0,
    p0,
    static_params=None,
    transition_reward_func=None,
    user_param_bounds=(),
    *,
    custom_param_bounds=None,
    fit_beta=False,
    policy_beta=sarsa.DEFAULT_POLICY_BETA,
) -> tuple[NDArray, float, NDArray, NDArray]
```

Optimise SARSA parameters against observed quintuples using `scipy.optimize.minimize`.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `quintuples` | `list[Quintuple]` | required | Observed transitions |
| `q0` | `NDArray` | required | Initial Q-table; shape `(*state_dims, n_actions)` |
| `p0` | `NDArray` | required | Initial flat parameter guess |
| `static_params` | `list[float \| None] \| None` | `None` | Fix specific parameters during optimisation; explicit fixed values override the default beta hyperparameter handling and must satisfy declared bounds |
| `transition_reward_func` | `Callable \| None` | `None` | Reward callback; omit for vanilla SARSA |
| `user_param_bounds` | `Sequence[tuple]` | `()` | Preferred bounds for user-defined parameters; empty for vanilla SARSA |
| `custom_param_bounds` | `Sequence[tuple] \| None` | `None` | Deprecated alias for `user_param_bounds` |
| `fit_beta` | `bool` | `False` | If `False`, keep beta fixed as a policy hyperparameter; if `True`, optimize beta unless `static_params` already fixes it |
| `policy_beta` | `float` | `sarsa.DEFAULT_POLICY_BETA` | Fixed beta value used when `fit_beta=False` and beta is not already fixed in `static_params` |

**Returns:**

| Index | Type | Description |
|-------|------|-------------|
| 0 | `NDArray` | Optimised flat parameter vector |
| 1 | `float` | Final cross-entropy loss |
| 2 | `NDArray` | Q-trajectory; shape `(T+1, *state_dims, n_actions)` |
| 3 | `NDArray` | Action probabilities; shape `(T, n_actions)` |

**Raises:** `ValueError` if parameter lengths are inconsistent, both bounds names are supplied, explicit fixed parameters violate bounds, `policy_beta` is invalid, or user-defined parameters are requested without a reward callback.

**Optimization note:** `fit()` removes fixed parameters from the optimization
subspace before calling SciPy and uses `L-BFGS-B` explicitly for bounded
optimization. SciPy bounds are inclusive (`lb <= x <= ub`), so the meaningful
edge cases above are allowed directly; only exact `gamma = 1` stays excluded via
`1 - EPS`.

---

### `run`

```python
sarsa.run(
    params,
    quintuples,
    q0,
    transition_reward_func=None,
) -> tuple[NDArray, NDArray, NDArray]
```

Execute SARSA forward pass over a sequence of quintuples.

**Behavior:**
- `params` is always a flat vector
- internally it is split into `(sarsa_params, user_params)`
- `transition_reward_func`, when provided, receives only `user_params`

**Returns:**

| Index | Type | Description |
|-------|------|-------------|
| 0 | `NDArray` | Q-trajectory; shape `(T+1, *state_dims, n_actions)` |
| 1 | `NDArray` | Log-probabilities per timestep; shape `(T, n_actions)` |
| 2 | `NDArray` | TD errors per timestep; shape `(T,)` |

---

### `update`

```python
sarsa.update(sarsa_params, quintuple, q) -> tuple[NDArray, float]
```

Apply the SARSA TD update for a single transition:

```
delta = r2 + gamma * Q(s2, a2) - Q(s1, a1)
Q(s1, a1) <- Q(s1, a1) + alpha * delta
```

Returns the updated Q-table and the TD error `delta`.

---

### `action_logprob`

```python
sarsa.action_logprob(sarsa_params, v) -> NDArray
```

Compute softmax log-probabilities over actions:

```
log P(a | s) = log_softmax(beta * Q(s, :))
```

---

### `concat_params`

```python
sarsa.concat_params(sarsa_params, user_params=()) -> NDArray
```

Concatenate the SARSA-owned block and the user-defined block into one flat optimizer vector.

---

### `split_params`

```python
sarsa.split_params(params) -> tuple[NDArray, NDArray]
```

Split a flat optimizer vector into `(sarsa_params, user_params)`.

---

### `to_prob`

```python
sarsa.to_prob(p) -> NDArray
```

Convert log-probabilities to probabilities via `exp`.

---

### `merge`

```python
sarsa.merge(params, static) -> NDArray
```

Replace trainable parameter positions with fixed values where `static[i]` is not `None`.
Used internally to enforce `static_params` during optimisation.

---

## Constants

| Name | Value | Description |
|------|-------|-------------|
| `EPS` | `1e-8` | Small positive value used to keep the gamma upper bound strictly below `1` |
| `DEFAULT_POLICY_BETA` | `5.0` | Default fixed beta used by `fit()` when `fit_beta=False`; chosen as a conservative large-yet-stable value on the bundled example session |
| `SARSA_PARAM_BOUNDS` | `[(0.0, 1.0), (0.0, None), (0.0, 1-EPS)]` | Preferred bounds for the canonical SARSA block: `alpha ∈ [0, 1]`, `beta ∈ [0, ∞)`, `gamma ∈ [0, 1)` |
| `PARAM_BOUNDS` | alias of `SARSA_PARAM_BOUNDS` | Backward-compatible alias |
