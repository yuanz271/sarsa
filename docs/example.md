# Example: Using SARSA End-to-End

This page shows a minimal end-to-end workflow for fitting SARSA.

## Overview

The library supports two modes:

- **Vanilla SARSA**: rewards are already stored in each `Quintuple.r2`
- **Extended SARSA**: rewards are computed on the fly via `transition_reward_func`

In both modes, state is composite: a length-`k` integer vector of discrete
state-factor indices, one entry per state factor. Action is flat: a single
integer index into the action axis.

## Step 1: Build quintuples

Construct transitions as `Quintuple(s1, a1, r2, s2, a2)`.

```python
import numpy as np
from sarsa import sarsa

quintuples = [
    sarsa.Quintuple(
        s1=np.array([0, 0], dtype=int),
        a1=1,
        r2=1.0,
        s2=np.array([0, 1], dtype=int),
        a2=0,
    ),
    sarsa.Quintuple(
        s1=np.array([0, 1], dtype=int),
        a1=0,
        r2=0.0,
        s2=np.array([1, 1], dtype=int),
        a2=1,
    ),
]

q0 = np.zeros((2, 2, 2))  # (*state_dims, n_actions)
```

## Step 2A: Vanilla SARSA

When rewards are already stored in `r2`, call `fit()` without a reward callback.

```python
params, loss, q_trajectory, action_prob = sarsa.fit(
    quintuples,
    q0=q0,
    p0=np.array([0.5, sarsa.DEFAULT_POLICY_BETA, 0.9]),
)
```

By default, `beta` is fixed (`fit_beta=False`) and `alpha`, `gamma` are optimized.
Set `fit_beta=True` if you want `beta` to be estimated.

## Step 2B: Extended SARSA

If transition reward depends on additional user-defined parameters, provide a
callback and bounds for those parameters.

```python
def transition_reward(user_params, s1, a1, s2):
    bonus = user_params[0]
    reward = bonus if s1[0] == 0 else 0.0
    return s2, reward

params, loss, q_trajectory, action_prob = sarsa.fit(
    quintuples,
    q0=q0,
    p0=np.array([0.5, sarsa.DEFAULT_POLICY_BETA, 0.9, 0.2]),
    transition_reward_func=transition_reward,
    user_param_bounds=[(0.0, None)],
)
```

## Step 3: Inspect outputs

```python
print(params)              # fitted flat parameter vector
print(loss)                # final cross-entropy loss
print(q_trajectory.shape)  # (T+1, *state_dims, n_actions)
print(action_prob.shape)   # (T, n_actions)
```

## See also

- [Algorithm](algorithm.md) — model equations and policy
- [Manual](manual.md) — full API reference
- [`src/sarsa/sarsa.py`](../src/sarsa/sarsa.py) — implementation details
