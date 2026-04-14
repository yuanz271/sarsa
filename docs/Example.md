# Example: Fitting SARSA to Behavioural Data

This page walks through the full workflow demonstrated in `examples/sarsa.ipynb`.

## Overview

The example fits SARSA to a single behavioural session (`examples/M1.csv`) from a
threat-conditioning task. The model learns three canonical SARSA parameters plus two
user-defined extension parameters. In the current fitting API, `beta` is fixed by
default as a policy hyperparameter unless you pass `fit_beta=True`.

| Parameter block | Role |
|----------------|------|
| `sarsa_params = (alpha, beta, gamma)` | Vanilla SARSA kernel |
| `user_params = (shock, avoidance)` | Task-specific extension state |

## Step 1: Define the reward function

Rewards are not directly observed in the data; instead, they are computed
on the fly from the current user-defined parameter block:

```python
from enum import IntEnum


class UserParamIndex(IntEnum):
    shock = 0
    avoidance = 1


def transition_reward(user_params, state, action, new_state):
    reward_value = 1.0
    shock_value = user_params[UserParamIndex.shock]
    escape_value = user_params[UserParamIndex.avoidance]
    val = 0.0

    if state[StateAxis.Loc] == Location.R and state[StateAxis.Light] > 0:
        val += reward_value          # liquid reward in reward zone

    if state[StateAxis.Tone] == 3:
        if state[StateAxis.Loc] == Location.P:
            val += escape_value      # successful avoidance
        else:
            val -= shock_value       # shock

    return new_state, val
```

## Step 2: Build quintuples

Each transition in the session is encoded as a `Quintuple(s1, a1, r2, s2, a2)`:

```python
quintuples = []
for t in range(len(behavior_data) - 2):
    s1 = row_to_state(behavior_data.iloc[t])
    s2 = row_to_state(behavior_data.iloc[t + 1])
    s3 = row_to_state(behavior_data.iloc[t + 2])
    quintuples.append(sarsa.Quintuple(
        s1=s1, a1=s2[StateAxis.Loc],
        r2=np.nan,                   # reward recomputed via callback
        s2=s2, a2=s3[StateAxis.Loc],
    ))
```

## Step 3: Fit the model

```python
import numpy as np
from sarsa import sarsa

USER_PARAM_BOUNDS = [
    (1.0, None),   # shock penalty >= 1.0
    (0.0, None),   # avoidance reward >= 0.0
]

q0 = np.zeros((*STATE_SPEC, ACTION_SIZE))
p0 = np.array([0.5, sarsa.DEFAULT_POLICY_BETA, 0.5, 1.5, 0.5])

params, loss, q_trajectory, action_prob = sarsa.fit(
    quintuples,
    q0=q0,
    p0=p0,
    static_params=None,
    transition_reward_func=transition_reward,
    user_param_bounds=USER_PARAM_BOUNDS,
    fit_beta=False,
)
```

By default this fits `alpha`, `gamma`, and `user_params` while keeping `beta`
fixed. Set `fit_beta=True` if you want to estimate `beta` explicitly.

If you want to make the split explicit in user code:

```python
params = sarsa.concat_params(
    sarsa_params=np.array([0.5, sarsa.DEFAULT_POLICY_BETA, 0.5]),
    user_params=np.array([1.5, 0.5]),
)
```

## Step 4: Inspect results

```python
print(params)         # fitted flat vector
print(loss)           # final cross-entropy loss
print(q_trajectory.shape)   # (T+1, *state_dims, n_actions)
print(action_prob.shape)    # (T, n_actions)

sarsa_params, user_params = sarsa.split_params(params)
print(sarsa_params)   # fitted (alpha, beta, gamma); beta is fixed unless fit_beta=True
print(user_params)    # fitted (shock, avoidance)
```

## Vanilla SARSA

If rewards are already recorded in the data, omit the reward callback entirely:

```python
params, loss, q_trajectory, action_prob = sarsa.fit(
    quintuples,   # each Quintuple must have finite r2
    q0=q0,
    p0=np.array([0.5, sarsa.DEFAULT_POLICY_BETA, 0.9]),
)
```

Again, `beta` is fixed by default here; use `fit_beta=True` if you want to
optimize it.

## Compatibility note

The current implementation also accepts `custom_param_bounds` as a deprecated
alias for `user_param_bounds`, but new code should use the `user_params` /
`user_param_bounds` terminology. Before `v0.3.0`, the reward callback received
`params` rather than `user_params`.

## See also

- [`examples/sarsa.ipynb`](../examples/sarsa.ipynb) — full runnable notebook
- [`examples/experiment.py`](../examples/experiment.py) — state construction and data processing helpers
- [Manual](Manual.md) — full API reference
