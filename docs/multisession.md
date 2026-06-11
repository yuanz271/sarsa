# Multi-session pooled fitting

The `sarsa.multisession` module fits several sessions of a **single subject**
jointly so that a chosen subset of parameters is **shared** across sessions
(one point estimate) while the rest stay **session-specific** (one estimate per
session).

This replaces the per-session practice of fitting each session separately with
warm-started `Q`/parameters: instead, sessions are stacked in order, `Q` is
threaded across session boundaries, and one pooled cross-entropy objective is
optimized.

## Concepts

- **Subject input**: an ordered `list[Session]` where `Session = list[Quintuple]`
  (earliest session first). There is no special session type — you pass plain
  nested lists of quintuples.
- **Share mask**: a length-`P` boolean list over the base parameters
  `[alpha, beta, gamma, *user_params]` (`P = 3 + n_user_params`). `True` shares
  the parameter across sessions; `False` makes it session-specific. Defaults to
  all-shared.
- **Gap rule**: how `Q` is transformed between sessions (no TD update crosses a
  gap):
  - `"carry"` (default): `Q_start[s+1] = Q_end[s]`
  - `"decay"`: `Q_start[s+1] = gap_decay · Q_end[s]`
  - `"reset"`: `Q_start[s+1] = Q0`
- **`gap_decay`**: a fixed constant in `[0, 1]` used only by `"decay"`. It is
  **never estimated** and is conceptually distinct from `gamma` (γ is the
  within-session temporal discount; `gap_decay` is between-session forgetting).
- **Objective**: pooled cross-entropy over all stacked trials.

## Quick start

```python
import numpy as np
from sarsa import multisession as ms

# sessions: list of sessions, each a list of Quintuple
result = ms.fit_subject(
    sessions,
    q0=np.zeros((*state_dims, n_actions)),
    p0=np.array([0.4, 5.0, 0.9]),     # base vector [alpha, beta, gamma]
    share_mask=[True, True, True],    # share all three across sessions
)

result.shared_params      # fitted shared values (base order subset)
result.session_params     # per-session full base vectors
result.loss               # pooled cross-entropy
result.q_trajectories     # per-session Q trajectories
result.action_probs       # per-session action probabilities
```

## Partial sharing

Share the learning parameters but let the policy temperature vary by session:

```python
result = ms.fit_subject(
    sessions,
    q0=q0,
    p0=np.array([0.4, 3.0, 0.9]),
    share_mask=[True, False, True],   # beta session-specific
    fit_beta=True,                    # estimate beta
)

result.shared_params                                   # [alpha, gamma]
[p[1] for p in result.session_params]                  # per-session beta
```

## Gap handling

```python
# Forget half of Q between sessions:
result = ms.fit_subject(
    sessions, q0=q0, p0=p0,
    gap_rule="decay", gap_decay=0.5,
)

# Reset Q to q0 at each session start:
result = ms.fit_subject(sessions, q0=q0, p0=p0, gap_rule="reset")
```

## Extended (user-defined) rewards

Multi-session fitting supports the same reward-callback extension as
[`fit`](manual.md). User parameters follow the same `share_mask`, appended after
the canonical block:

```python
result = ms.fit_subject(
    sessions,
    q0=q0,
    p0=np.array([0.4, 3.0, 0.9, 0.2]),      # canonical + 1 user param
    share_mask=[True, True, True, True],     # share the user param too
    transition_reward_func=my_reward_func,
    user_param_bounds=[(0.0, None)],
)
```

## Parameter layout

The optimizer vector is laid out as:

```
[ shared values ] + [ session 0 specific ] + ... + [ session S-1 specific ]
```

Length `n_shared + S · n_spec`, where `n_shared` is the number of `True` entries
in `share_mask` and `n_spec = P − n_shared`. There is no fitted gap-decay term.

## API

### `fit_subject`

```python
ms.fit_subject(
    sessions, q0, p0,
    share_mask=None,                 # default: all shared
    static_params=None,
    transition_reward_func=None,
    user_param_bounds=(),
    *,
    fit_beta=False,
    policy_beta=sarsa.DEFAULT_POLICY_BETA,
    gap_rule="carry",
    gap_decay=1.0,
) -> SubjectFitResult
```

`p0` is either a length-`P` base vector (broadcast to every session) or a
`(n_sessions, P)` array of per-session base vectors. `static_params`,
`fit_beta`, and `policy_beta` apply at the base-parameter level: a fixed shared
parameter fixes its single value; a fixed session-specific parameter is
broadcast to all sessions.

### `run_subject`

```python
ms.run_subject(
    full_vec, sessions, q0, share_mask,
    *, gap_rule="carry", gap_decay=1.0, transition_reward_func=None,
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]
```

Returns per-session `(q_trajectories, log_probs, td_errors)`.

### `run_and_loss_subject`

```python
ms.run_and_loss_subject(
    full_vec, sessions, q0, share_mask,
    *, gap_rule="carry", gap_decay=1.0, transition_reward_func=None,
) -> float
```

Pooled cross-entropy over all stacked trials.

### `SubjectFitResult`

| Field | Type | Description |
|-------|------|-------------|
| `shared_params` | `NDArray` | Fitted shared base parameters (base-order subset) |
| `session_params` | `list[NDArray]` | Per-session full base vectors, length `P` each |
| `gap_decay` | `float` | Fixed gap-decay constant used |
| `loss` | `float` | Final pooled cross-entropy |
| `q_trajectories` | `list[NDArray]` | Per-session Q trajectories |
| `action_probs` | `list[NDArray]` | Per-session action probabilities |

## Model comparison

Because the per-session model nests the pooled model, its in-sample
cross-entropy is always lower. Use **BIC** (`k·ln N + 2·NLL`, with `N` the total
stacked trials) rather than AIC to compare: AIC's `2k` penalty can be overcome
by in-sample overfitting at moderate `N`, whereas BIC favors the simpler model
when the data are genuinely pooled. Compute the total negative log-likelihood as
`loss · N` (the loss is mean cross-entropy).

## See also

- [Manual](manual.md) — single-session `fit` API and parameter conventions
- [Algorithm](algorithm.md) — SARSA model and update rule
