# Spec 02 — Multi-session pooled fitting (one subject)

**Status:** draft / planning
**Branch:** `hierarchy`
**Depends on:** existing `src/sarsa/sarsa.py` (`run`, `update`, `fit`, param helpers)

## 1. Goal

Fit a single subject's multiple sessions jointly so that some parameters
(`alpha`, `beta`, `gamma`, and/or user parameters) are **shared** (one point
estimate) across sessions, while others may remain **session-specific**.

Mechanism (per user decision): stack sessions in order as one continuous
stream, thread `Q` across session boundaries, optionally transform `Q` at each
gap, and minimize one pooled cross-entropy objective.

This is **complete pooling** for shared parameters (no priors / no Bayesian
hierarchy in this iteration).

## 2. Decisions (locked)

- **D1 — masking:** user supplies a per-parameter `share_mask`; any subset of
  canonical + user parameters can be shared or session-specific.
- **D2 — pooling type:** single point estimate per shared parameter (complete
  pooling). No priors.
- **D3 — Q across gaps:** configurable rule, user-selectable.
- **D4 — boundary transitions:** no quintuple straddles two sessions; the gap
  only transforms `Q`, never produces a TD update.
- **D5 — reward mode:** support both vanilla (`r2`) and extended
  (`transition_reward_func` + `user_params`); user params follow the same mask.
- **D6 — loss:** plain pooled cross-entropy over all stacked trials.
- **D7 — scope:** one subject per call; no cross-subject parameters.

### Residual sub-decisions (all locked)

- **D3a gap rule:** `gap_rule ∈ {"carry", "decay", "reset"}`, default `"carry"`.
  - `carry`: `Q_start[s+1] = Q_end[s]`
  - `decay`: `Q_start[s+1] = d · Q_end[s]`
  - `reset`: `Q_start[s+1] = Q0`
- **D3b decay value:** `gap_decay` is **always a fixed float** in `[0, 1]`,
  default `1.0`. The gap-decay parameter is **never estimated** — no `d` ever
  enters the optimizer vector. `d` is conceptually separate from `gamma`
  (γ = within-session temporal discount; `d` = between-session forgetting), but
  is a user-supplied constant, not a fitted quantity.
- **D-module:** new module `src/sarsa/multisession.py`; public API
  `run_subject`, `run_and_loss_subject`, `fit_subject`. No edits to existing
  `fit`/`run` (pure addition, backward compatible).
- **D-fixed-vs-mask:** `static_params` / `fit_beta` / `policy_beta` apply at the
  **base-parameter** level. A fixed shared parameter fixes its single value; a
  fixed session-specific parameter is broadcast to all sessions in v1
  (per-session fixed values deferred).
- **D-input shape:** `Session = list[Quintuple]` (type alias, not a dataclass);
  subject input is `list[Session]` = `list[list[Quintuple]]`, ordered earliest
  first. No user-facing session wrapper.
- **D-p0:** accept a single base `p0` (broadcast to sessions) **or** a list of
  per-session base vectors.

## 3. Data model

No change to `Quintuple`. Represent a subject as an ordered list of sessions:

```python
Session = list[Quintuple]                 # type alias only, NOT a dataclass
Subject = list[Session]                   # ordered sessions for one subject
```

`Session` is a plain type alias for readability; users only ever pass nested
lists of quintuples. Sessions are assumed pre-ordered (earliest first). A
metadata-carrying session type is explicitly out of scope for v1.

## 4. Parameter layout

Let:
- `P = 3 + U` base parameters (canonical α,β,γ then `U` user params),
- `S` = number of sessions,
- `share_mask` = length-`P` booleans (`True` = shared),
- `n_shared` = count of `True`, `n_spec = P − n_shared`.

**Optimizer (full) vector layout, in order:**

```
[ shared values        ]   # length n_shared, base order
[ session 0 spec values]   # length n_spec
[ session 1 spec values]   # length n_spec
...
[ session S-1 spec vals ]  # length n_spec
```

Total length `L = n_shared + S·n_spec`. (No `d`: gap decay is never fitted.)

**Scatter** `theta_s` (full base vector for session `s`) is reconstructed by
placing shared values at their masked positions and session `s`'s specific
values at the remaining positions. Helper:

```python
def scatter_session_params(full_vec, share_mask, S) -> list[NDArray]:
    """Return [theta_0, ..., theta_{S-1}], each length P."""
```

Inverse **gather** packs `[theta_s]` (+ optional d) into the optimizer vector,
used to build `p0`.

## 5. Bounds, static, trainable subspace

- Base bounds: `SARSA_PARAM_BOUNDS + list(user_param_bounds)` (length `P`),
  reusing the existing constants.
- Expanded bounds follow the layout: shared positions once, then each session's
  specific positions repeated. No `d` entry (gap decay is a fixed constant).
- Resolve base static via existing `resolve_static_params(..., fit_beta,
  policy_beta)`; validate via `validate_fixed_params_against_bounds`.
- Expand base static to a full-length `static` vector matching the optimizer
  layout (shared fixed once; session-specific fixed broadcast to all sessions).
- Reduce to trainable subspace with the existing
  `select_trainable_params` / `select_trainable_bounds` /
  `materialize_params`. **These helpers are reused verbatim** — they are layout
  agnostic (operate on flat vectors + a `None`-mask).

## 6. Algorithms

### 6.1 `run_subject`

```python
def run_subject(
    full_vec, sessions, q0, share_mask,
    *, gap_rule="carry", gap_decay=1.0,
    transition_reward_func=None,
) -> SubjectRun:
    thetas = scatter_session_params(full_vec, share_mask, S)
    d = gap_decay   # always a fixed constant
    q = q0
    per_session_qs, per_session_logprob, per_session_error = [], [], []
    for s, session in enumerate(sessions):
        qs, logprob, err = run(thetas[s], session, q, transition_reward_func)
        per_session_qs.append(qs)
        per_session_logprob.append(logprob)
        per_session_error.append(err)
        q_end = qs[-1]
        q = gap_transform(q_end, q0, gap_rule, d)   # carry / decay / reset
    return SubjectRun(per_session_qs, per_session_logprob, per_session_error)
```

- `run` is reused per session unchanged (handles both reward modes, validation,
  per-step logprobs).
- `gap_transform`: `carry→q_end`, `decay→d*q_end`, `reset→q0`.
- No cross-session quintuple is constructed (D4).

### 6.2 `run_and_loss_subject`

```python
def run_and_loss_subject(full_vec, sessions, q0, share_mask, *, ...):
    run_out = run_subject(...)
    actions  = concat([a1 of each quintuple, across sessions in order])
    logprob  = concat(per_session_logprob)
    return float(cross_entropy(logprob, actions))   # pooled over all trials
```

Pooled CE over the stacked trials = D6. (Implemented as one `cross_entropy`
call on concatenated arrays; mean-vs-sum is a constant factor over a fixed
dataset and does not change the argmin.)

### 6.3 `run_and_loss_subject_trainable`

Mirror of `run_and_loss_trainable`: materialize full vector from trainable
coordinates + static, then call `run_and_loss_subject`.

### 6.4 `fit_subject`

```python
def fit_subject(
    sessions, q0, p0, share_mask,
    static_params=None, transition_reward_func=None,
    user_param_bounds=(), *,
    fit_beta=False, policy_beta=DEFAULT_POLICY_BETA,
    gap_rule="carry", gap_decay=1.0,
) -> SubjectFitResult:
    # 1. validate sessions (each via _validate_quintuples against q0)
    # 2. base bounds/static via existing resolvers
    # 3. build full p0 (broadcast base p0 or accept per-session list) + optional d
    # 4. expand bounds + static to optimizer layout
    # 5. reduce to trainable subspace; L-BFGS-B on run_and_loss_subject_trainable
    # 6. materialize; run_subject for trajectories
    # 7. boundary warnings on shared canonical params (+ per-session if desired)
    # 8. return shared params, per-session params, d, per-session q-traj,
    #    per-session action_prob, pooled loss
```

Keep `method="L-BFGS-B"` and the INFO logging pattern from `fit`.

## 7. Return structure

```python
@dataclass
class SubjectFitResult:
    shared_params: NDArray           # length n_shared (base order subset)
    session_params: list[NDArray]    # per session, full base vector length P
    gap_decay: float                 # fitted or fixed d
    loss: float                      # pooled CE
    q_trajectories: list[NDArray]    # per session, (T_s+1, *state_dims, A)
    action_probs: list[NDArray]      # per session, (T_s, A)
```

## 8. Edge cases & validation

Invariants to assert in tests:

1. **Reduces to `fit`:** `S=1`, `share_mask=all True`, `gap_rule="carry"` →
   identical params/loss to `fit()` on that session.
2. **Stacking equivalence:** two sessions, all-shared, `gap_rule="carry"`,
   `d=1` → `run_subject` Q-thread and pooled loss equal `run()` on the
   concatenated quintuple list (continuous Q, no boundary update).
3. **Reset:** `gap_rule="reset"` with `S=1` equals `fit()`.
4. **Mask:** session-specific `beta` recovers two distinct betas on synthetic
   data generated with different betas but shared α,γ.
5. **Decay identifiability:** small sweep of true `d` recovered within tolerance
   on synthetic data; flag if `d` trades off with α/γ.
6. **Recovery + comparison:** simulate from known shared params across `N`
   sessions; pooled fit recovers them and beats per-session fits by AIC/BIC.

Other checks:
- empty subject / empty session → `ValueError`.
- inconsistent state dims across sessions vs `q0` → `ValueError` (reuse
  `_validate_quintuples` per session).
- `share_mask` length ≠ `P` → `ValueError`.
- `gap_decay` float outside `[0,1]` → `ValueError`. `gap_decay` is only
  meaningful when `gap_rule="decay"` (ignored otherwise).

## 9. Implementation order

1. `scatter_session_params` / gather helpers + unit tests.
2. `gap_transform` + `run_subject` + invariants (1)(2)(3).
3. `run_and_loss_subject` (+ trainable wrapper).
4. `fit_subject` (bounds/static expansion, optimizer, return struct).
5. Mask + decay tests (4)(5).
6. Synthetic recovery + AIC/BIC comparison (6).
7. `notebooks/` demo on this branch; `docs/` page later (on `main` at merge).

## 10. Status

All design decisions locked. Ready to implement per the order in §9.

Deferred (not in v1): per-session fixed values, per-gap decay, fitted gap
decay, `Session` metadata type, cross-subject parameters.
```
