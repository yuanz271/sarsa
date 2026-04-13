# Beta as a fixed policy hyperparameter by default

**Status:** implemented on `beta-hyperparam` (sensitivity sweep pending)  
**Branch:** `beta-hyperparam`

## Summary

Treat `beta` as a **policy hyperparameter** rather than a default fitted parameter.
By default, `fit()` should fix `beta` to a configurable, finite value that is large
enough to approximate greedy choice while keeping the objective smooth.

The optimization problem should then focus on:

- `alpha`
- `gamma`
- `user_params`

with `beta` only fit when the caller explicitly opts in.

This design has two goals:

1. reduce the `beta`–`gamma` identifiability issue in action-only fitting
2. move fixed parameters out of the optimizer search space entirely

---

## Motivation

### Identifiability

The softmax policy depends on the product of `beta` and the action values:

\[
\pi(a \mid s) \propto \exp(\beta Q(s, a))
\]

so action data directly constrain the scale of `beta · Q`, not `beta` and `Q`
separately.

In SARSA, `gamma` changes the scale and temporal propagation of `Q` via:

\[
\delta_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
\]

Therefore, if delayed rewards are present but temporal structure is limited,
`beta` and `gamma` can become weakly identifiable.

In the extended model, the user-defined reward parameters can also trade off with
`beta`, further weakening interpretability.

### Modeling stance

If the scientific target is **learning/valuation dynamics** rather than
**choice stochasticity**, then `beta` is better treated as a global policy-readout
hyperparameter than as a subject-level latent parameter.

---

## Core decision

### Default behavior

By default, `fit()` should **fix `beta`** and optimize only the remaining free
parameters.

### Optional behavior

Users may still choose to fit `beta`, but that should be an explicit opt-in.

### Interpretation

A fixed `beta` should be documented as:

- a **policy sharpness / scale-setting hyperparameter**
- not a fitted psychological parameter in the default model

---

## Design goals

1. Fix `beta` by default to reduce scale non-identifiability
2. Keep `beta` configurable for sensitivity analysis
3. Preserve the ability to fit `beta` when explicitly requested
4. Remove fixed parameters from the optimization subspace entirely
5. Keep `run()` and `run_and_loss()` operating on the full flat parameter vector
6. Make the optimization method explicit and reproducible

---

## Non-goals

1. Claim that fixing `beta` fully resolves `gamma` identifiability
   - it reduces one confound but does not create temporal information absent from
     the task

2. Remove `beta` from the SARSA parameter block
   - `beta` remains part of the model semantics and full parameter vector
   - it is only fixed by default during fitting

3. Treat the chosen numeric default for `beta` as provisional rather than final
   - the implementation should expose a named constant
   - downstream scientific use should still check sensitivity to the chosen fixed value

---

## Proposed API changes

### New fit-level hyperparameter

Add a high-level argument to `fit()`:

```python
def fit(
    quintuples,
    q0,
    p0,
    static_params=None,
    transition_reward_func=None,
    user_param_bounds=(),
    *,
    custom_param_bounds=None,
    fit_beta: bool = False,
    policy_beta: float = DEFAULT_POLICY_BETA,
):
    ...
```

### Semantics

- `fit_beta=False` (default): `beta` is fixed
- `fit_beta=True`: `beta` is optimized normally
- `policy_beta`: the fixed default value used when `fit_beta=False`

### Precedence rules

`static_params` remains the low-level explicit override.

Recommended precedence:

1. If `static_params[ParamIndex.beta]` is explicitly not `None`, use that value
2. Else if `fit_beta=False`, fix `beta = policy_beta`
3. Else if `fit_beta=True`, leave `beta` free

This preserves backward compatibility with explicit fixed-parameter workflows.

---

## Optimization refactor

This feature should be implemented together with a reduced-subspace optimizer.

### Problem with current fixed-parameter handling

Overwriting fixed coordinates inside the objective leaves dead dimensions in the
search space. For example, if `beta` is fixed, the optimizer still wastes effort
proposing updates to the `beta` coordinate even though those updates are ignored.

### Proposed helper functions

Add three optimization helpers:

```python
def select_trainable_params(full_params, static_params) -> NDArray: ...
def materialize_params(trainable_params, static_params) -> NDArray: ...
def select_trainable_bounds(full_bounds, static_params) -> list[tuple[float | None, float | None]]: ...
```

### New optimization flow

1. Construct the full parameter vector and full bounds
2. Apply default beta fixation through `static_params`
3. Extract only the trainable coordinates for `x0`
4. Optimize over the reduced subspace only
5. Reconstruct the full flat vector before calling `run()` / `run_and_loss()`

### Explicit optimizer selection

Set the optimizer explicitly:

```python
method="L-BFGS-B"
```

Rationale:
- current problem uses only box bounds
- no analytic Hessian / trust-region infrastructure is available
- explicit solver selection improves reproducibility

---

## Default beta choice

Introduce a named constant:

```python
DEFAULT_POLICY_BETA = 5.0
```

This branch implements `5.0` as a provisional default. A small sensitivity sweep
is still recommended before treating that value as scientifically settled.

Requirements for the chosen default:

- finite
- reasonably large
- approximates greedy choice
- does not saturate the softmax so severely that optimization becomes numerically brittle

The concrete numeric default should still be stress-tested with a small sensitivity
sweep, for example over a grid such as:

- low
- medium
- moderately high
- high

The exact values are an implementation detail and should not be scattered inline
throughout the code.

---

## Documentation changes

### README / wiki / notebook

Document the default model as:

- fitting `alpha`, `gamma`, and `user_params`
- using fixed `beta` by default as a policy hyperparameter
- allowing optional `beta` fitting for sensitivity analysis or research use cases

### Wording guidance

Recommended phrasing:

> We fix `beta` by default as a policy hyperparameter to approximate greedy action
> selection while preserving a smooth objective. This reduces scale
> non-identifiability and focuses inference on learning dynamics and user-defined
> valuation parameters.

---

## Validation plan

### Unit tests

Add tests for:

1. **Default beta fixation**
   - calling `fit()` with defaults does not optimize `beta`
   - returned `beta` equals `policy_beta`

2. **Opt-in beta fitting**
   - calling `fit(..., fit_beta=True)` leaves `beta` free

3. **Precedence rules**
   - explicit `static_params[beta]` overrides `policy_beta`
   - `fit_beta=True` and explicit fixed beta cannot silently disagree

4. **Reduced-subspace optimization**
   - fixed coordinates are absent from the optimized vector
   - the reconstructed full vector matches the requested fixed values

5. **All-fixed edge case**
   - if every parameter is fixed, skip `minimize` and evaluate the loss once

### Empirical sensitivity checks

Before finalizing the default beta value, compare a small grid of fixed values and
examine:

- predictive loss / cross-entropy
- stability of fitted `alpha`
- stability of fitted `gamma`
- stability of fitted `user_params`

Interpretation:
- if results are stable across a reasonable beta range, the hyperparameter choice
  is benign
- if results vary strongly with beta, then `gamma` / reward-scale interpretability
  remains weak and should be documented carefully

---

## Risks and caveats

1. Fixing `beta` removes one confound but does not guarantee `gamma` is fully identifiable
2. If subjects truly differ in policy stochasticity, fixing `beta` may force that
   variation into `gamma` or `user_params`
3. The absolute scale of user-defined reward parameters becomes more dependent on
   the chosen fixed beta value
4. This is a modeling assumption and should be made explicit in scientific use

---

## Acceptance criteria

This spec is satisfied when all of the following are true:

1. `fit()` fixes `beta` by default
2. `beta` remains configurable through a named hyperparameter
3. users can explicitly opt in to fitting `beta`
4. fixed parameters are removed from the optimization subspace
5. `L-BFGS-B` is selected explicitly in `fit()`
6. README / notebook / wiki describe beta as a default policy hyperparameter
7. tests cover default fixation, free-beta opt-in, and precedence rules

---

## Follow-up ideas (not in this patch)

1. Subject-level hyperparameter tuning or hierarchical treatment of `beta`
2. Alternative policy mappings beyond softmax
3. A more explicit distinction between latent-state dynamics parameters and observation-model parameters
