# Parameter handling refactor: `sarsa_params` + `user_params`

**Status:** draft  
**Branch:** `refactor-params`

## Summary

Refactor parameter handling so the flat optimizer vector is treated as the
concatenation of two explicit blocks:

\[
\theta = \operatorname{concat}(\theta_{\mathrm{SARSA}}, \theta_{\mathrm{user}})
\]

where:

- `sarsa_params` are the parameters consumed by the vanilla SARSA kernel
  (currently `alpha`, `beta`, `gamma`)
- `user_params` are all additional user-defined extension parameters

The key API change is that `transition_reward_func` will receive only
`user_params`, not the full flat parameter vector.

This removes the brittle dependency on absolute positions like `params[3]` and
`params[4]`, and decouples extension code from the current size of the vanilla
SARSA block.

---

## Motivation

Current design assumptions:

- `params[0]` = `alpha`
- `params[1]` = `beta`
- `params[2]` = `gamma`
- `params[3:]` = extension parameters

Problems with the current contract:

1. **Magic offsets leak into user code**
   - Example and tests use `params[3]`, `params[4]`
   - This couples extension code to the current core block size

2. **No isolation of ownership**
   - The vanilla SARSA kernel only consumes `alpha`, `beta`, `gamma`
   - User extensions still receive the entire vector

3. **Poor extensibility**
   - If the core SARSA block grows beyond 3 parameters, every extension callback
     that assumes `params[3:]` breaks

4. **The real abstraction boundary is hidden**
   - The meaningful split is not "reward params vs non-reward params"
   - It is "SARSA-owned params vs user-owned extension params"

---

## Design goals

1. Preserve the flat optimizer vector used by `scipy.optimize.minimize`
2. Make the `sarsa_params` / `user_params` split explicit and centralized
3. Remove absolute indexing like `params[3]` from first-party examples/tests
4. Ensure `transition_reward_func` depends only on user-defined extension state
5. Derive the SARSA block size from one canonical source rather than hard-coded `3`
6. Preserve vanilla SARSA behavior unchanged

---

## Non-goals

1. Generalize the vanilla SARSA kernel beyond `alpha`, `beta`, `gamma` in this patch
   - This refactor isolates the boundary
   - It does **not** add new core SARSA semantics

2. Introduce a named parameter container/dictionary for user parameters in this patch
   - Raw arrays are acceptable for the first refactor
   - Named user-parameter views can be a follow-up improvement

3. Support both old and new reward-callback signatures simultaneously
   - Old callback: `transition_reward_func(params, s1, a1, s2)`
   - New callback: `transition_reward_func(user_params, s1, a1, s2)`
   - These have identical positional arity, so automatic compatibility detection
     is not reliable

---

## Core decision

Keep a single flat parameter vector for optimization, but centralize packing and
unpacking through helper functions.

### New conceptual contract

- `sarsa_params`: parameters interpreted by vanilla SARSA internals
- `user_params`: opaque extension parameters passed to user callbacks

### New callback contract

```python
def transition_reward_func(user_params, s1, a1, s2) -> tuple[NDArray, float]:
    ...
```

The callback no longer receives `alpha`, `beta`, or `gamma`.

---

## Proposed API changes

### New helper functions

Add explicit pack/unpack helpers:

```python
def concat_params(
    sarsa_params: NDArray,
    user_params: Sequence[float] | NDArray = (),
) -> NDArray:
    ...


def split_params(params: NDArray) -> tuple[NDArray, NDArray]:
    ...
```

Properties:

- `concat_params` is the only public constructor for the flat parameter vector
- `split_params` is the canonical unpacking point
- The SARSA block size is derived from `len(SARSA_PARAM_BOUNDS)` (or the current
  canonical equivalent), not hard-coded as `3`

### Core function signatures

Refactor core internals to consume only `sarsa_params`:

```python
def action_logprob(sarsa_params: NDArray, v: NDArray) -> NDArray: ...
def update(sarsa_params: NDArray, quintuple: Quintuple, q: NDArray) -> tuple[NDArray, float]: ...
```

### `run()` behavior

Inside `run()`:

1. Split the flat vector into `(sarsa_params, user_params)`
2. Use `sarsa_params` for policy and TD update
3. Pass only `user_params` to `transition_reward_func`

### `fit()` naming

Rename:

- `custom_param_bounds` → `user_param_bounds`

Recommended transition strategy:

- accept `user_param_bounds` as the primary API
- optionally accept `custom_param_bounds` as a deprecated alias for one release
- raise if both are provided

### Constants / naming

Introduce:

- `SARSA_PARAM_BOUNDS`

Compatibility option:

- keep `PARAM_BOUNDS` as an alias during the transition

`ParamIndex` can remain unchanged in the first patch, since it already refers to
SARSA-owned parameters.

---

## Behavioral semantics

### Vanilla mode

If `transition_reward_func is None`:

- `run()` uses `Quintuple.r2` directly
- `user_params` must be empty in `fit()`
- the flat vector contains only `sarsa_params`

### Extended mode

If `transition_reward_func is not None`:

- `run()` passes `user_params` to the callback
- the callback returns `(s2, r2)`
- `sarsa_params` are still used only for policy / TD update

This preserves the current mathematical separation:

\[
\delta_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
\]

with:

- `alpha`, `beta`, `gamma` owned by the SARSA kernel
- `r_{t+1}` optionally generated from user-defined extension state

---

## Implementation plan

### Slice 1 — centralize parameter splitting

Files:

- `src/sarsa/sarsa.py`

Changes:

1. Add `concat_params()`
2. Add `split_params()`
3. Refactor `action_logprob()` to accept `sarsa_params`
4. Refactor `update()` to accept `sarsa_params`
5. Refactor `run()` to split once and pass only `user_params` to the callback

Expected effect:

- internal code no longer depends on a scattered implicit split
- callback boundary becomes explicit

### Slice 2 — rename extension vocabulary

Files:

- `src/sarsa/sarsa.py`
- `README.md`
- `examples/sarsa.ipynb`
- `tests/test_sarsa.py`

Changes:

1. Rename `custom_param_bounds` to `user_param_bounds`
2. Introduce `SARSA_PARAM_BOUNDS`
3. Keep compatibility aliases only if needed for transition

Expected effect:

- public naming matches the true ownership split

### Slice 3 — remove absolute extension indexing from examples/tests

Files:

- `examples/sarsa.ipynb`
- `tests/test_sarsa.py`

Changes:

1. Replace `params[3]`, `params[4]` with `user_params[...]`
2. Prefer local enums or named constants over raw `0`, `1`

Example:

```python
class UserParamIndex(IntEnum):
    shock = 0
    avoidance = 1
```

Expected effect:

- first-party code no longer depends on absolute offsets into the global vector

### Slice 4 — documentation cleanup

Files:

- `README.md`
- wiki/manual later if needed

Changes:

1. Replace wording like `params[3:]` with `user_params`
2. Clarify that the optimizer still sees one flat vector, but extension hooks do not

---

## Validation plan

### Unit / integration tests

Add or update tests for:

1. **Round-trip packing**
   - `split_params(concat_params(sarsa, user)) == (sarsa, user)`

2. **Vanilla run path**
   - `run()` with no callback uses only `sarsa_params`
   - finite `Quintuple.r2` still required

3. **Extended run path**
   - callback receives only `user_params`
   - reward recomputation still works
   - input quintuples remain unchanged

4. **Validation**
   - no callback + nonempty `user_param_bounds` raises `ValueError`
   - inconsistent total parameter length raises `ValueError`

5. **Regression**
   - current extended example/test behavior remains numerically consistent after
     updating the callback signature

### Static checks

- `uv run pytest tests/test_sarsa.py -v`
- `uvx ruff check src/sarsa/sarsa.py tests/test_sarsa.py`

---

## Risks and caveats

1. **Callback signature change is breaking**
   - Existing user code with `transition_reward_func(params, ...)` must be updated

2. **Concatenation does not eliminate splitting**
   - The goal is not “no slicing anywhere”
   - The goal is “split once in one canonical place”

3. **This refactor improves extensibility of the boundary, not the core algorithm semantics**
   - If a fourth SARSA-owned parameter is added later, it must still be consumed by
     some core function

---

## Acceptance criteria

This spec is satisfied when all of the following are true:

1. `transition_reward_func` receives only `user_params`
2. First-party examples/tests no longer use `params[3]`, `params[4]`
3. Core SARSA functions consume `sarsa_params`, not the full flat vector
4. The SARSA block size is derived from one canonical source rather than hard-coded `3`
5. Vanilla and extended tests both pass
6. README wording no longer describes the extension as `params[3:]`

---

## Follow-up ideas (not in this patch)

1. Named user-parameter containers instead of raw arrays
2. Multiple extension blocks instead of one `user_params` tail
3. A more general `ParamLayout` abstraction if the library grows beyond a two-block split
