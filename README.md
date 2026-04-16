# SARSA

SARSA toolkit for fitting behavioural datasets and running reproducible experiments.

## Installation

### Install latest release

```bash
pip install git+https://github.com/yuanz271/sarsa.git@v0.4.0
```

### Install development version

```bash
pip install git+https://github.com/yuanz271/sarsa.git
```

## Example

See [`docs/example.md`](docs/example.md) for an end-to-end usage walkthrough,
including vanilla and extended `fit()` usage patterns.

## Model

This library implements tabular on-policy SARSA with a softmax policy.
It uses a **composite state** and a **flat action** representation: state is
represented as a length-k integer vector of discrete state-factor indices, one
entry per state factor; action is a single integer index into the action axis.
Equivalently, `Q` has shape `(*state_dims, n_actions)`.

For example, if `Q.shape == (3, 4, 4, 3)`, then a state like `s = [2, 1, 3]`
and action `a = 0` index the value `Q[2, 1, 3, 0]`.

Temporal-difference error:

$$
\delta_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
$$

SARSA update:

$$
Q(s_t, a_t)^{\mathrm{new}} \leftarrow Q(s_t, a_t) + \alpha \delta_t
$$

Policy:

$$
\pi(a \mid s) = \frac{\exp\left(\beta Q(s, a)\right)}{\sum_{a'} \exp\left(\beta Q(s, a')\right)}
$$

where `alpha` is the learning rate, `beta` is the inverse temperature, and
`gamma` is the discount factor.

## Usage Modes

### Vanilla SARSA

When rewards are already stored in each `Quintuple.r2`, fit the canonical
SARSA parameter block directly:

```python
params, loss, q_trajectory, action_prob = sarsa.fit(
    quintuples,
    q0=q0,
    p0=np.array([0.5, sarsa.DEFAULT_POLICY_BETA, 0.9]),
)
```

By default, `fit()` treats `beta` as a fixed policy hyperparameter and optimizes
`alpha` and `gamma`. The default is `beta = 5.0`, chosen as a conservative
large-yet-stable setting. Set `fit_beta=True` to estimate
`beta` explicitly, and consider a sensitivity sweep on new tasks or reward scales.

Canonical parameter domains now follow edge-safe box constraints:
- `alpha ∈ [0, 1]`
- `beta ∈ [0, ∞)`
- `gamma ∈ [0, 1)`

This means `alpha = 0`, `alpha = 1`, `beta = 0`, and `gamma = 0` are all valid
edge cases, while exact `gamma = 1` remains excluded through the optimizer bound
`1 - EPS`. `fit()` emits a warning when trainable canonical SARSA parameters land
on active bounds, since that often signals weak identifiability or conditioning.

This uses the observed rewards in the input data as-is; SARSA does not
recompute them on the fly.

### Extended SARSA with user-defined parameters

If rewards depend on additional latent or task-specific parameters, provide a
`transition_reward_func(user_params, s1, a1, s2)` and matching
`user_param_bounds`. This callback is the extension hook for returning the next
state and transition reward on the fly from user-defined parameters rather than
reading the reward from `Quintuple.r2`. It should return `(next_state, reward)`,
where `next_state` matches the recorded next state and `reward` is the scalar
transition reward for that step.

In this mode, the optimizer still fits one flat parameter vector, but the
callback receives only the user-defined parameter block rather than the full
SARSA vector. By default, `fit()` keeps `beta` fixed as a policy hyperparameter;
set `fit_beta=True` if you want `beta` to be optimized.

`custom_param_bounds` is still accepted as a deprecated compatibility alias in
`fit()`.

## Output Notes

- `run()` now returns temporal-difference errors with length `T`, aligned with the number of transitions.

## Documentation

Full documentation is available in the repo under [`docs/`](docs/README.md):

- [`docs/algorithm.md`](docs/algorithm.md) — SARSA model and update rule
- [`docs/manual.md`](docs/manual.md) — API reference and usage guide
- [`docs/example.md`](docs/example.md) — Full walkthrough

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
