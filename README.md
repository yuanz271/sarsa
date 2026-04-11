# SARSA

SARSA toolkit for fitting behavioural datasets and running reproducible experiments.

## Installation

### Using pip (recommended for most users)

```bash
pip install git+https://github.com/yuanz271/sarsa.git@v0.2.0
```

## Example

Fit to the session `examples/M1.csv`:

```bash
jupyter lab examples/sarsa.ipynb
```

## Usage Modes

### Vanilla SARSA

When rewards are already stored in each `Quintuple.r2`, fit the canonical
SARSA parameter block directly:

```python
params, loss, q_trajectory, action_prob = sarsa.fit(
    quintuples,
    q0=q0,
    p0=np.array([0.5, 1.0, 0.9]),
)
```

This uses the observed rewards in the input data as-is; SARSA does not
recompute them on the fly.

### Extended SARSA with user-defined parameters

> **Note:** The `user_params` / `user_param_bounds` interface described below is
> available on the current development branch and will appear in the next
> release. The latest release (`v0.2.0`) still uses
> `transition_reward_func(params, s1, a1, s2)` together with
> `custom_param_bounds`.

If rewards depend on additional latent or task-specific parameters, provide a
`transition_reward_func(user_params, s1, a1, s2)` and matching
`user_param_bounds`. In this mode, the optimizer still fits one flat parameter
vector, but the callback receives only the user-defined parameter block rather
than the full SARSA vector. This refactor isolates user-defined parameters from
the extension hook; the vanilla SARSA kernel still canonically interprets
`alpha`, `beta`, and `gamma`.

## Data Assumptions

- The example preprocessing expects a `TIME (S)` column in behavioral data for resampling.

## Output Notes

- `run()` now returns temporal-difference errors with length `T`, aligned with the number of transitions.

## Documentation

Full documentation is available on the [GitHub Wiki](https://github.com/yuanz271/sarsa/wiki):

- [Algorithm](https://github.com/yuanz271/sarsa/wiki/Algorithm) — SARSA model and update rule
- [Manual](https://github.com/yuanz271/sarsa/wiki/Manual) — API reference and usage guide
- [Example](https://github.com/yuanz271/sarsa/wiki/Example) — Full walkthrough

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
