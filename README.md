# SARSA

SARSA toolkit for fitting behavioural datasets and running reproducible experiments.

## Installation

### Using pip (recommended for most users)

```bash
pip install git+https://github.com/yuanz271/sarsa.git@v0.2.0
```

### Using uv (for development)

```bash
uv sync                        # Runtime deps only
uv sync --extra examples       # + JupyterLab for notebook
uv sync --group dev            # + pytest for testing
```

### Editable install from source

```bash
pip install -e .               # Runtime deps only
pip install -e .[examples]     # + JupyterLab for notebook
```

## Example

Fit to the session `examples/M1.csv`:

```bash
jupyter lab examples/sarsa.ipynb
```

## Usage Modes

### Vanilla SARSA

When rewards are already stored in each `Quintuple.r2`, fit the canonical
three-parameter model directly:

```python
params, loss, q_trajectory, action_prob = sarsa.fit(
    quintuples,
    q0=q0,
    p0=np.array([0.5, 1.0, 0.9]),
)
```

This uses the observed rewards in the input data as-is; SARSA does not
recompute them on the fly.

### Extended SARSA with trainable reward parameters

If rewards depend on additional latent/task-specific parameters, provide a
`transition_reward_func` and matching `custom_param_bounds`. In this mode,
`params[3:]` can be used to learn reward-related quantities jointly with
`alpha`, `beta`, and `gamma`.

## Data Assumptions

- The example preprocessing expects a `TIME (S)` column in behavioral data for resampling.

## Output Notes

- `run()` now returns temporal-difference errors with length `T`, aligned with the number of transitions.

## Testing

```bash
uv run pytest tests/ -v
```

## Linting

```bash
uvx ruff check
uvx ruff format
```

## License

[MIT](LICENSE)
