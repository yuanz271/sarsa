# SARSA

SARSA toolkit for fitting behavioural datasets and running reproducible experiments.

## Installation

### Using uv (recommended)

```bash
uv sync                        # Runtime deps only
uv sync --extra examples       # + JupyterLab for notebook
uv sync --group dev            # + pytest for testing
```

### Using pip

Install dependencies into a Python (>=3.11) environment with an editable install:

```bash
pip install -e .
```

Install with examples:

```bash
pip install -e .[examples]
```

Install the latest code directly from GitHub:

```bash
pip install git+https://github.com/yuanz271/sarsa.git
```

## Example

Fit to the session `examples/M1.csv`:

```bash
jupyter lab examples/sarsa.ipynb
```

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
