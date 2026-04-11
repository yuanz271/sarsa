# Contributing

## Development setup

[uv](https://github.com/astral-sh/uv) is the recommended tool for managing
the development environment.

```bash
uv sync --group dev            # Install dev dependencies (pytest, ruff, ty)
uv sync --extra examples       # Also install JupyterLab for the notebook
```

### Editable install from source

```bash
pip install -e .               # Runtime deps only
pip install -e .[examples]     # + JupyterLab for notebook
```

## Running tests

```bash
uv run pytest tests/ -v
```

The test suite requires `examples/M1.csv` (~6.3 MB) to be present.

## Linting and formatting

```bash
uvx ruff check
uvx ruff format
```

Treat warnings as errors before committing.

## Type checking

```bash
uvx ty check
```

## Workflow

- Branch from `main`, keep branches focused and short-lived.
- Open a pull request and merge via review.
- Never commit directly to `main`.
