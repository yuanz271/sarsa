# Contributing

## Development setup

[uv](https://github.com/astral-sh/uv) is the recommended tool for managing
the development environment.

```bash
uv sync --group dev            # Install dev dependencies (pytest, ruff, ty, pandas)
uvx --from jupyterlab jupyter lab
```

### Editable install from source

```bash
pip install -e .               # Runtime deps only
```

## Running tests

```bash
uv run pytest tests/ -v
```

The test suite uses synthetic in-repo fixtures and does not depend on external datasets.

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
