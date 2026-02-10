# Repository Guidelines

**Generated:** 2025-01-18 | **Commit:** 3142777 | **Branch:** main

## Overview

SARSA reinforcement learning toolkit for fitting behavioural datasets. Pure Python library (numpy, scipy) with Jupyter notebook workflow.

## Structure

```
sarsa/
├── src/sarsa/           # Core SARSA algorithm (sarsa.py, __init__.py)
├── examples/            # Notebook demo + experiment helpers + sample data
│   ├── sarsa.ipynb      # Primary entry point - run this
│   ├── experiment.py    # Task-specific state/reward helpers
│   ├── README.md        # Examples directory guide
│   └── M1.csv           # Sample behavioural dataset (6.3MB)
├── tests/               # Test suite (pytest)
│   ├── __init__.py      # Package marker for pytest discovery
│   └── test_sarsa.py    # Integration tests mirroring notebook workflow
├── .gitignore           # Git ignore rules
├── AGENTS.md            # Repository guidelines for AI agents
├── CHANGELOG.md         # Release history
├── LICENSE              # MIT license
├── README.md            # Project overview and installation
├── pyproject.toml       # Build config (hatchling), deps (numpy, scipy)
└── uv.lock              # Locked dependencies
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| SARSA algorithm | `src/sarsa/sarsa.py` | `fit()`, `run()`, `update()`, `Quintuple` |
| State construction | `examples/experiment.py` | `row_to_state()`, `process_data()` |
| Full workflow demo | `examples/sarsa.ipynb` | Start here for understanding |
| Add new analysis | `src/sarsa/` | Keep package task-agnostic |
| Add experiment helper | `examples/` | Task-specific code goes here |

## Code Map

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `fit` | function | sarsa.py:232 | Main entry - optimize SARSA params |
| `run` | function | sarsa.py:157 | Execute SARSA over quintuples |
| `update` | function | sarsa.py:122 | Single SARSA TD update |
| `Quintuple` | dataclass | sarsa.py:37 | (s1, a1, r2, s2, a2) transition |
| `ParamIndex` | enum | sarsa.py:29 | alpha=0, beta=1, gamma=2 |
| `PARAM_BOUNDS` | const | sarsa.py:22 | Default bounds for optimizer |
| `action_logprob` | function | sarsa.py:48 | Softmax log-probabilities for actions |
| `to_prob` | function | sarsa.py:67 | Convert log-probs to probabilities |
| `cross_entropy` | function | sarsa.py:83 | Cross-entropy loss against observed actions |
| `merge` | function | sarsa.py:102 | Combine trainable and fixed params |
| `run_and_loss` | function | sarsa.py:201 | Run SARSA and compute cross-entropy loss |

## Commands

```bash
# Install
uv sync                        # Runtime deps only
uv sync --extra examples       # + JupyterLab for notebook
uv sync --group dev            # + pytest for testing
pip install -e .               # Alternative editable install

# Run
jupyter lab examples/sarsa.ipynb   # Primary workflow

# Test
uv run pytest tests/ -v            # Run test suite

# Lint (treat warnings as errors before commit)
uvx ruff check
uvx ruff format
```

## Conventions

- **PEP 8** with 4-space indent
- **Type hints** on public functions: `def fit(quintuples: list, q0: NDArray, p0: NDArray, ...) -> tuple[NDArray, float, NDArray, NDArray]`
- **NumPy docstrings** with `Parameters`, `Returns`, `Raises` sections
- **One class/solver per file**
- **Constants**: UPPERCASE (`ACTION_SIZE`, `EPS`)
- **Imports**: Use `from sarsa import sarsa` after editable install

## Anti-Patterns

- **No `# type: ignore` / type suppression** - fix types properly
- **No CLI entry points** - library only, use notebook or import
- **No commits without asking** - always ask the user before committing
- **Never push** - user will push manually

## Notes

- State/action must be integer numpy arrays
- First 3 params are always (alpha, beta, gamma); custom params follow
- `transition_reward_func` callback computes rewards on-the-fly during `run()`
- Large datasets: place outside `examples/`, document location
