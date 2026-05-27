# Repository Guidelines

**Generated:** 2025-01-18 | **Commit:** 3142777 | **Branch:** main

## Overview

SARSA reinforcement learning toolkit. Pure Python library (numpy, scipy) with in-repo Markdown documentation.

## Git Workflow

- Follow GitHub flow: branch from `main`, open a PR, and merge via review.
- Keep branches focused and short-lived.
- Never commit, push, merge, rebase, or tag without user approval.

## Branch Policy

- `main` — general library branch; no demo or notebook assets.
- `pact` — superset of `main`; contains demo and notebook-specific assets.
  - Sync `main` into `pact` freely: switch to `pact` and run `git merge main`.
  - Never run `git merge pact` on `main`; use `git cherry-pick <commit>` to bring specific commits to `main` if needed.
  - `pact`-specific files (notebooks, demos) must never be committed to `main`.

## Structure

```
sarsa/
├── src/sarsa/           # Core SARSA algorithm (sarsa.py, __init__.py)
├── tests/               # Test suite (pytest)
│   ├── __init__.py      # Package marker for pytest discovery
│   └── test_sarsa.py    # Integration tests
├── docs/                # Project documentation (Algorithm, Manual, Example)
├── .gitignore           # Git ignore rules
├── AGENTS.md            # Repository guidelines for AI agents
├── CHANGELOG.md         # Release history
├── LICENSE              # MIT license
├── README.md            # Project overview and installation
└── pyproject.toml       # Build config (hatchling), deps (numpy, scipy)
```

## Where to Look

| Task | Location | Notes |
|------|----------|-------|
| SARSA algorithm | `src/sarsa/sarsa.py` | `fit()`, `run()`, `update()`, `Quintuple` |
| API details | `docs/manual.md` | Full parameter and function reference |
| Model equations | `docs/algorithm.md` | SARSA update + policy |
| End-to-end usage | `docs/example.md` | Vanilla + extended usage walkthrough |

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
# Install (dependency + venv management)
uv sync                        # Runtime deps only
uv sync --group dev            # + pandas/pytest for testing
pip install -e .               # Alternative editable install

# Docs preview (optional)
uvx --from jupyterlab jupyter lab

# Test
uv run pytest tests/ -v            # Run test suite

# Lint & format (treat warnings as errors before commit)
uvx ruff check
uvx ruff format

# Type check
uvx ty check
```

## Tooling

- **uv** for dependency and virtual environment management.
- **ruff** for linting and formatting.
- **ty** for static type checking.
- **pytest** for tests.

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

## Documentation Updates

- Update documentation whenever changes are made (README, inline docs, changelog, or other relevant docs).

## Notes

- State/action must be integer numpy arrays
- First 3 params are always (alpha, beta, gamma); custom params follow
- `transition_reward_func` callback computes rewards on-the-fly during `run()`
