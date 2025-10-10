# Repository Guidelines

## Project Structure & Module Organization
The reinforcement learning logic is concentrated in `src/sarsa.py`, with experiment definitions and state helpers in `src/experiment.py`. Use `src/run_sarsa.py` as the entry point; it wires data loading, state construction, and solver configuration. Keep new analysis utilities alongside the SARSA helpers under `src/` to preserve import paths. Shared datasets such as `data/M1.csv` should remain version-controlled; place large or sensitive files outside `data/` and document their location.

## Build, Test, and Development Commands
- `uv sync` — installs all runtime dependencies defined in `pyproject.toml`/`uv.lock` using Python 3.11.
- `pip install -e .` — editable install alternative when `uv` is unavailable.
- `python -m src.run_sarsa` — executes the default learning session against `data/M1.csv` for a smoke test.
- `uvx ruff check` — lints the codebase; treat warnings as errors before committing.
- `uvx ruff format` — applies standardized formatting to Python sources; run after substantial edits.
When adding notebooks or scripts, prefer relative imports (e.g., `from src import sarsa`) so they work after editable installation.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and limit files to one primary class or solver. Module-level constants stay uppercase (e.g., `ACTION_SIZE`), enums and classes use `CamelCase`, and helper functions stay `snake_case`. Use type hints on public functions (`def run_session(path: Path, rng: np.random.Generator) -> None`) and document them with NumPy-style docstrings including `Parameters`, `Returns`, and `Raises` sections where applicable.

## Testing Guidelines
No automated tests ship today, so please add `pytest`-style cases under a new `tests/` directory whenever you contribute behaviour. Name files `test_<module>.py` and mirror the structure of `src/`. Provide fixtures for sample quintuples rather than loading full CSVs when possible, and ensure `pytest` runs cleanly via `pytest -q` before submitting.

## Commit & Pull Request Guidelines
While no Git history is tracked in this workspace, use short, imperative commit subjects (`Add SARSA policy evaluation`) with optional detail in the body. Reference linked issues or experiment IDs in square brackets for traceability. Pull requests should summarize algorithmic intent, list datasets touched, and include screenshots or table snippets for major outcome changes. Flag data dependencies and new parameters in the PR checklist so reviewers can reproduce results. Update `CHANGELOG.md` with notable additions, fixes, and behavioural changes before requesting review.
