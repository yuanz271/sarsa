# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Refactored reward callback to transition-based signature (`transition_reward_func` now receives `(params, s1, a1, s2)` and returns `(s2, reward)`).
- Updated `AGENTS.md` with accurate paths and code map.
- Added validation for quintuples and Q-table shapes in `run`/`fit`, raising clearer errors for invalid indices.

### Fixed
- Fixed SARSA updates to propagate Q-values sequentially during `run` and `fit`.

## [0.1.0] - 2025-01-10

### Added
- Documented Ruff linting and formatting commands in `AGENTS.md`.
- Provided an `examples` optional dependency that includes JupyterLab for notebook workflows.

### Changed
- Applied `uvx ruff format` across the repository to enforce consistent style.
- Converted the source tree to a `src/sarsa` package layout for distribution.
- Relocated the SARSA runner and experiment helpers into a top-level `examples/` directory to keep algorithms standalone.
- Removed the package-level entry point and captured orchestration inside `examples/sarsa.ipynb` as an interactive example.
- Kept experiment helpers next to the walkthrough so the `sarsa` package remains task-agnostic.
- Compute stepwise rewards during `run` so `update` can consume them directly, keeping reward-related parameters consistent across trajectories.

[Unreleased]: https://github.com/yuanz271/sarsa/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yuanz271/sarsa/releases/tag/v0.1.0
