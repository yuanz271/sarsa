# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.4.0] - 2026-04-13

### Changed
- `fit()` now treats `beta` as a fixed policy hyperparameter by default and only optimizes it when `fit_beta=True` is requested.
- Fixed parameters are now removed from the optimization subspace instead of being overwritten inside the objective, improving conditioning when parameters such as `beta` are held constant.
- `fit()` now selects `L-BFGS-B` explicitly for bounded optimization.
- Canonical SARSA bounds now use edge-safe box domains: `alpha ∈ [0, 1]`, `beta ∈ [0, ∞)`, and `gamma ∈ [0, 1)`.
- `fit()` now warns when trainable canonical SARSA parameters land on active bounds, since this often indicates weak identifiability or conditioning.
- Explicit fixed parameters supplied through `static_params` are now validated against their declared bounds.

## [0.3.0] - 2026-04-11

### Added
- Added `concat_params()` and `split_params()` helpers to centralize packing and unpacking of flat optimizer vectors into SARSA-owned and user-defined parameter blocks.

### Changed
- Refactored parameter handling around explicit `sarsa_params` and `user_params` blocks while preserving a single flat optimizer vector internally.
- Changed the reward callback contract: `transition_reward_func` now receives `user_params` rather than the full parameter vector — **breaking change**.
- Introduced `user_param_bounds` as the preferred `fit()` argument name; `custom_param_bounds` remains accepted as a deprecated compatibility alias.
- Switched package metadata to file-backed dynamic versioning with `src/sarsa/__about__.py` as the single source of truth.

## [0.2.0] - 2026-04-11

### Added
- Added vanilla SARSA mode: `run`/`fit` use `Quintuple.r2` directly when no `transition_reward_func` is provided, removing boilerplate for standard reward setups.
- Added validation that extra trainable parameters require a reward callback.

### Changed
- Refactored reward callback to transition-based signature (`transition_reward_func` now receives `(params, s1, a1, s2)` and returns `(s2, reward)`) — **breaking change**.
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
- Computed stepwise rewards during `run` so `update` can consume them directly, keeping reward-related parameters consistent across trajectories.

[Unreleased]: https://github.com/yuanz271/sarsa/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/yuanz271/sarsa/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yuanz271/sarsa/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yuanz271/sarsa/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yuanz271/sarsa/releases/tag/v0.1.0
