# Changelog

## [Unreleased]

- **Phase 4 Refactoring**: Centralized path, callback, and logger setup across runners with
  lazy imports and fail-fast error handling. Added `ensure_output_dirs()`, `build_callbacks()`,
  and unified logger creation. Narrowed exception handling in predict runner. Created follow-up
  plan for deprecated `PathUtils` removal to prevent AI agent confusion.
- **Python 3.11 Migration**: Upgraded minimum Python version from 3.10 to 3.11. Updated all
  configuration files, CI/CD workflows, Docker images, and dependencies. Installed Python
  3.11.14 via pyenv and regenerated `uv.lock` with Python 3.11 support.
- Restructured the `outputs/` directory into a structured `experiments/` and `artifacts/`
  layout and documented cleanup rules.
- Updated Hydra and paths configs so new runs write under
  `outputs/experiments/<kind>/<task>/<name>/<run_id>/` and loggers use canonical
  `outputs_root`-based paths.

