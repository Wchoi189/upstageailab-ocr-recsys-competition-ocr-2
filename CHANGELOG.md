# Changelog

## [Unreleased]

- **Python 3.11 Migration**: Upgraded minimum Python version from 3.10 to 3.11. Updated all
  configuration files, CI/CD workflows, Docker images, and dependencies. Installed Python
  3.11.14 via pyenv and regenerated `uv.lock` with Python 3.11 support.
- Restructured the `outputs/` directory into a structured `experiments/` and `artifacts/`
  layout and documented cleanup rules.
- Updated Hydra and paths configs so new runs write under
  `outputs/experiments/<kind>/<task>/<name>/<run_id>/` and loggers use canonical
  `outputs_root`-based paths.

