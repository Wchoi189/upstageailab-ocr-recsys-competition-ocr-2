# Changelog

## [Unreleased]

- Restructured the `outputs/` directory into a structured `experiments/` and `artifacts/`
  layout and documented cleanup rules.
- Updated Hydra and paths configs so new runs write under
  `outputs/experiments/<kind>/<task>/<name>/<run_id>/` and loggers use canonical
  `outputs_root`-based paths.

