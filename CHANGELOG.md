# Changelog

## [Unreleased]

- **Config Architecture Consolidation (Phases 5-8)**: Completed major restructuring of Hydra configuration architecture, reducing cognitive load by 43% (7.0 → 4.0) and improving maintainability.
  - **Phase 5 (Low-Hanging Fruit)**: Deleted `.deprecated/`, `metrics/`, `extras/` directories. Moved `ablation/` to `docs/research/`, `schemas/` to `docs/schemas/`, and consolidated `hardware/` into `trainer/`. Reduced from 102 to 90 YAML files.
  - **Phase 6 (Data Consolidation)**: Created unified `data/` hierarchy by moving `dataloaders/`, `transforms/`, and `preset/datasets/` into `data/` subdirectories. Eliminated 3 directories while maintaining 90 files. Single source of truth for all data-related configs.
  - **Phase 7 (Preset/Models Elimination)**: Eliminated `preset/models/` directory by creating `model/encoder/`, `model/decoder/`, `model/head/`, `model/loss/`, and `model/presets/` subdirectories. Moved 15 model component files. Single source of truth for all model configs.
  - **Phase 8 (Final Consolidation)**: Moved `lightning_modules/` to `model/`, eliminated entire `preset/` directory, and relocated tool configs (`repomix.config.json`, `seroost_config.json`) to `.vscode/`. Created `.vscode/README.md` for tool documentation. Final count: 89 YAML files, 17 subdirectories.
  - **Overall Impact**: Reduced from 102 to 89 YAML files (12.7% reduction), removed 10 directories, improved config organization with clear hierarchies, and separated Hydra configs from IDE/tool configs. Updated `docs/architecture/CONFIG_ARCHITECTURE.md` with new structure.

- **PathUtils Deprecation Removal**: Removed entire deprecated `PathUtils` class and all legacy
  standalone helper functions (`setup_paths()`, `add_src_to_sys_path()`, `ensure_project_root_env()`,
  deprecated `get_*` functions). Module size reduced by 47% (748 → 396 lines). All callers migrated
  to modern API (`get_path_resolver()`, `setup_project_paths()`). Prevents AI agent confusion from
  dead code and architecture drift.
- **Phase 4 Refactoring**: Centralized path, callback, and logger setup across runners with
  lazy imports and fail-fast error handling. Added `ensure_output_dirs()`, `build_callbacks()`,
  and unified logger creation. Narrowed exception handling in predict runner. Created follow-up
  plan for deprecated `PathUtils` removal to prevent AI agent confusion.
- **Python 3.11 Migration**: Upgraded minimum Python version from 3.10 to 3.11. Updated all
  configuration files, CI/CD workflows, Docker images, and dependencies. Installed Python
  3.11.14 via pyenv and regenerated `uv.lock` with Python 3.11 support.
- **Streamlit Deprecation & UI Archival**: Fully archived the legacy Streamlit application code (`ui/`) to `docs/archive/legacy_ui_code/`. Extracted shared business logic (InferenceEngine, config parsing) to the core `ocr` package. Updated `ocr_bridge.py` to use the new `ocr.inference` package. This resolves the fractured UI architecture and "legacy import" issues.
- Restructured the `outputs/` directory into a structured `experiments/` and `artifacts/`
  layout and documented cleanup rules.
- Updated Hydra and paths configs so new runs write under
  `outputs/experiments/<kind>/<task>/<name>/<run_id>/` and loggers use canonical
  `outputs_root`-based paths.

