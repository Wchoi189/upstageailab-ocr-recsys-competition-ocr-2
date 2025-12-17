# Scripts Directory

This directory contains various utility scripts and tools for the OCR project.

## Quick Reference

### Checkpoint Migration
Migrate existing checkpoints to the new naming scheme:
```bash
# Dry run (preview changes)
python scripts/checkpoints/migrate.py --dry-run --verbose

# Migrate and cleanup (delete early epochs)
python scripts/checkpoints/migrate.py --delete-old
```

See `docs/CHECKPOINT_MIGRATION_GUIDE.md` for details.

### UI Schema Validation
Validate the UI inference compatibility schema:
```bash
# Validate schema correctness
python scripts/validation/schemas/validate_ui_schema.py
```

See `docs/UI_INFERENCE_COMPATIBILITY_SCHEMA.md` for details.

### Data Cleaning
Scan and clean training dataset for problematic samples:
```bash
# Scan and report issues (dry run)
uv run python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json

# Generate report and save to file
uv run python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json --output-report reports/data_cleaning_report.json

# Remove problematic samples (with backup)
uv run python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json --remove-bad --backup
```

---

## Organization

### `agent_tools/`
Scripts and tools for AI agent integration and automation (run with `python -m scripts.agent_tools ...`).

### `bug_tools/`
Bug-related tools (e.g., `next_bug_id.py`).

### `checkpoints/`
All checkpoint-related operations.
- `migrate.py` - Migrate checkpoints to new hierarchical naming scheme
- `generate_metadata.py` - Generate checkpoint metadata
- `convert_legacy_checkpoints.py` - Convert legacy checkpoints to V2 metadata format

### `data/`
Data preprocessing and diagnostics.
- `preprocess.py` - End-to-end preprocessing entrypoint
- `preprocess_maps.py` - Generate probability/threshold maps for training
- `fix_canonical_orientation_images.py` - Fix image orientation issues
- `report_orientation_mismatches.py` - Report orientation mismatch issues
- `clean_dataset.py` - Scan and clean training dataset for problematic samples
- `check_training_data.py` - Quick validation of dataset integrity

### `documentation/`
Documentation generation and management tools.
- `manage_diagrams.sh` - Manage diagram files
- `ci_update_diagrams.sh` - CI script for updating diagrams
- `standardize_content.py` - Standardize documentation content

**Note:** `generate_diagrams.py` is located at the root level for backward compatibility with Makefile and CI scripts.

### `migration_refactoring/`
Scripts for migrating data formats and refactoring code.
- `migrate_checkpoint_names.py` - Migrate checkpoint filenames to new format
- `refactor_ocr_pl.py` - Automated refactoring of OCR Lightning module

### `monitoring/`
System monitoring and resource management tools.
- `monitor.sh` - AI-powered system monitoring via Qwen MCP server
- `monitor_resources.sh` - Resource monitoring script
- `process_monitor.py` - Process monitoring utilities

### `performance/`
All performance analysis, benchmarking, and reporting tools.
- `analyze_imports.py` - Analyze import structure and dependencies
- `benchmark.py` - Performance benchmarking
- `benchmark_optimizations.py` - Benchmark data loading optimizations
- `benchmark_validation.py` - Validate benchmarking results
- `compare_baseline_vs_optimized.py` - Compare baseline vs optimized runs
- `compare_three_runs.py` - Compare three different runs
- `decoder_benchmark.py` - Benchmark decoder performance
- `generate_baseline_report.py` - Generate performance baseline reports
- `performance_measurement.py` - General performance measurements
- `performance_test.py` - Performance testing utilities
- `profile_imports.py` - Profile import times to identify startup bottlenecks
- `quick_performance_validation.py` - Quick performance validation

### `seroost/`
Seroost semantic search indexing tools and configuration.
- `SEROOST_INDEXING_SETUP.md` - Complete setup documentation
- `setup_seroost_indexing.py` - Python script for indexing setup
- `test_seroost_config.py` - Configuration validation script
- `run_seroost_indexing.sh` - Shell script for running indexing
- `install_and_run_seroost.sh` - Complete installation and setup script
- `seroost_indexing/` - Additional indexing documentation

### `setup/`
Project setup and configuration scripts.
- `00_setup-environment.sh` - Environment setup
- `01_setup-professional-linting.sh` - Linting setup
- `02_setup-bash-aliases.sh` - Bash aliases setup
- `code-quality.sh` - Automated code quality maintenance
- `qwen-advanced.sh` - Advanced Qwen memory optimization
- `qwen-memfix.sh` - Qwen memory fixes
- `qwen-version.sh` - Qwen version management
- `setup_mcp_github_only.sh` - MCP GitHub setup

### `troubleshooting/`
Interactive debugging and hardware diagnostics.
- `debug_cuda.sh` / `diagnose_cuda_issue.py` - CUDA troubleshooting helpers
- `debug_imports.py` / `debug_wandb_import.py` - Import diagnostics
- `test_basic_cuda.py` / `test_cudnn_stability.sh` - CUDA sanity checks
- `test_model_forward_backward.py` - Training loop smoke test
- `test_wandb_multiprocessing_fix.sh` etc. - Targeted regression scripts

### `utilities/`
General utility scripts.
- `cache_manager.py` - Cache management utilities

**Note:** `process_manager.py` is located at the root level for backward compatibility with Makefile and documentation.

### `validation/`
All validation scripts organized by type.
- `checkpoints/` - Checkpoint validation scripts
  - `validate_coordinate_consistency.py` - Validate coordinate consistency
- `docs/` - Documentation validation scripts
  - `validate_links.py` - Validate documentation links
- `schemas/` - Schema validation scripts
  - `validate_ui_schema.py` - Validate UI inference compatibility schema
- `templates/` - Template validation scripts
  - `validate_templates.py` - Validate template files

## Usage

### System Monitoring
```bash
# AI-powered system monitoring
./scripts/monitoring/monitor.sh "Show system health status"

# Resource monitoring
./scripts/monitoring/monitor_resources.sh
```

### Seroost Indexing
```bash
# Complete setup (install + index)
./scripts/seroost/install_and_run_seroost.sh

# Just run indexing (if seroost is already installed)
./scripts/seroost/run_seroost_indexing.sh

# Manual setup
python ./scripts/seroost/setup_seroost_indexing.py
```

## Root Scripts

Scripts at the root level for backward compatibility and convenience:
- `_bootstrap.py` - Ensures project modules resolve when scripts run directly
- `preprocess_data.py` - Delegates to `scripts/data/preprocess.py`
- `validate_config.py` - Configuration validation utility
- `download_hf_sample.py` - HuggingFace model/dataset download utility
- `generate_diagrams.py` - Automated Mermaid diagram generation (used by Makefile and CI scripts)
- `ci_update_diagrams.sh` - CI script for diagram updates
- `manage_diagrams.sh` - Diagram management utility

## Contributing

When adding new scripts:
1. Place them in the appropriate subdirectory
2. Update this README with documentation
3. Ensure scripts are executable (`chmod +x script.sh`)
4. Include usage examples and descriptions
