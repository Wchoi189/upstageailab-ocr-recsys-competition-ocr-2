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
Scripts and tools for AI agent integration and automation.

### `analysis_validation/`
Scripts for analyzing experiments, validating pipeline contracts, and profiling performance.
- `analyze_experiment.py` - Analyze training runs and detect anomalies
- `validate_pipeline_contracts.py` - Validate data contracts across OCR pipeline
- `profile_data_loading.py` - Profile data loading performance
- `profile_transforms.py` - Profile data transformation performance

### `bug_tools/`
Bug-related tools and utilities.

### `checkpoints/` ⭐ NEW
All checkpoint-related operations.
- `migrate.py` - Migrate checkpoints to new hierarchical naming scheme
- `generate_metadata.py` - Generate checkpoint metadata

### `data/` ⭐ RENAMED (from `data_processing/`)
Scripts for preprocessing and transforming data.
- `preprocess.py` - Preprocess data (moved from `preprocess_data.py`)
- `preprocess_maps.py` - Generate probability/threshold maps for training
- `fix_canonical_orientation_images.py` - Fix image orientation issues
- `report_orientation_mismatches.py` - Report orientation mismatch issues
- `clean_dataset.py` - Scan and clean training dataset for problematic samples

### `documentation/` ⭐ NEW
Documentation generation and management tools.
- `generate_diagrams.py` - Generate documentation diagrams
- `manage_diagrams.sh` - Manage diagram files
- `ci_update_diagrams.sh` - CI script for updating diagrams
- `standardize_content.py` - Standardize documentation content

### `migration_refactoring/`
Scripts for migrating data formats and refactoring code.
- `migrate_checkpoint_names.py` - Migrate checkpoint filenames to new format
- `refactor_ocr_pl.py` - Automated refactoring of OCR Lightning module

### `monitoring/`
System monitoring and resource management tools.
- `monitor.sh` - AI-powered system monitoring via Qwen MCP server
- `monitor_resources.sh` - Resource monitoring script
- `process_monitor.py` - Process monitoring utilities

### `performance/` ⭐ MERGED (includes `performance_benchmarking/`)
All performance analysis, benchmarking, and reporting tools.
- `benchmark.py` - Performance benchmarking (moved from `benchmark_performance.py`)
- `generate_baseline_report.py` - Generate performance baseline reports
- `compare_baseline_vs_optimized.py` - Compare baseline vs optimized runs
- `compare_three_runs.py` - Compare three different runs
- `benchmark_optimizations.py` - Benchmark data loading optimizations
- `benchmark_validation.py` - Validate benchmarking results
- `decoder_benchmark.py` - Benchmark decoder performance
- `performance_measurement.py` - General performance measurements
- `performance_test.py` - Performance testing utilities
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

### `utilities/` ⭐ NEW
General utility scripts.
- `cache_manager.py` - Cache management utilities
- `process_manager.py` - Process management utilities

### `validation/` ⭐ NEW
All validation scripts organized by type.
- `checkpoints/` - Checkpoint validation scripts
  - `validate_coordinate_consistency.py` - Validate coordinate consistency
- `docs/` - Documentation validation scripts
  - `validate_links.py` - Validate documentation links
- `schemas/` - Schema validation scripts
  - `validate_ui_schema.py` - Validate UI inference compatibility schema
- `templates/` - Template validation scripts
  - `validate_templates.py` - Validate template files

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

The following scripts remain in the root directory for backward compatibility or debugging:
- `debug_cuda.sh` - CUDA debugging utilities
- `debug_imports.py` - Import debugging utilities
- `debug_wandb_import.py` - Weights & Biases import debugging

## Contributing

When adding new scripts:
1. Place them in the appropriate subdirectory
2. Update this README with documentation
3. Ensure scripts are executable (`chmod +x script.sh`)
4. Include usage examples and descriptions
