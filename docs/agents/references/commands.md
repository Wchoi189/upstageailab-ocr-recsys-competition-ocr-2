# Command Reference

**Purpose:** Quick reference for common commands. For detailed context, see `docs/maintainers/`.

## Training

```bash
# Basic training
uv run python runners/train.py preset=<name>

# With overrides
uv run python runners/train.py preset=<name> model.optimizer.lr=0.0005 data.batch_size=16

# Fast dev run (smoke test)
uv run python runners/train.py preset=<name> trainer.fast_dev_run=True
```

## Testing

```bash
# Run tests
uv run python runners/test.py preset=<name> checkpoint_path="<path>"
```

## Prediction

```bash
# Generate predictions
uv run python runners/predict.py preset=<name> checkpoint_path="<path>"
```

## UI Tools

```bash
# Command builder
python run_ui.py command_builder

# Inference UI
python run_ui.py inference

# Evaluation viewer
python run_ui.py evaluation_viewer

# Resource monitor
python run_ui.py resource_monitor
```

## Validation

```bash
# Validate config
uv run python scripts/agent_tools/validate_config.py --config-name <name>

# Validate manifest
uv run python scripts/agent_tools/documentation/validate_manifest.py
```

## Data Tools

```bash
# Generate samples
uv run python scripts/agent_tools/generate_samples.py --num-samples 5

# List checkpoints
uv run python scripts/agent_tools/list_checkpoints.py

# Data analyzer
uv run python tests/debug/data_analyzer.py --mode orientation|polygons|both [--limit N]

# Visualize predictions
uv run python ui/visualize_predictions.py --image_dir <path> --checkpoint <path> [--max_images N] [--score_threshold T]

# Clean dataset (scan and remove problematic samples)
uv run python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json
uv run python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json --output-report reports/cleaning_report.json
uv run python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json --list-bad
uv run python scripts/data/clean_dataset.py --image-dir data/datasets/images/train --annotation-file data/datasets/jsons/train.json --remove-bad --backup
```

## Experiment Tools

```bash
# Collect results
uv run python scripts/collect_results.py

# Generate ablation table
uv run python scripts/generate_ablation_table.py

# Next run proposer
uv run python scripts/agent_tools/ocr/next_run_proposer.py <wandb_run_id>
```

## Context Logging

```bash
# Start log
make context-log-start LABEL="<task>"

# Summarize log
make context-log-summarize LOG=logs/agent_runs/<file>.jsonl
```

## Code Quality

```bash
# Format code
uv run ruff format .

# Check code
uv run ruff check . --fix

# Type check
uv run mypy <files>
```

## Process Monitoring

```bash
# View logs
python scripts/process_monitor.py
```
