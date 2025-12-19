# Safe State Management

## Overview

The Experiment Tracker uses YAML-based state files with atomic write operations to prevent corruption during editing. This system provides safe, reliable state management for experiment metadata.

## Background

Previously, experiment state was stored in JSON format (`state.json`), which was prone to corruption during direct edits. The new system uses YAML format with atomic operations and automatic backups.

## Key Features

- **YAML Format**: More readable and error-tolerant than JSON
- **Atomic Writes**: Changes written to temporary files, then atomically moved
- **Automatic Backups**: Original files backed up before modifications
- **Validation**: Built-in structure validation
- **CLI Tools**: Command-line interface for safe operations

## File Location

State files are located at:
```
experiment-tracker/experiments/{experiment_id}/state.yml
```

## Safe Operations

### Command Line Usage

```bash
# Validate state file integrity
python scripts/safe_state_manager.py experiments/{experiment_id}/state.yml --validate

# Get a specific section
python scripts/safe_state_manager.py experiments/{experiment_id}/state.yml --get tasks

# Set a section value
python scripts/safe_state_manager.py experiments/{experiment_id}/state.yml --set status completed

# Update nested values
python scripts/safe_state_manager.py experiments/{experiment_id}/state.yml --set tasks.0.status completed
```

### Python API

```python
from pathlib import Path
from experiment_tracker.scripts.safe_state_manager import SafeStateManager

# Initialize manager
state_path = Path("experiments/my_experiment/state.yml")
manager = SafeStateManager(state_path)

# Load current state
state = manager.load_state()

# Update safely
manager.update_section("status", "completed")

# Validate
issues = manager.validate_state()
if issues:
    print("Validation issues:", issues)
```

## Migration from JSON

Existing JSON state files are automatically migrated to YAML:

```bash
python scripts/migrate_state_to_yaml.py experiments/{experiment_id}/state.json
```

This creates `state.yml` and backs up the original as `state.json.backup`.

## Integration with ETK

The safe state manager integrates with the Experiment Tracker Kit:

```bash
cd experiment-tracker

# Sync state with metadata files
etk sync --all

# Validate experiment compliance
etk validate

# Check for inconsistencies
etk validate --consistency
```

## File Format

State files use YAML with this structure:

```yaml
id: "experiment_id"
status: "active"
created_at: "2025-12-18T00:00:00"
updated_at: "2025-12-18T19:00:00"
type: "experiment_type"
description: "Experiment description"

# Experiment-specific data
checkpoint_info:
  path: "outputs/checkpoints/model.ckpt"
  performance: "97% hmean"

baseline_metrics:
  background_color_variance: 36.5
  color_tint_score: 58.1

# Tasks, decisions, insights arrays
tasks: []
decisions: []
insights: []

# VLM integration data
vlm_integration:
  backend: "dashscope"
  model: "qwen3-vl-plus-2025-09-23"
```

## Safety Guidelines

### ✅ Do
- Use the safe state manager for all programmatic changes
- Run validation after manual edits
- Use `etk sync --all` to keep metadata consistent
- Check backups exist before risky operations

### ❌ Don't
- Edit `state.yml` files directly with text editors
- Modify JSON files without migration
- Skip validation steps
- Ignore backup creation

## Error Handling

- **Invalid YAML**: Automatic rollback to backup file
- **Validation Failures**: Writes prevented, errors reported
- **File Corruption**: Backup restoration available
- **Permission Issues**: Clear error messages with recovery steps

## Troubleshooting

### State File Not Found
```bash
# Check if experiment exists
etk list

# Create missing state file
python scripts/safe_state_manager.py experiments/{experiment_id}/state.yml --set id "{experiment_id}"
```

### Validation Errors
```bash
# Run detailed validation
python scripts/safe_state_manager.py experiments/{experiment_id}/state.yml --validate

# Check for common issues
etk validate experiments/{experiment_id}
```

### Sync Issues
```bash
# Force sync with metadata
etk sync --experiment {experiment_id} --force

# Check consistency
etk validate --consistency
```

## Implementation Details

The safe state manager uses:
- `pyyaml` for YAML parsing
- Temporary files for atomic writes
- Automatic backup rotation
- Structured validation rules

For more technical details, see the source code in `scripts/safe_state_manager.py`.
