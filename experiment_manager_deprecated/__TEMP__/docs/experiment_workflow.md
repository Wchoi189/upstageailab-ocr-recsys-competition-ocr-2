# Experiment Workflow Guide

## Overview

This guide explains how to reference and work with experiments in the experiment_manager system.

## Naming Conventions

### experiment_manager Reference

**Full Path**: `experiment_manager/` (from project root)

**Short Reference**: `@experiment_manager` or `experiment_manager/`

**In Code/Documentation**: Use `experiment_manager/` for clarity

### Experiment ID Format

Experiments are identified by their **experiment ID** in the format:
```
YYYYMMDD_HHMMSS_<type>
```

**Example**: `20251122_172313_perspective_correction`

**Components**:
- `YYYYMMDD`: Date (2025-11-22)
- `HHMMSS`: Time (17:23:13)
- `<type>`: Experiment type (e.g., `perspective_correction`)

### Experiment Path

**Full Path**: `experiment_manager/experiments/<experiment_id>/`

**Example**: `experiment_manager/experiments/20251122_172313_perspective_correction/`

## Starting a New Experiment

### Method 1: Using CLI Script (Recommended)

```bash
cd experiment_manager
python scripts/start-experiment.py \
    --type perspective_correction \
    --intention "Improve corner detection accuracy"
```

**What happens**:
- Creates new experiment directory: `experiments/YYYYMMDD_HHMMSS_perspective_correction/`
- Sets it as the current active experiment (`.current` file)
- Initializes metadata files (tasks.yml, decisions.yml, state.yml)
- Returns the experiment ID

### Method 2: Direct Reference

If you know the experiment ID, you can reference it directly:

```bash
# Reference by full path
experiment_manager/experiments/20251122_172313_perspective_correction/

# Reference by ID only (when context is clear)
20251122_172313_perspective_correction
```

## Resuming an Existing Experiment

### Method 1: Using Experiment ID (Recommended)

**Step 1: Find the experiment ID**

```bash
# List all experiments
ls experiment_manager/experiments/

# Or check the .current file
cat experiment_manager/experiments/.current
```

**Step 2: Reference the experiment**

In conversation/documentation:
- **Full reference**: `experiment_manager/experiments/20251122_172313_perspective_correction`
- **Short reference**: `@20251122_172313_perspective_correction` or just the ID

**Step 3: Work with the experiment**

All scripts automatically use the current experiment (from `.current` file), or you can specify:

```bash
# Scripts will use .current file automatically
python experiment_manager/scripts/add-task.py --description "Continue testing"

# Or specify experiment ID explicitly (if script supports it)
python experiment_manager/scripts/record-artifact.py \
    --path output.jpg \
    --experiment-id 20251122_172313_perspective_correction
```

### Method 2: Create Resume Script (Recommended Addition)

**Proposed script**: `experiment_manager/scripts/resume-experiment.py`

```bash
# Resume by ID
python experiment_manager/scripts/resume-experiment.py \
    --id 20251122_172313_perspective_correction

# Resume by type (latest)
python experiment_manager/scripts/resume-experiment.py \
    --type perspective_correction

# List available experiments
python experiment_manager/scripts/resume-experiment.py --list
```

## Reference Patterns

### In Conversation/Documentation

**Full Path** (most explicit):
```
experiment_manager/experiments/20251122_172313_perspective_correction
```

**Short Reference** (when context is clear):
```
@20251122_172313_perspective_correction
# or
20251122_172313_perspective_correction
```

**By Type** (when referring to latest):
```
@perspective_correction (latest)
```

### In Code

```python
# Full path
experiment_path = Path("experiment_manager/experiments/20251122_172313_perspective_correction")

# Using ExperimentTracker
from experiment_tracker.core import ExperimentTracker
tracker = ExperimentTracker()
experiment_id = "20251122_172313_perspective_correction"
tracker.add_task("Continue work", experiment_id=experiment_id)
```

### In File Paths

```bash
# Absolute from project root
experiment_manager/experiments/20251122_172313_perspective_correction/scripts/test.py

# Relative from experiment_manager
experiments/20251122_172313_perspective_correction/scripts/test.py
```

## Best Practices

### 1. Always Use Experiment ID for Specific References

✅ **Good**:
- `experiment_manager/experiments/20251122_172313_perspective_correction`
- `@20251122_172313_perspective_correction`

❌ **Avoid**:
- "the perspective correction experiment" (ambiguous)
- "experiment_manager/experiments/perspective_correction" (may not exist)

### 2. Check Current Experiment

Before starting work, check what's active:

```bash
cat experiment_manager/experiments/.current
```

### 3. Use Descriptive Intentions

When starting experiments, use clear intentions:

✅ **Good**: `"Improve corner detection accuracy on rembg-processed images"`
❌ **Avoid**: `"Fix bugs"` or `"Testing"`

### 4. Document Experiment Context

When resuming, check:
- `.metadata/state.yml` - Current status
- `.metadata/tasks.yml` - Pending tasks
- `.metadata/decisions.yml` - Key decisions
- `assessments/` - Previous findings

### 5. Use Experiment Type for Grouping

Experiments of the same type can be grouped:
- `perspective_correction` - All perspective correction experiments
- `data_augmentation` - All data augmentation experiments
- `training_run` - All training experiments

## Workflow Examples

### Starting Fresh

```bash
# 1. Start new experiment
cd experiment_manager
python scripts/start-experiment.py \
    --type perspective_correction \
    --intention "Test improved corner detection"

# 2. Verify it's active
cat experiments/.current
# Output: 20251123_130000_perspective_correction

# 3. Start working
python scripts/add-task.py --description "Run comprehensive tests"
```

### Resuming Work

```bash
# 1. List available experiments
ls experiment_manager/experiments/

# 2. Check experiment status
cat experiment_manager/experiments/20251122_172313_perspective_correction/.metadata/state.yml

# 3. Resume (if resume script exists)
python experiment_manager/scripts/resume-experiment.py \
    --id 20251122_172313_perspective_correction

# 4. Continue work
python experiment_manager/scripts/add-task.py --description "Continue from where we left off"
```

### Switching Between Experiments

```bash
# 1. Stash current (if needed)
python experiment_manager/scripts/stash-incomplete.py

# 2. Resume different experiment
python experiment_manager/scripts/resume-experiment.py \
    --id 20251120_100000_data_augmentation

# 3. Work on new experiment
# ... do work ...

# 4. Switch back
python experiment_manager/scripts/resume-experiment.py \
    --id 20251122_172313_perspective_correction
```

## Quick Reference

| Action | Command | Reference Format |
|--------|---------|-----------------|
| Start new | `start-experiment.py --type X --intention Y` | Creates new ID |
| Resume | `resume-experiment.py --id <ID>` | `@<ID>` or full path |
| List | `ls experiments/` | See all IDs |
| Current | `cat experiments/.current` | Active experiment ID |
| Reference | In docs: `@<ID>` or full path | `experiment_manager/experiments/<ID>` |

## Recommended Scripts to Add

1. **`resume-experiment.py`** - Resume by ID or type
2. **`list-experiments.py`** - List all experiments with status
3. **`show-experiment.py`** - Show experiment details and status
4. **`switch-experiment.py`** - Switch between experiments (stash current)

---

**Last Updated**: 2025-11-23 13:00 (KST)

