# Experiment Tracker - EDS v1.0

Experiment Documentation Standard v1.0 with CLI tool and automated enforcement.

## Status

✅ **EDS v1.0 Implementation Complete**
- 100% compliance achieved across all experiments
- Pre-commit hooks operational (blocking violations)
- CLI tool (ETK v1.0.0) ready for production use
- Database integration active (FTS5 search, analytics)
- AI-only documentation standard enforced

## Key Features

- **EDS v1.0 Compliance**: Enforced via pre-commit hooks
- **CLI Tool (ETK)**: 9 commands for artifact management
- **Database Integration**: SQLite with FTS5 full-text search
- **Query Interface**: Search artifacts across all experiments
- **Analytics Dashboard**: Experiment statistics and trends
- **Auto-generation**: Frontmatter, timestamps, slugs
- **Validation**: Automated compliance checking
- **Safe State Management**: YAML-based state files with corruption prevention

## Safe State Management

State files (`state.yml`) are managed automatically by the ETK CLI.

```bash
# Sync with metadata to database
etk sync --all

# Check experiment status
etk status
```

**Important**: Never edit `state.yml` files directly. Use ETK commands to prevent corruption.

## Quick Start

### Install CLI Tool

```bash
bash install-etk.sh
source ~/.bashrc  # or ~/.zshrc
```

### Create Experiment

```bash
etk init my_experiment_name --intention "Experiment description" --tags "tag1,tag2"
# Note: folder name includes timestamp
cd experiments/20251217_HHMMSS_my_experiment_name
```

### Create Artifacts

```bash
etk create assessment "Initial baseline evaluation"
etk create report "Performance metrics analysis" --metrics "accuracy,f1,latency"
etk create guide "Setup and configuration instructions"
etk create script "Automation script documentation"
```

### Check Status

```bash
etk status              # Current experiment
etk list                # All experiments
etk validate            # Validate current
etk validate --all      # Validate all
```

### Database Integration

```bash
etk sync --all          # Sync all artifacts to database
etk query "search term" # Full-text search across artifacts
etk analytics           # View analytics dashboard
```

## Features

✅ **CLI Tool (ETK)**: Unified interface for artifact lifecycle
✅ **Pre-Commit Hooks**: Automated enforcement
✅ **Compliance Dashboard**: Monitor adherence
✅ **Database Integration**: SQLite with FTS5
✅ **Legacy Fixer**: Automated migration

## Directory Structure

```
experiment_manager/
├── AGENTS.yaml                 # AI quick-reference (~50 tokens)
├── .config/                    # System configuration
├── .templates/                 # Reusable templates (YAML)
├── src/etk/                    # Python package
│   ├── cli.py                  # ETK CLI implementation
│   ├── core.py                 # ExperimentTracker class
│   └── utils/                  # Path/state utilities
├── scripts/                    # Automation scripts
│   └── SCRIPTS.yaml            # Script consolidation map
└── experiments/
    └── YYYYMMDD_HHMMSS_<id>/   # Experiment containers
        ├── .metadata/          # Artifact subdirectories
        └── manifest.json       # Experiment state (JSON)
```

## Context Tracking

Enhance your experiments with rich context tracking using the CLI:

-   **Add Tasks**:
    ```bash
    etk task "Implement robust validation"
    ```

-   **Record Decisions**:
    ```bash
    etk create assessment "Decision: Use YAML for metadata" --type decision
    ```

-   **Log Insights**:
    ```bash
    etk create assessment "Insight: Perspective correction fails on low contrast images" --type insight
    ```

## Integration

To use ETK components in python:

```python
from etk.core import ExperimentTracker

tracker = ExperimentTracker()
# ...
```

## Path Resolution

**Always use path_utils for path resolution in experiment scripts:**

```python
from pathlib import Path
import sys

# Add etk src to path
etk_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(etk_root / "src"))

from etk.utils.path_utils import setup_script_paths

# Auto-detect experiment ID and setup paths
TRACKER_ROOT, EXPERIMENT_ID, EXPERIMENT_PATHS = setup_script_paths(Path(__file__))
```

**Benefits**: Auto-detects experiment ID, eliminates hardcoded paths, portable across environments.

## Experiment Reference Guide

### Naming Conventions

- **Experiment-Tracker**: `experiment-tracker/` (from project root)
- **Experiment ID Format**: `YYYYMMDD_HHMMSS_<type>` (e.g., `20251122_172313_perspective_correction`)
- **Full Path**: `experiment-tracker/experiments/<experiment_id>/`

### Reference Patterns

**In conversation/documentation**:
- Full: `experiment-tracker/experiments/20251122_172313_perspective_correction`
- Short: `@20251122_172313_perspective_correction` or just the ID

**In code**:
```python
experiment_id = "20251122_172313_perspective_correction"
tracker.add_task("Continue work", experiment_id=experiment_id)
```

See [docs/experiment_workflow.md](docs/experiment_workflow.md) for detailed workflow guide.
