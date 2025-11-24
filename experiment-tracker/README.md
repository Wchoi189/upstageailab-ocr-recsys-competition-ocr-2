# Experiment Tracker

A robust, AI-Agent-integrated experiment tracking system that transforms chaotic experimental workflows into systematic, reusable knowledge assets.

## Features

- **Interruption Resilience**: Automatically stashes incomplete experiments.
- **AI-Human Collaboration**: Explicit intention declaration and feedback mechanisms.
- **Standardized Structure**: Consistent folder hierarchy and metadata.
- **CLI Tools**: Easy-to-use scripts for managing experiments.

## Quick Start

1.  **Start an experiment**:
    ```bash
    ./scripts/start-experiment.py --type perspective_correction --intention "Analyze failure cases in urban scene correction"
    ```

2.  **Resume an experiment**:
    ```bash
    # Resume by ID
    ./scripts/resume-experiment.py --id 20251122_172313_perspective_correction

    # Resume latest of type
    ./scripts/resume-experiment.py --type perspective_correction

    # List all experiments
    ./scripts/resume-experiment.py --list

    # Show current experiment
    ./scripts/resume-experiment.py --current
    ```

3.  **Record an artifact**:
    ```bash
    ./scripts/record-artifact.py --path path/to/image.jpg --type poor_performance --metadata '{"technique": "homography_basic"}'
    ```

4.  **Generate an assessment**:
    ```bash
    ./scripts/generate-assessment.py --template perspective_analysis --verbose minimal
    ```

5.  **Generate an incident report**:
    ```bash
    ./scripts/generate-incident-report.py --title "Perspective Overshoot" --severity high --tags "perspective,corner-detection"
    ```

6.  **Export experiment**:
    ```bash
    ./scripts/export-experiment.py --format archive --destination ./exports
    ```

## Directory Structure

```
experiment-tracker/
├── .config/                    # System configuration
├── .schemas/                   # Validation schemas
├── .templates/                 # Reusable templates
├── experiments/
│   ├── YYYYMMDD_HHMMSS_<id>/  # Individual experiment containers
│   │   ├── .metadata/         # Hidden metadata folder
│   │   ├── artifacts/         # Generated outputs
│   │   ├── assessments/       # Multiple markdown assessments
│   │   ├── incident_reports/  # Defect analysis and incident reports
│   │   └── state.json         # Internal experiment state
├── knowledge-base/            # Reusable insights and patterns
└── scripts/                   # CLI automation tools
```

## Configuration

Configuration is located in `.config/config.yaml`. You can customize experiment types, paths, and other settings.

## Context Tracking

Enhance your experiments with rich context tracking:

-   **Add Tasks**:
    ```bash
    ./scripts/add-task.py --description "Implement robust validation" --status in_progress
    ```

-   **Record Decisions**:
    ```bash
    ./scripts/record-decision.py --decision "Use YAML for metadata" --rationale "Human readable and easy to merge"
    ```

-   **Log Insights**:
    ```bash
    ./scripts/log-insight.py --insight "Perspective correction fails on low contrast images" --category observation
    ```

## Integration

Automatically track script execution using the Python decorator:

```python
from experiment_tracker import track_experiment

@track_experiment()
def main():
    # Your experiment code here
    pass
```

The decorator automatically detects the experiment ID (if running inside an experiment folder) and logs execution start/end and any errors.

## Path Resolution

**Always use path_utils for path resolution in experiment scripts:**

```python
from pathlib import Path
import sys

# Add tracker src to path
tracker_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(tracker_root / "src"))

from experiment_tracker.utils.path_utils import setup_script_paths

# Auto-detect experiment ID and setup paths
TRACKER_ROOT, EXPERIMENT_ID, EXPERIMENT_PATHS = setup_script_paths(Path(__file__))

# Use TRACKER_ROOT.parent for workspace root (replaces hardcoded paths)
workspace_root = TRACKER_ROOT.parent
data_dir = workspace_root / "data" / "datasets" / "images" / "train"
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
