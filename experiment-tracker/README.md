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

2.  **Record an artifact**:
    ```bash
    ./scripts/record-artifact.py --path path/to/image.jpg --type poor_performance --metadata '{"technique": "homography_basic"}'
    ```

3.  **Generate an assessment**:
    ```bash
    ./scripts/generate-assessment.py --template perspective_analysis --verbose minimal
    ```

4.  **Export experiment**:
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
