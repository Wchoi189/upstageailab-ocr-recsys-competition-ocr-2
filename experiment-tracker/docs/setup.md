# Setup Guide

## Prerequisites

- Python 3.8+
- `pyyaml` package (install via `pip install pyyaml`)

## Installation

1.  Clone the repository.
2.  Navigate to `experiment-tracker`.
3.  Ensure scripts are executable: `chmod +x scripts/*.py`.

## Usage

Add the `scripts` directory to your PATH for easier access, or run them directly.

## Integration

To integrate with your existing Python code:

```python
import sys
sys.path.append("/path/to/experiment-tracker/src")
from experiment_tracker.core import ExperimentTracker

tracker = ExperimentTracker()
tracker.start_experiment("My intention", "custom_type")
tracker.record_artifact("path/to/file", {"key": "value"})
```
