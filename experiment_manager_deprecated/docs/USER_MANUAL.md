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

State files (`state.yml`) use YAML format for better safety and readability:

```bash
# Validate state file
python scripts/safe_state_manager.py experiments/exp_id/state.yml --validate

# Safe editing (use instead of direct file edits)
python scripts/safe_state_manager.py experiments/exp_id/state.yml --set status completed

# Sync with metadata
etk sync --all
```

**Important**: Never edit `state.yml` files directly. Use the safe state manager or ETK commands to prevent corruption.

See [Safe State Management Documentation](docs/safe-state-management.md) for complete usage guide.

## Quick Start

### Install CLI Tool

```bash
bash install-etk.sh
source ~/.bashrc  # or ~/.zshrc
```

### Create Experiment

```bash
etk init my_experiment_name --description "Experiment description" --tags "tag1,tag2"
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

### Status and Validation

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

✅ **CLI Tool (ETK)**: 9 commands for complete artifact lifecycle
✅ **Pre-Commit Hooks**: Automated enforcement (blocks ALL-CAPS, validates frontmatter)
✅ **Compliance Dashboard**: Monitor adherence across all experiments
✅ **Database Integration**: SQLite with FTS5 full-text search
✅ **Query Interface**: Search artifacts across all experiments
✅ **Analytics Dashboard**: Experiment statistics and trends
✅ **Legacy Fixer**: Automated migration of old artifacts
✅ **Integration Tests**: Comprehensive validation suite

## Critical Rules

1. ❌ **NEVER** create artifacts manually (use `etk create`)
2. ❌ **NEVER** use ALL-CAPS filenames (pre-commit blocks)
3. ❌ **NEVER** write user-facing documentation (AI-only standard)
4. ✅ **ALWAYS** use CLI tool for artifact creation
5. ✅ **ALWAYS** sync to database after changes (`etk sync --all`)
6. ✅ **ALWAYS** run from experiment directory or specify `--experiment`

## Validation

**Automated** (pre-commit hooks):
- Blocks ALL-CAPS filenames
- Requires `.metadata/` structure
- Validates YAML frontmatter

**Manual**:
```bash
etk validate              # Current experiment
etk validate --all        # All experiments
```

## Tools

| Tool | Purpose | Location |
|------|---------|----------|
| `etk` | CLI tool | `etk.py` |
| Compliance Dashboard | Monitor compliance | `.ai-instructions/tier4-workflows/generate-compliance-report.py` |
| Legacy Fixer | Migrate old artifacts | `.ai-instructions/tier4-workflows/fix-legacy-artifacts.py` |
| ALL-CAPS Renamer | Rename violations | `.ai-instructions/tier4-workflows/rename-all-caps-files.py` |
| Integration Tests | Test hooks | `.ai-instructions/tier4-workflows/tests/test_precommit_hooks.py` |

## Documentation

- **Specification**: `.ai-instructions/schema/eds-v1.0-spec.yaml`
- **Templates**: `.ai-instructions/tier2-framework/artifact-catalog.yaml`
- **Rules**: `.ai-instructions/tier1-sst/*.yaml`
- **Database Roadmap**: `.ai-instructions/tier2-framework/database-integration-roadmap.md`
- **CHANGELOG**: `CHANGELOG.md`

## Help

```bash
etk --help
etk create --help
etk init --help
```

## Directory Structure

```
experiment_manager/
├── AGENTS.yaml                 # AI quick-reference (~50 tokens)
├── .config/                    # System configuration
├── .templates/                 # Reusable templates (YAML)
├──                     # Python package
│   ├── cli.py                  # ETK CLI implementation
│   ├── core.py                 # ExperimentTracker class
│   └── utils/                  # Path/state utilities
├── scripts/                    # Automation scripts
│   ├── SCRIPTS.yaml            # Script consolidation map
│   └── *.py                    # Legacy/specialized scripts
└── experiments/
    └── YYYYMMDD_HHMMSS_<id>/   # Experiment containers
        ├── .metadata/          # Artifact subdirectories
        └── state.yml           # Experiment state (YAML)
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

- **experiment_manager**: `experiment_manager/` (from project root)
- **Experiment ID Format**: `YYYYMMDD_HHMMSS_<type>` (e.g., `20251122_172313_perspective_correction`)
- **Full Path**: `experiment_manager/experiments/<experiment_id>/`

### Reference Patterns

**In conversation/documentation**:
- Full: `experiment_manager/experiments/20251122_172313_perspective_correction`
- Short: `@20251122_172313_perspective_correction` or just the ID

**In code**:
```python
experiment_id = "20251122_172313_perspective_correction"
tracker.add_task("Continue work", experiment_id=experiment_id)
```

See [docs/experiment_workflow.md](docs/experiment_workflow.md) for detailed workflow guide.
