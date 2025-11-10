---
title: Tracking Database API Reference
date: 2025-01-27
type: api_reference
category: tracking
status: active
version: 1.0.0
---

# Tracking Database API Reference

Complete API documentation for the SQLite tracking database system used for development/debug tracking and experiment management.

## Overview

The tracking database (`data/ops/tracking.db`) provides a structured way to track:
- **Feature Plans**: Implementation plans with tasks
- **Refactors**: Code refactoring efforts
- **Debug Sessions**: Debugging context and notes
- **Experiments**: Experimental runs with parameters, metrics, and outcomes
- **Summaries**: Ultra-concise summaries of entities

## Database Schema

### Tables

#### `feature_plans`
Tracks feature implementation plans.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `key` | TEXT | UNIQUE, NOT NULL | Unique plan identifier (e.g., `2025-1106_feat_vector-cache`) |
| `title` | TEXT | NOT NULL | Plan title |
| `status` | TEXT | NOT NULL, CHECK | Status: `pending`, `in_progress`, `paused`, `completed`, `cancelled` |
| `owner` | TEXT | | Plan owner |
| `started_at` | TEXT | | ISO8601 UTC timestamp when started |
| `updated_at` | TEXT | | ISO8601 UTC timestamp of last update |

#### `plan_tasks`
Tasks associated with feature plans.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `plan_id` | INTEGER | NOT NULL, FK → `feature_plans.id` | Parent plan ID |
| `title` | TEXT | NOT NULL | Task title |
| `status` | TEXT | NOT NULL, CHECK | Status: `pending`, `in_progress`, `paused`, `completed`, `cancelled` |
| `notes` | TEXT | | Optional task notes |
| `created_at` | TEXT | NOT NULL | ISO8601 UTC timestamp |
| `updated_at` | TEXT | | ISO8601 UTC timestamp of last update |

#### `refactors`
Tracks code refactoring efforts.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `key` | TEXT | UNIQUE, NOT NULL | Unique refactor identifier |
| `title` | TEXT | NOT NULL | Refactor title |
| `status` | TEXT | NOT NULL, CHECK | Status: `pending`, `in_progress`, `paused`, `completed`, `cancelled` |
| `notes` | TEXT | | Optional refactor notes |
| `started_at` | TEXT | | ISO8601 UTC timestamp when started |
| `updated_at` | TEXT | | ISO8601 UTC timestamp of last update |

#### `debug_sessions`
Debugging sessions with context.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `key` | TEXT | UNIQUE, NOT NULL | Unique session identifier |
| `title` | TEXT | NOT NULL | Session title |
| `status` | TEXT | NOT NULL, CHECK | Status: `pending`, `in_progress`, `paused`, `completed`, `cancelled` |
| `hypothesis` | TEXT | | Debugging hypothesis |
| `scope` | TEXT | | Scope of debugging |
| `started_at` | TEXT | NOT NULL | ISO8601 UTC timestamp |
| `updated_at` | TEXT | | ISO8601 UTC timestamp of last update |

#### `debug_notes`
Notes associated with debug sessions.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `session_id` | INTEGER | NOT NULL, FK → `debug_sessions.id` | Parent session ID |
| `note` | TEXT | NOT NULL | Note content |
| `created_at` | TEXT | NOT NULL | ISO8601 UTC timestamp |

#### `experiments`
Experimental efforts.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `key` | TEXT | UNIQUE, NOT NULL | Unique experiment identifier |
| `title` | TEXT | NOT NULL | Experiment title |
| `objective` | TEXT | | Experiment objective |
| `owner` | TEXT | | Experiment owner |
| `status` | TEXT | NOT NULL, CHECK | Status: `pending`, `in_progress`, `paused`, `completed`, `cancelled` |
| `created_at` | TEXT | NOT NULL | ISO8601 UTC timestamp |
| `updated_at` | TEXT | | ISO8601 UTC timestamp of last update |

#### `experiment_runs`
Individual runs within an experiment.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `experiment_id` | INTEGER | NOT NULL, FK → `experiments.id` | Parent experiment ID |
| `run_no` | INTEGER | NOT NULL | Run number (1-indexed) |
| `params_json` | TEXT | NOT NULL | JSON-encoded parameters |
| `metrics_json` | TEXT | | JSON-encoded metrics |
| `outcome` | TEXT | NOT NULL, CHECK | Outcome: `pass`, `fail`, `inconclusive` |
| `created_at` | TEXT | NOT NULL | ISO8601 UTC timestamp |
| UNIQUE(`experiment_id`, `run_no`) | | | One run number per experiment |

#### `experiment_artifacts`
Artifacts linked to experiments (plots, notebooks, results).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `experiment_id` | INTEGER | NOT NULL, FK → `experiments.id` | Parent experiment ID |
| `run_id` | INTEGER | FK → `experiment_runs.id` | Optional: specific run ID |
| `type` | TEXT | NOT NULL | Artifact type (e.g., `result`, `plot`, `notebook`) |
| `path` | TEXT | NOT NULL | File path to artifact |
| `created_at` | TEXT | NOT NULL | ISO8601 UTC timestamp |

#### `summaries`
Ultra-concise summaries of entities.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | INTEGER | PRIMARY KEY | Auto-increment ID |
| `entity_type` | TEXT | NOT NULL | Entity type (e.g., `plan`, `experiment`, `debug`) |
| `entity_id` | INTEGER | NOT NULL | Entity ID |
| `style` | TEXT | NOT NULL | Summary style (e.g., `short`, `delta`) |
| `text` | TEXT | NOT NULL | Summary text (≤280 chars for `short` style) |
| `created_at` | TEXT | NOT NULL | ISO8601 UTC timestamp |

## Core Functions

### Database Initialization

#### `init_db() -> None`

Initializes the database schema. Creates all tables if they don't exist. Idempotent - safe to call multiple times.

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import init_db

init_db()
```

#### `get_connection(readonly: bool = False) -> sqlite3.Connection`

Gets a database connection. Auto-creates parent directory if needed.

**Parameters:**
- `readonly` (bool): If `True`, raises `FileNotFoundError` if DB doesn't exist. Default: `False`.

**Returns:**
- `sqlite3.Connection`: Connection with `row_factory=sqlite3.Row`.

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import get_connection

conn = get_connection()
# Use connection...
conn.close()
```

## Feature Plans API

### `upsert_feature_plan(key: str, title: str, owner: str | None = None) -> int`

Creates or updates a feature plan. If `key` exists, updates title and owner. Otherwise, creates new plan with status `pending`.

**Parameters:**
- `key` (str): Unique plan identifier (e.g., `2025-1106_feat_vector-cache`)
- `title` (str): Plan title
- `owner` (str | None): Optional plan owner

**Returns:**
- `int`: Plan ID

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import upsert_feature_plan

plan_id = upsert_feature_plan(
    key="2025-1106_feat_vector-cache",
    title="Vector Cache Implementation",
    owner="alice"
)
```

### `set_plan_status(key: str, status: str) -> None`

Updates a plan's status. Sets `started_at` if transitioning to `in_progress`.

**Parameters:**
- `key` (str): Plan key
- `status` (str): One of: `pending`, `in_progress`, `paused`, `completed`, `cancelled`

**Raises:**
- `ValueError`: If status is invalid
- `KeyError`: If plan not found

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import set_plan_status

set_plan_status("2025-1106_feat_vector-cache", "in_progress")
```

### `add_plan_task(plan_key: str, title: str, notes: str | None = None) -> int`

Adds a task to a plan. Task starts with status `pending`.

**Parameters:**
- `plan_key` (str): Parent plan key
- `title` (str): Task title
- `notes` (str | None): Optional task notes

**Returns:**
- `int`: Task ID

**Raises:**
- `KeyError`: If plan not found

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import add_plan_task

task_id = add_plan_task(
    plan_key="2025-1106_feat_vector-cache",
    title="Implement cache lookup",
    notes="Use LRU eviction"
)
```

### `set_task_done(task_id: int) -> None`

Marks a task as completed.

**Parameters:**
- `task_id` (int): Task ID

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import set_task_done

set_task_done(42)
```

### `get_plan_status(key: str | None = None) -> list[dict[str, Any]]`

Gets plan status with open task counts. Auto-initializes DB if missing.

**Parameters:**
- `key` (str | None): If provided, returns status for specific plan. Otherwise, returns all plans.

**Returns:**
- `list[dict[str, Any]]`: List of dicts with keys: `key`, `title`, `status`, `open_tasks`

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import get_plan_status

# Get all plans
all_plans = get_plan_status()

# Get specific plan
plan = get_plan_status("2025-1106_feat_vector-cache")
```

## Refactors API

### `upsert_refactor(key: str, title: str, notes: str | None = None) -> int`

Creates or updates a refactor. If `key` exists, updates title and notes. Otherwise, creates new refactor with status `pending`.

**Parameters:**
- `key` (str): Unique refactor identifier
- `title` (str): Refactor title
- `notes` (str | None): Optional refactor notes

**Returns:**
- `int`: Refactor ID

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import upsert_refactor

refactor_id = upsert_refactor(
    key="2025-1106_refactor_api-cleanup",
    title="API Cleanup",
    notes="Remove deprecated endpoints"
)
```

### `set_refactor_status(key: str, status: str) -> None`

Updates a refactor's status. Sets `started_at` if transitioning to `in_progress`.

**Parameters:**
- `key` (str): Refactor key
- `status` (str): One of: `pending`, `in_progress`, `paused`, `completed`, `cancelled`

**Raises:**
- `ValueError`: If status is invalid
- `KeyError`: If refactor not found

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import set_refactor_status

set_refactor_status("2025-1106_refactor_api-cleanup", "completed")
```

## Debug Sessions API

### `create_debug_session(key: str, title: str, hypothesis: str = "", scope: str = "") -> int`

Creates a debug session. Uses `INSERT OR IGNORE` - if key exists, returns existing session ID.

**Parameters:**
- `key` (str): Unique session identifier
- `title` (str): Session title
- `hypothesis` (str): Debugging hypothesis (default: `""`)
- `scope` (str): Scope of debugging (default: `""`)

**Returns:**
- `int`: Session ID

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import create_debug_session

session_id = create_debug_session(
    key="2025-1106_debug_memory-leak",
    title="Memory Leak Investigation",
    hypothesis="Unclosed file handles",
    scope="File I/O operations"
)
```

### `add_debug_note(session_key: str, note: str) -> int`

Adds a note to a debug session.

**Parameters:**
- `session_key` (str): Session key
- `note` (str): Note content

**Returns:**
- `int`: Note ID

**Raises:**
- `KeyError`: If session not found

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import add_debug_note

note_id = add_debug_note(
    session_key="2025-1106_debug_memory-leak",
    note="Found 3 unclosed file handles in process_data()"
)
```

### `set_debug_status(key: str, status: str) -> None`

Updates a debug session's status.

**Parameters:**
- `key` (str): Session key
- `status` (str): One of: `pending`, `in_progress`, `paused`, `completed`, `cancelled`

**Raises:**
- `ValueError`: If status is invalid

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import set_debug_status

set_debug_status("2025-1106_debug_memory-leak", "completed")
```

## Experiments API

### `upsert_experiment(key: str, title: str, objective: str = "", owner: str | None = None) -> int`

Creates or updates an experiment. If `key` exists, updates title, objective, and owner. Otherwise, creates new experiment with status `in_progress`.

**Parameters:**
- `key` (str): Unique experiment identifier
- `title` (str): Experiment title
- `objective` (str): Experiment objective (default: `""`)
- `owner` (str | None): Optional experiment owner

**Returns:**
- `int`: Experiment ID

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import upsert_experiment

exp_id = upsert_experiment(
    key="2025-1106_exp_learning-rate",
    title="Learning Rate Tuning",
    objective="Find optimal learning rate for model convergence",
    owner="bob"
)
```

### `add_experiment_run(experiment_key: str, run_no: int, params: dict[str, Any], metrics: dict[str, Any] | None, outcome: str) -> int`

Adds or replaces an experiment run. Uses `INSERT OR REPLACE` - if run exists, updates it.

**Parameters:**
- `experiment_key` (str): Experiment key
- `run_no` (int): Run number (1-indexed)
- `params` (dict[str, Any]): Run parameters (JSON-serialized)
- `metrics` (dict[str, Any] | None): Run metrics (JSON-serialized, optional)
- `outcome` (str): One of: `pass`, `fail`, `inconclusive`

**Returns:**
- `int`: Run ID

**Raises:**
- `ValueError`: If outcome is invalid
- `KeyError`: If experiment not found

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import add_experiment_run

run_id = add_experiment_run(
    experiment_key="2025-1106_exp_learning-rate",
    run_no=1,
    params={"lr": 0.001, "batch_size": 32},
    metrics={"loss": 0.45, "accuracy": 0.92},
    outcome="pass"
)
```

### `link_experiment_artifact(experiment_key: str, type_: str, path: str, run_no: int | None = None) -> int`

Links an artifact to an experiment (optionally to a specific run).

**Parameters:**
- `experiment_key` (str): Experiment key
- `type_` (str): Artifact type (e.g., `result`, `plot`, `notebook`)
- `path` (str): File path to artifact
- `run_no` (int | None): Optional run number to link to specific run

**Returns:**
- `int`: Artifact ID

**Raises:**
- `KeyError`: If experiment or run not found

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import link_experiment_artifact

artifact_id = link_experiment_artifact(
    experiment_key="2025-1106_exp_learning-rate",
    type_="plot",
    path="results/lr_tuning_plot.png",
    run_no=1
)
```

### `get_experiment_runs_export() -> list[dict[str, Any]]`

Gets all experiment runs for export. Auto-initializes DB if missing.

**Returns:**
- `list[dict[str, Any]]`: List of dicts with keys: `experiment_key`, `run_no`, `params_json`, `metrics_json`, `outcome`, `created_at`

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import get_experiment_runs_export
import json

runs = get_experiment_runs_export()
for run in runs:
    params = json.loads(run["params_json"])
    metrics = json.loads(run["metrics_json"] or "{}")
    print(f"{run['experiment_key']} run {run['run_no']}: {run['outcome']}")
```

## Summaries API

### `save_summary(entity_type: str, entity_id: int, style: str, text: str) -> int`

Saves an ultra-concise summary for an entity.

**Parameters:**
- `entity_type` (str): Entity type (e.g., `plan`, `experiment`, `debug`)
- `entity_id` (int): Entity ID
- `style` (str): Summary style (e.g., `short`, `delta`)
- `text` (str): Summary text (must be ≤280 chars if `style="short"`)

**Returns:**
- `int`: Summary ID

**Raises:**
- `ValueError`: If `style="short"` and text > 280 characters

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import save_summary

summary_id = save_summary(
    entity_type="experiment",
    entity_id=42,
    style="short",
    text="LR=0.001 achieved 92% accuracy; optimal for this dataset."
)
```

## Status Values

All status fields use the same enum values:

- `pending`: Not yet started
- `in_progress`: Currently active
- `paused`: Temporarily paused
- `completed`: Finished successfully
- `cancelled`: Abandoned

## Outcome Values

Experiment run outcomes:

- `pass`: Run succeeded
- `fail`: Run failed
- `inconclusive`: Results were inconclusive

## Error Handling

All functions that query by key will raise `KeyError` if the entity is not found, except:
- `get_plan_status()`: Returns empty list if not found
- `get_experiment_runs_export()`: Returns empty list if no runs exist

Functions that set status will raise `ValueError` if the status is invalid.

## Timestamps

All timestamps are stored as ISO8601 UTC strings (e.g., `2025-01-27T14:30:00+00:00`). Generated using `datetime.now(UTC).isoformat(timespec="seconds")`.

## JSON Fields

- `params_json` and `metrics_json` in `experiment_runs` are stored as JSON strings
- Use `json.dumps()` when storing and `json.loads()` when reading
- `ensure_ascii=False` is used when serializing to preserve Unicode characters

## Usage Examples

### Complete Workflow: Feature Plan

```python
from scripts.agent_tools.utilities.tracking.db import (
    init_db,
    upsert_feature_plan,
    set_plan_status,
    add_plan_task,
    set_task_done,
    get_plan_status
)

# Initialize database
init_db()

# Create plan
plan_id = upsert_feature_plan(
    key="2025-1106_feat_vector-cache",
    title="Vector Cache Implementation",
    owner="alice"
)

# Start plan
set_plan_status("2025-1106_feat_vector-cache", "in_progress")

# Add tasks
task1 = add_plan_task(
    plan_key="2025-1106_feat_vector-cache",
    title="Design cache structure"
)
task2 = add_plan_task(
    plan_key="2025-1106_feat_vector-cache",
    title="Implement LRU eviction"
)

# Complete task
set_task_done(task1)

# Check status
status = get_plan_status("2025-1106_feat_vector-cache")
print(f"Plan: {status[0]['title']}, Open tasks: {status[0]['open_tasks']}")
```

### Complete Workflow: Experiment

```python
from scripts.agent_tools.utilities.tracking.db import (
    upsert_experiment,
    add_experiment_run,
    link_experiment_artifact,
    get_experiment_runs_export
)
import json

# Create experiment
exp_id = upsert_experiment(
    key="2025-1106_exp_learning-rate",
    title="Learning Rate Tuning",
    objective="Find optimal learning rate",
    owner="bob"
)

# Add runs
add_experiment_run(
    experiment_key="2025-1106_exp_learning-rate",
    run_no=1,
    params={"lr": 0.001, "batch_size": 32},
    metrics={"loss": 0.45, "accuracy": 0.92},
    outcome="pass"
)

add_experiment_run(
    experiment_key="2025-1106_exp_learning-rate",
    run_no=2,
    params={"lr": 0.01, "batch_size": 32},
    metrics={"loss": 0.52, "accuracy": 0.88},
    outcome="fail"
)

# Link artifact
link_experiment_artifact(
    experiment_key="2025-1106_exp_learning-rate",
    type_="plot",
    path="results/lr_comparison.png"
)

# Export runs
runs = get_experiment_runs_export()
for run in runs:
    if run["experiment_key"] == "2025-1106_exp_learning-rate":
        params = json.loads(run["params_json"])
        print(f"Run {run['run_no']}: LR={params['lr']}, Outcome={run['outcome']}")
```

## Module Constants

### `DB_PATH: Path`

Path to the tracking database file: `data/ops/tracking.db`

**Example:**
```python
from scripts.agent_tools.utilities.tracking.db import DB_PATH

print(f"Database location: {DB_PATH}")
```

## See Also

- [Tracking CLI Quick Reference](cli_reference.md) - Tracking commands and status recipes
- [Tracking Query Module](../../scripts/agent_tools/utilities/tracking/query.py) - Ultra-concise status queries for chat integration (`get_status()`)
- [Tracking CLI Module](../../scripts/agent_tools/utilities/tracking/cli.py) - Command-line interface for tracking operations
- [Streamlit Tracking Dashboard](../../streamlit_app/pages/10_Tracking_Dashboard.py) - Visual dashboard
