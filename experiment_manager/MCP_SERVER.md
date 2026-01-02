# Experiment Manager MCP Server

## Overview

The Experiment Manager MCP server exposes experiment lifecycle management tools, enabling standardized experiment creation, task tracking, insight logging, and artifact organization.

## Purpose

- **Standardization**: Consistent experiment structure across all experiments
- **Organization**: Reduces chaos from excessive experiment artifacts
- **Tracking**: Automatic logging of tasks, insights, and decisions
- **Protocol Adherence**: Enforces experiment workflow standards

## Available Resources

### 1. `experiments://agent_interface`
**ETK Command Reference**

Complete reference of all ETK (Experiment Toolkit) commands with usage examples, including:
- `etk init` - Initialize experiments
- `etk status` - Check experiment status
- `etk task` - Manage tasks
- `etk log` - Log insights/decisions/failures
- `etk sync` - Sync to database

**Usage**:
```python
content = read_resource(
    ServerName="experiments",
    Uri="experiments://agent_interface"
)
```

### 2. `experiments://active_list`
**Active Experiments (Dynamic)**

Dynamically generated list of all active experiments with metadata.

**Example Response**:
```json
{
  "experiments": [
    {
      "id": "baseline-kie-v2",
      "name": "Baseline KIE v2",
      "status": "active",
      "created": "2026-01-02T15:30:00"
    },
    {
      "id": "ocr-optimization",
      "name": "OCR Pipeline Optimization",
      "status": "completed",
      "created": "2026-01-01T10:00:00"
    }
  ]
}
```

### 3. `experiments://schemas/manifest`
**Manifest JSON Schema**

JSON schema defining the structure of experiment manifests, including required fields, metadata, and validation rules.

### 4. `experiments://schemas/artifact`
**Artifact JSON Schema**

JSON schema for experiment artifacts, defining structure for assessments, reports, and other experiment outputs.

---

## Available Tools

### 1. `init_experiment`
**Initialize New Experiment**

Creates a new experiment with standardized directory structure:
```
experiments/{name}/
├── src/              # Experiment-specific code
├── artifacts/        # Generated outputs
├── data/             # Input/output data
├── .metadata/        # Internal state tracking
└── manifest.json     # Experiment metadata
```

**Parameters**:
- `name` (required): Unique experiment identifier (slug format)
- `description` (optional): Experiment description
- `tags` (optional): Comma-separated tags

**Example**:
```python
result = call_tool(
    ServerName="experiments",
    ToolName="init_experiment",
    Arguments={
        "name": "baseline-kie-v3",
        "description": "KIE baseline with optimized dataset v3",
        "tags": "kie,baseline,optimization"
    }
)
```

**Returns**:
```json
{
  "success": true,
  "experiment_id": "baseline-kie-v3",
  "output": "Created experiment: experiments/baseline-kie-v3/"
}
```

### 2. `get_experiment_status`
**Get Experiment Status**

Returns detailed status information for current or specified experiment.

**Parameters**:
- `experiment_id` (optional): Specific experiment (uses current if not specified)

**Example**:
```python
result = call_tool(
    ServerName="experiments",
    ToolName="get_experiment_status",
    Arguments={
        "experiment_id": "baseline-kie-v3"
    }
)
```

**Returns JSON-like status summary** including:
- Experiment metadata
- Active tasks
- Recent insights
- Artifact counts
- Last sync time

### 3. `add_task`
**Add Task to Experiment**

Adds a task to the experiment plan and updates tracking state.

**Parameters**:
- `description` (required): Task description
- `experiment_id` (optional): Target experiment (uses current if not specified)

**Example**:
```python
result = call_tool(
    ServerName="experiments",
    ToolName="add_task",
    Arguments={
        "description": "Optimize data loading pipeline for 2x speedup",
        "experiment_id": "baseline-kie-v3"
    }
)
```

**Returns**:
```json
{
  "success": true,
  "output": "Added task: Optimize data loading pipeline for 2x speedup"
}
```

### 4. `log_insight`
**Log Insight, Decision, or Failure**

Logs a key finding, decision, or failure to the experiment for future reference.

**Parameters**:
- `insight` (required): The insight/decision/failure to log
- `type` (optional): One of "insight" (default), "decision", "failure"
- `experiment_id` (optional): Target experiment (uses current if not specified)

**Example - Insight**:
```python
result = call_tool(
    ServerName="experiments",
    ToolName="log_insight",
    Arguments={
        "insight": "Batch size 32 achieves best GPU utilization at 95%",
        "type": "insight"
    }
)
```

**Example - Decision**:
```python
result = call_tool(
    ServerName="experiments",
    ToolName="log_insight",
    Arguments={
        "insight": "Switching from AdamW to Lion optimizer for better convergence",
        "type": "decision"
    }
)
```

**Example - Failure**:
```python
result = call_tool(
    ServerName="experiments",
    ToolName="log_insight",
    Arguments={
        "insight": "OOM error with batch size 64, reducing to 32",
        "type": "failure"
    }
)
```

### 5. `sync_experiment`
**Sync to Database**

Synchronizes experiment artifacts and metadata to the SQLite database, ensuring filesystem state matches database.

**Parameters**:
- `experiment_id` (optional): Target experiment (uses current if not specified)

**Example**:
```python
result = call_tool(
    ServerName="experiments",
    ToolName="sync_experiment",
    Arguments={}
)
```

**Usage**: Call after significant changes to ensure tracking database is up-to-date.

---

## Typical Workflows

### Starting a New Experiment

```python
# 1. Check active experiments
active = read_resource(
    ServerName="experiments",
    Uri="experiments://active_list"
)

# 2. Initialize new experiment
result = call_tool(
    ServerName="experiments",
    ToolName="init_experiment",
    Arguments={
        "name": "my-experiment",
        "description": "Testing new approach",
        "tags": "research,experimental"
    }
)

# 3. Add initial tasks
call_tool(
    ServerName="experiments",
    ToolName="add_task",
    Arguments={
        "description": "Prepare dataset",
        "experiment_id": "my-experiment"
    }
)

call_tool(
    ServerName="experiments",
    ToolName="add_task",
    Arguments={
        "description": "Run baseline training",
        "experiment_id": "my-experiment"
    }
)
```

### Logging Experiment Progress

```python
# Log key insights as they occur
call_tool(
    ServerName="experiments",
    ToolName="log_insight",
    Arguments={
        "insight": "Data augmentation improves F1 by 3%",
        "type": "insight"
    }
)

# Log decisions
call_tool(
    ServerName="experiments",
    ToolName="log_insight",
    Arguments={
        "insight": "Adopting cosine learning rate schedule",
        "type": "decision"
    }
)

# Log failures (for debugging later)
call_tool(
    ServerName="experiments",
    ToolName="log_insight",
    Arguments={
        "insight": "NaN loss at epoch 5, restart with gradient clipping",
        "type": "failure"
    }
)

# Sync to database periodically
call_tool(
    ServerName="experiments",
    ToolName="sync_experiment",
    Arguments={}
)
```

### Checking Experiment Status

```python
# Get current experiment status
status = call_tool(
    ServerName="experiments",
    ToolName="get_experiment_status",
    Arguments={}
)

# Get specific experiment status
status = call_tool(
    ServerName="experiments",
    ToolName="get_experiment_status",
    Arguments={"experiment_id": "baseline-kie-v3"}
)
```

---

## Directory Structure

Each experiment follows this standardized structure:

```
experiments/{experiment_id}/
├── manifest.json           # Experiment metadata (name, description, created_at, etc.)
├── src/                    # Experiment-specific source code
│   ├── train.py
│   ├── eval.py
│   └── ...
├── artifacts/              # Generated outputs
│   ├── assessments/
│   ├── reports/
│   └── ...
├── data/                   # Experiment-specific data
│   ├── input/
│   └── output/
└── .metadata/              # Internal state
    ├── tasks.json          # Task tracking
    ├── insights.json       # Logged insights
    └── ...
```

---

## Technical Details

### Server Implementation
- **Location**: `experiment_manager/mcp_server.py`
- **Dependencies**: `mcp`, `etk.factory`, `etk.core`
- **Execution**: Wraps ETK CLI commands via subprocess

### URI Scheme
All resources use the `experiments://` scheme:
- `experiments://agent_interface` - Command reference
- `experiments://active_list` - Dynamic experiment list
- `experiments://schemas/*` - JSON schemas

### Tool Execution
Tools execute ETK commands via `subprocess` with `uv run`, ensuring they run in the correct environment with all dependencies.

### Error Handling
- Command failures return stderr in error field
- Success/failure indicated by boolean `success` field
- All responses are structured JSON

---

## Integration with AgentQMS

Experiment artifacts can be managed via AgentQMS:

```python
# Create implementation plan for experiment
call_tool(
    ServerName="agentqms",
    ToolName="create_artifact",
    Arguments={
        "artifact_type": "implementation_plan",
        "name": "baseline-kie-v3-plan",
        "title": "Baseline KIE v3 Implementation Plan"
    }
)

# Log experiment progress
call_tool(
    ServerName="experiments",
    ToolName="log_insight",
    Arguments={
        "insight": "Implementation plan created and validated",
        "type": "decision"
    }
)
```

---

## Benefits

✅ **Standardized Structure**: Every experiment follows same layout
✅ **Automatic Tracking**: Tasks and insights logged to database
✅ **Reduced Chaos**: Organized artifact management
✅ **Quick Status**: Instant experiment status checks
✅ **Historical Record**: All insights and decisions preserved
✅ **Easy Discovery**: List all active experiments
✅ **Protocol Enforcement**: Standards automatically applied

---

## Troubleshooting

### Command Failures
If a tool returns `success: false`, check the `error` or `output` field for details. Common issues:
- Experiment already exists
- Invalid experiment ID
- Missing required parameters

### Missing Experiments
Use `experiments://active_list` resource to see all available experiments.

### Database Sync Issues
If filesystem and database are out of sync, run:
```python
call_tool(
    ServerName="experiments",
    ToolName="sync_experiment",
    Arguments={}
)
```
