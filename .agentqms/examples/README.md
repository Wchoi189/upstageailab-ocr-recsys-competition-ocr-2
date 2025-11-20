# State Tracking Examples

This directory contains working examples demonstrating the Agent Framework State Tracking system.

## Examples

### 01_basic_usage.py

Demonstrates basic state tracking usage:
- Creating artifacts with automatic state tracking
- Querying artifacts
- Viewing statistics

**Run:**
```bash
uv run python .agentqms/examples/01_basic_usage.py
```

### 02_session_tracking.py

Demonstrates session lifecycle management:
- Starting and ending sessions
- Tracking goals, outcomes, and challenges
- Restoring context from previous sessions

**Run:**
```bash
uv run python .agentqms/examples/02_session_tracking.py
```

### 03_artifact_relationships.py

Demonstrates artifact relationships and dependencies:
- Adding dependencies between artifacts
- Querying dependency trees
- Status propagation

**Run:**
```bash
uv run python .agentqms/examples/03_artifact_relationships.py
```

### 04_querying_state.py

Demonstrates various ways to query state:
- Finding artifacts by type, status, tags
- Searching sessions by criteria
- Analyzing statistics

**Run:**
```bash
uv run python .agentqms/examples/04_querying_state.py
```

## Prerequisites

Ensure that:
1. The `.agentqms/` directory is initialized
2. `.agentqms/config.yaml` exists
3. State tracking is enabled

## Notes

- Examples can be run in any order
- Each example is self-contained
- Examples will modify the state (create artifacts, sessions)
- Check `.agentqms/state.json` to see the state changes
