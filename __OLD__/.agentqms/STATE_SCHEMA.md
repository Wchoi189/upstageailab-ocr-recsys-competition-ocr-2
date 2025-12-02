# Agent Framework State Schema Documentation

## Overview

The Agent Framework State Tracking system uses JSON-based storage to maintain persistent state across conversations, track artifacts, manage sessions, and preserve context.

## File Structure

```
.agentqms/
├── config.yaml              # Framework configuration (versioned in git)
├── state.json              # Current state (gitignored)
├── sessions/               # Session snapshots (gitignored)
│   └── session_<id>.json
├── backups/                # State backups (gitignored)
│   └── state_backup_<timestamp>.json
└── schemas/                # JSON schemas for validation
    ├── state_schema.json
    └── session_schema.json
```

## State Schema (state.json)

### Root Structure

```json
{
  "schema_version": "1.0.0",
  "last_updated": "2025-11-20T12:54:00+09:00",
  "framework": { ... },
  "current_context": { ... },
  "sessions": { ... },
  "artifacts": { ... },
  "relationships": { ... },
  "statistics": { ... }
}
```

### Framework Section

Contains framework metadata and version information.

```json
"framework": {
  "name": "upstageailab-ocr-recsys-competition-ocr-2",
  "version": "1.0.0"
}
```

### Current Context Section

Tracks the current active state of the framework.

```json
"current_context": {
  "active_session_id": "session-123",
  "current_branch": "main",
  "current_phase": "phase-2",
  "active_artifacts": [
    "artifacts/plan.md",
    "artifacts/assessment.md"
  ],
  "pending_tasks": []
}
```

**Fields:**
- `active_session_id`: ID of currently active session (null if no active session)
- `current_branch`: Current git branch
- `current_phase`: Current project phase
- `active_artifacts`: List of artifact paths currently being worked on
- `pending_tasks`: List of pending task identifiers

### Sessions Section

Tracks session history and active session.

```json
"sessions": {
  "total_count": 10,
  "active_session": "session-123",
  "session_history": [
    "session-123",
    "session-122",
    "session-121"
  ]
}
```

**Fields:**
- `total_count`: Total number of sessions
- `active_session`: ID of currently active session
- `session_history`: List of recent session IDs (newest first)

### Artifacts Section

Tracks all artifacts in the project with their metadata.

```json
"artifacts": {
  "total_count": 25,
  "by_type": {
    "implementation_plan": 10,
    "assessment": 8,
    "architecture": 7
  },
  "by_status": {
    "draft": 12,
    "validated": 10,
    "deployed": 3
  },
  "index": [
    {
      "path": "artifacts/implementation_plans/plan.md",
      "type": "implementation_plan",
      "status": "validated",
      "created_at": "2025-11-20T10:00:00+09:00",
      "last_updated": "2025-11-20T12:00:00+09:00",
      "metadata": {
        "author": "ai-agent",
        "tags": ["state-tracking", "framework"]
      }
    }
  ]
}
```

**Artifact Object Fields:**
- `path`: Path to the artifact file
- `type`: Type of artifact (e.g., "implementation_plan", "assessment")
- `status`: Current status ("draft", "in_progress", "validated", "deployed", "deprecated")
- `created_at`: ISO 8601 timestamp of creation
- `last_updated`: ISO 8601 timestamp of last update
- `metadata`: Additional metadata (flexible object)

### Relationships Section

Tracks relationships between artifacts and sessions.

```json
"relationships": {
  "artifact_dependencies": {
    "artifacts/plan_v2.md": [
      "artifacts/plan_v1.md",
      "artifacts/assessment.md"
    ]
  },
  "session_artifacts": {
    "session-123": [
      "artifacts/plan_v2.md",
      "artifacts/implementation.md"
    ]
  }
}
```

**Fields:**
- `artifact_dependencies`: Map of artifact path to list of dependency paths
- `session_artifacts`: Map of session ID to artifacts created/modified in that session

### Statistics Section

Aggregate statistics for analytics and reporting.

```json
"statistics": {
  "total_sessions": 10,
  "total_artifacts_created": 25,
  "total_artifacts_validated": 15,
  "total_artifacts_deployed": 5,
  "last_session_timestamp": "2025-11-20T12:00:00+09:00"
}
```

## Session Snapshot Schema (sessions/session_<id>.json)

Session snapshots preserve detailed information about each session for context restoration.

```json
{
  "session_id": "session-123",
  "started_at": "2025-11-20T10:00:00+09:00",
  "ended_at": "2025-11-20T12:00:00+09:00",
  "branch": "feature/state-tracking",
  "phase": "phase-1",
  "artifacts_created": [
    "artifacts/plan.md",
    "artifacts/assessment.md"
  ],
  "artifacts_modified": [
    "artifacts/existing_plan.md"
  ],
  "context": {
    "goals": [
      "Implement state tracking module",
      "Create unit tests"
    ],
    "outcomes": [
      "Successfully implemented StateManager class",
      "All 28 tests passing"
    ],
    "challenges": [
      "Fixed artifact update bug in add_artifact method"
    ],
    "decisions": [
      {
        "decision": "Use JSON for state storage instead of SQLite",
        "rationale": "Simpler, more portable, sufficient for current needs"
      }
    ]
  },
  "summary": "Implemented core state tracking module with full test coverage",
  "statistics": {
    "duration_minutes": 120,
    "artifacts_created_count": 2,
    "artifacts_modified_count": 1,
    "files_changed": 5
  },
  "related_sessions": [
    "session-122"
  ]
}
```

## Configuration Schema (config.yaml)

Framework configuration file (versioned in git).

```yaml
framework:
  name: "project-name"
  version: "1.0.0"
  state_schema_version: "1.0.0"

paths:
  state_file: ".agentqms/state.json"
  sessions_dir: ".agentqms/sessions"
  artifacts_dir: "artifacts"

settings:
  max_sessions: 100
  max_session_age_days: 90
  auto_backup: true
  backup_dir: ".agentqms/backups"
  max_backups: 10

tracking:
  enable_session_tracking: true
  enable_artifact_tracking: true
  enable_context_preservation: true
  auto_index_artifacts: true
```

## Schema Validation

JSON schemas are provided in `.agentqms/schemas/` for validation:

- `state_schema.json`: Validates state.json structure
- `session_schema.json`: Validates session snapshot structure

## State Management Best Practices

1. **Atomic Updates**: State updates are atomic using temporary files and rename
2. **Backup Strategy**: Automatic backups before each save (if enabled)
3. **Corruption Recovery**: Automatic detection and recovery from corrupted state files
4. **Schema Versioning**: Schema version tracked for future migrations
5. **Immutable History**: Session snapshots are never modified after creation

## Usage Examples

### Initialize State Manager

```python
from agent_qms.toolbelt import StateManager

# Initialize with default config path
state_manager = StateManager()

# Or specify custom config path
state_manager = StateManager(config_path=".agentqms/config.yaml")
```

### Update Context

```python
# Set active session
state_manager.set_active_session("session-123")

# Update branch and phase
state_manager.update_current_context(
    current_branch="feature/new-feature",
    current_phase="phase-2"
)

# Add active artifact
state_manager.add_active_artifact("artifacts/plan.md")
```

### Track Artifacts

```python
# Add new artifact
state_manager.add_artifact(
    artifact_path="artifacts/plan.md",
    artifact_type="implementation_plan",
    status="draft",
    metadata={"author": "ai-agent", "priority": "high"}
)

# Update artifact status
state_manager.update_artifact_status("artifacts/plan.md", "validated")

# Get artifact info
artifact = state_manager.get_artifact("artifacts/plan.md")

# Query artifacts
plans = state_manager.get_artifacts_by_type("implementation_plan")
validated = state_manager.get_artifacts_by_status("validated")
```

### Health Checks

```python
# Validate state structure
is_valid = state_manager.validate_state()

# Get health metrics
health = state_manager.get_state_health()
print(f"State is valid: {health['is_valid']}")
print(f"Total artifacts: {health['total_artifacts']}")
print(f"State file size: {health['state_file_size_bytes']} bytes")

# Get statistics
stats = state_manager.get_statistics()
print(f"Total sessions: {stats['total_sessions']}")
print(f"Artifacts created: {stats['total_artifacts_created']}")
```

## Migration Strategy

When schema versions change, migration utilities will:

1. Detect schema version mismatch
2. Load old schema
3. Transform to new schema
4. Backup old state
5. Save new state with updated schema version

## Error Handling

The StateManager handles errors gracefully:

- **Missing config**: Raises `StateError` with clear message
- **Corrupted state**: Automatically backs up and reinitializes
- **Save failures**: Atomic operations prevent partial updates
- **Missing artifacts**: Returns `None` instead of raising errors

## Performance Considerations

- State operations complete in < 100ms
- Automatic backup cleanup (keeps last N backups)
- Session pruning by age (configurable)
- Efficient JSON serialization with atomic writes
