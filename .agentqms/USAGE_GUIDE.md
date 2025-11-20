# Agent Framework State Tracking - Usage Guide

## Overview

The Agent Framework State Tracking system provides persistent state management across conversations, enabling agents to maintain context, track artifacts, and preserve session history.

## Quick Start

### Using the Simple API

The easiest way to use state tracking is through the simple API:

```python
from agent_qms.toolbelt import state_api

# Get current context
context = state_api.get_current_context()
print(f"Active session: {context['active_session_id']}")
print(f"Current branch: {context['current_branch']}")

# Get recent artifacts
artifacts = state_api.get_recent_artifacts(limit=5)
for artifact in artifacts:
    print(f"{artifact['path']}: {artifact['status']}")

# Get statistics
stats = state_api.get_statistics()
print(f"Total artifacts: {stats['total_artifacts_created']}")
print(f"Total sessions: {stats['total_sessions']}")
```

### Using the Toolbelt (Automatic Tracking)

When you create artifacts using AgentQMSToolbelt, state tracking happens automatically:

```python
from agent_qms.toolbelt import AgentQMSToolbelt

# Initialize toolbelt (state tracking enabled by default)
toolbelt = AgentQMSToolbelt()

# Create an artifact - automatically tracked in state
artifact_path = toolbelt.create_artifact(
    artifact_type="implementation_plan",
    title="New Feature Implementation",
    content="# Implementation Plan\n\n...",
    tags=["feature", "backend"]
)

# Artifact is now automatically:
# - Added to state index
# - Tracked in current session (if active)
# - Available for querying
```

## Core Concepts

### 1. State Manager

The StateManager handles state persistence, artifact tracking, and context management.

```python
from agent_qms.toolbelt import StateManager

# Initialize
state_mgr = StateManager()

# Add artifact manually
state_mgr.add_artifact(
    artifact_path="artifacts/plan.md",
    artifact_type="implementation_plan",
    status="draft",
    metadata={"priority": "high"}
)

# Update artifact status
state_mgr.update_artifact_status("artifacts/plan.md", "validated")

# Query artifacts
plans = state_mgr.get_artifacts_by_type("implementation_plan")
drafts = state_mgr.get_artifacts_by_status("draft")

# Get specific artifact
artifact = state_mgr.get_artifact("artifacts/plan.md")
```

### 2. Session Manager

The SessionManager handles session lifecycle, snapshots, and context preservation.

```python
from agent_qms.toolbelt import SessionManager, StateManager

state_mgr = StateManager()
session_mgr = SessionManager(state_mgr)

# Start a new session
session_id = session_mgr.start_session(
    branch="feature/new-feature",
    phase="implementation",
    goals=["Implement feature X", "Write tests"]
)

# Track context during session
session_mgr.add_outcome("Feature X implemented successfully")
session_mgr.add_challenge("Integration tests failed initially")
session_mgr.add_decision(
    decision="Use Redis for caching",
    rationale="Better performance for our use case"
)

# End session
session_mgr.end_session(
    summary="Successfully implemented feature X",
    outcomes=["Feature deployed", "Tests passing"],
    challenges=["Performance optimization needed"]
)

# Later: Restore context from previous session
context = session_mgr.restore_session_context(session_id)
print(f"Previous goals: {context['goals']}")
print(f"Previous outcomes: {context['outcomes']}")
```

### 3. Artifact Relationships

Track dependencies between artifacts:

```python
# Add dependency (plan2 depends on plan1)
state_mgr.add_artifact_dependency(
    artifact_path="artifacts/plan2.md",
    dependency_path="artifacts/plan1.md"
)

# Get dependencies
deps = state_mgr.get_artifact_dependencies("artifacts/plan2.md")
print(f"Dependencies: {deps}")

# Get reverse lookup (what depends on plan1?)
depending = state_mgr.get_artifacts_depending_on("artifacts/plan1.md")
print(f"Artifacts depending on plan1: {depending}")

# Get full dependency tree
tree = state_mgr.get_dependency_tree("artifacts/plan2.md")

# Propagate status changes
updated = state_mgr.propagate_status_update("artifacts/plan1.md", "deprecated")
print(f"Updated artifacts: {updated}")  # plan1 and all dependents
```

## Common Use Cases

### Use Case 1: Resuming Work After Conversation Reset

```python
from agent_qms.toolbelt import state_api

# Check if there's an active session
active_session = state_api.get_active_session()

if active_session:
    print(f"Resuming session: {active_session['session_id']}")
    print(f"Goals: {active_session['context']['goals']}")
    print(f"Progress: {active_session['context']['outcomes']}")

# Get recent artifacts created
recent = state_api.get_recent_artifacts(limit=5)
print(f"Recent artifacts: {[a['path'] for a in recent]}")

# Get current phase
phase = state_api.get_current_phase()
print(f"Current phase: {phase}")
```

### Use Case 2: Tracking Implementation Progress

```python
from agent_qms.toolbelt import SessionManager, StateManager, state_api

# Start session for implementation
session_mgr = SessionManager(StateManager())
session_id = session_mgr.start_session(
    branch="feature/api-v2",
    phase="implementation",
    goals=[
        "Implement API endpoints",
        "Write unit tests",
        "Update documentation"
    ]
)

# As you work, track progress
session_mgr.add_outcome("API endpoints implemented")
session_mgr.track_artifact_creation("artifacts/api-design.md")

# Later: Check progress
session = state_api.get_active_session()
completed_goals = len(session['context']['outcomes'])
total_goals = len(session['context']['goals'])
print(f"Progress: {completed_goals}/{total_goals} goals completed")
```

### Use Case 3: Querying Related Artifacts

```python
# Find all implementation plans in draft status
drafts = state_api.get_artifacts_by_status("draft")
plans = [a for a in drafts if a['type'] == "implementation_plan"]

print(f"Draft plans: {len(plans)}")
for plan in plans:
    print(f"  - {plan['path']} (created: {plan['created_at']})")

# Find artifacts by tag (from metadata)
all_artifacts = state_api.get_recent_artifacts(limit=100)
feature_artifacts = [
    a for a in all_artifacts
    if "feature" in a.get('metadata', {}).get('tags', [])
]

print(f"Feature-related artifacts: {len(feature_artifacts)}")
```

### Use Case 4: Analyzing Session History

```python
# Get recent sessions
recent_sessions = state_api.get_recent_sessions(limit=10)

for session in recent_sessions:
    duration = session['statistics'].get('duration_minutes', 0)
    artifacts = len(session['artifacts_created'])
    print(f"Session {session['session_id'][:20]}...")
    print(f"  Duration: {duration:.1f} minutes")
    print(f"  Artifacts: {artifacts}")
    print(f"  Branch: {session['branch']}")
    print()

# Search for sessions by criteria
feature_sessions = state_api.search_sessions(
    branch="feature/api-v2",
    phase="implementation"
)

print(f"Found {len(feature_sessions)} sessions for API v2 implementation")
```

## Advanced Features

### Circular Dependency Detection

The system automatically prevents circular dependencies:

```python
# This is OK: plan3 -> plan2 -> plan1
state_mgr.add_artifact_dependency("plan2.md", "plan1.md")
state_mgr.add_artifact_dependency("plan3.md", "plan2.md")

# This will raise StateError (would create cycle)
try:
    state_mgr.add_artifact_dependency("plan1.md", "plan3.md")
except StateError as e:
    print(f"Prevented circular dependency: {e}")
```

### Status Propagation

When you deprecate an artifact, all dependent artifacts are automatically deprecated:

```python
# plan2 and plan3 depend on plan1
state_mgr.add_artifact_dependency("plan2.md", "plan1.md")
state_mgr.add_artifact_dependency("plan3.md", "plan1.md")

# Deprecate plan1 - automatically deprecates plan2 and plan3
updated = state_mgr.propagate_status_update("plan1.md", "deprecated")
print(f"Deprecated: {updated}")  # ['plan1.md', 'plan2.md', 'plan3.md']
```

### State Health Monitoring

Check the health of your state:

```python
health = state_api.get_state_health()

print(f"State valid: {health['is_valid']}")
print(f"Total artifacts: {health['total_artifacts']}")
print(f"Total sessions: {health['total_sessions']}")
print(f"State file size: {health['state_file_size_bytes']} bytes")
print(f"Active session: {health['active_session']}")
```

### Session Cleanup

Clean up old sessions to save space:

```python
from agent_qms.toolbelt import SessionManager, StateManager

session_mgr = SessionManager(StateManager())

# Delete sessions older than 90 days
deleted = session_mgr.cleanup_old_sessions(max_age_days=90)
print(f"Deleted {deleted} old sessions")
```

## Best Practices

### 1. Always Start/End Sessions for Major Work

```python
# Start session when beginning new work
session_id = session_mgr.start_session(
    branch=current_branch,
    phase=current_phase,
    goals=["Goal 1", "Goal 2"]
)

try:
    # Do your work
    # Track progress with add_outcome(), add_challenge()
    pass
finally:
    # Always end session when done
    session_mgr.end_session(summary="Work completed")
```

### 2. Track Artifact Dependencies

```python
# When creating a plan based on an assessment
assessment_path = "artifacts/assessments/feature-assessment.md"
plan_path = "artifacts/implementation_plans/feature-plan.md"

# Create the plan
toolbelt.create_artifact(...)

# Track dependency
state_mgr.add_artifact_dependency(plan_path, assessment_path)
```

### 3. Use the Simple API for Queries

```python
# Use state_api for simple queries
from agent_qms.toolbelt import state_api

# Don't create StateManager instances for simple queries
artifacts = state_api.get_recent_artifacts(limit=10)

# Use StateManager directly only for complex operations
from agent_qms.toolbelt import StateManager
state_mgr = StateManager()
# Complex operations...
```

### 4. Handle State Tracking Failures Gracefully

```python
# State tracking failures should not block main work
try:
    state_mgr.add_artifact(...)
except Exception as e:
    print(f"Warning: State tracking failed: {e}")
    # Continue with main work
```

## Configuration

Edit `.agentqms/config.yaml` to customize behavior:

```yaml
settings:
  max_sessions: 100  # Maximum sessions to keep
  max_session_age_days: 90  # Auto-cleanup after 90 days
  auto_backup: true  # Backup state before saves
  backup_dir: ".agentqms/backups"
  max_backups: 10  # Keep last 10 backups

tracking:
  enable_session_tracking: true
  enable_artifact_tracking: true
  enable_context_preservation: true
  auto_index_artifacts: true
```

## Troubleshooting

### State File Corrupted

The system automatically recovers from corrupted state files:

1. Corrupted file is backed up with timestamp
2. Fresh state is initialized with default schema
3. Warning message is printed

No manual intervention needed!

### State Tracking Disabled

If you see "State tracking disabled":

1. Check that `.agentqms/config.yaml` exists
2. Check that `.agentqms/state.json` exists (will be created if missing)
3. Check file permissions

### Performance Issues

If state operations are slow:

1. Check state file size: `ls -lh .agentqms/state.json`
2. Clean up old sessions: `session_mgr.cleanup_old_sessions(90)`
3. Reduce `max_sessions` in config

## API Reference

See `.agentqms/STATE_SCHEMA.md` for detailed API documentation and schema definitions.

## Examples

See `.agentqms/examples/` directory for complete working examples.
