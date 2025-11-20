# Quick Reference Guide

Fast lookup for common tasks, commands, and patterns.

## Artifact Creation

### Using AgentQMS Toolbelt (Preferred)

```python
from agent_qms.toolbelt import AgentQMSToolbelt

toolbelt = AgentQMSToolbelt()
artifact_path = toolbelt.create_artifact(
    artifact_type="assessment",  # or "implementation_plan", "bug_report"
    title="My Artifact",
    content="## Summary\n\nContent here...",
    tags=["tag1", "tag2"]
)
```

### Bug Reports (Special Case)

```python
import subprocess

# 1. Generate bug ID first
bug_id = subprocess.check_output(
    ["uv", "run", "python", "scripts/bug_tools/next_bug_id.py"],
    text=True
).strip()

# 2. Create bug report with ID
artifact_path = toolbelt.create_artifact(
    artifact_type="bug_report",
    title="Bug Description",
    content="## Summary\n...",
    bug_id=bug_id,
    severity="High"
)
```

## Validation Commands

```bash
# Validate artifact frontmatter
python scripts/agent_tools/documentation/validate_manifest.py

# Format code
uv run ruff format .

# Check code
uv run ruff check . --fix

# Type check (if mypy configured)
uv run mypy .
```

## Coding Standards Quick Check

**Type Hints:** Required for all functions
```python
def process_data(items: list[str]) -> dict[str, int]:
    ...
```

**Naming:**
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

**Formatting:**
- Line length: 100 characters
- Use ruff for formatting
- Imports: stdlib, third-party, local

## State Tracking Quick Start

```python
from agent_qms.toolbelt import state_api

# Get recent artifacts
artifacts = state_api.get_recent_artifacts(limit=10)

# Get current context
context = state_api.get_current_context()

# Get statistics
stats = state_api.get_statistics()
```

## Common File Locations

### Critical Documentation
- **Data Contracts**: `docs/pipeline/data_contracts.md`
- **API Contracts**: `docs/api/pipeline-contract.md`
- **Coding Standards**: `docs/agents/protocols/development.md`
- **State Tracking**: `.agentqms/USAGE_GUIDE.md`

### Agent Documentation
- **System Instructions**: `docs/agents/system.md`
- **Protocols**: `docs/agents/protocols/`
- **References**: `docs/agents/references/`

### Artifacts
- **Implementation Plans**: `artifacts/implementation_plans/`
- **Assessments**: `artifacts/assessments/`
- **Bug Reports**: `docs/bug_reports/`

## Filename Conventions

**Timestamped** (assessments, implementation plans):
```
YYYY-MM-DD_HHMM_descriptive-name.md
2025-11-20_1400_feature-assessment.md
```

**Semantic** (other artifacts):
```
descriptive-name.md
api-design-document.md
```

**Bug Reports**:
```
BUG-YYYYMMDD-###_descriptive-name.md
BUG-20251120-001_login-error.md
```

## Frontmatter Template

```yaml
---
title: Document Title
author: ai-agent
timestamp: 2025-11-20 14:00 KST
branch: feature/branch-name
status: draft
tags:
- tag1
- tag2
type: assessment  # or implementation_plan, bug_report
category: evaluation  # or development, troubleshooting
---
```

## Git Workflow

```bash
# Check status
git status

# Stage changes
git add <files>

# Commit with message
git commit -m "feat: description"

# Push to branch
git push -u origin <branch-name>
```

## Tool Discovery

```bash
# List all available tools
python scripts/agent_tools/core/discover.py --list

# Get artifact guide
python scripts/agent_tools/core/artifact_guide.py

# Unified CLI
python -m scripts.agent_tools list
```

## Pre-Commit Checklist

- [ ] Artifact frontmatter validated
- [ ] Code formatted (`ruff format`)
- [ ] Code checked (`ruff check --fix`)
- [ ] Data contracts reviewed (if pipeline changes)
- [ ] API contracts reviewed (if API changes)
- [ ] Tests pass (if applicable)

## Status Update Format

```markdown
## Status Update: [Title]

### ✅ Completed
- Task 1
- Task 2

### 🔄 In Progress
- Current task

### ⏭️ Next Steps
1. Next task

### 📊 Progress
- X/Y tasks complete
```

## Common Patterns

### Check Before Write
```python
from agent_qms.toolbelt import check_before_write

# Prevents manual artifact creation
check_before_write("artifacts/my-file.md")  # Raises error
```

### Session Management
```python
from agent_qms.toolbelt import SessionManager, StateManager

session_mgr = SessionManager(StateManager())

# Start session
session_id = session_mgr.start_session(
    branch="feature/x",
    goals=["Goal 1", "Goal 2"]
)

# Track progress
session_mgr.add_outcome("Completed task X")

# End session
session_mgr.end_session(summary="Work complete")
```

### Artifact Dependencies
```python
state_mgr.add_artifact_dependency(
    artifact_path="artifacts/plan.md",
    dependency_path="artifacts/assessment.md"
)
```

## Error Resolution

**"Artifact validation failed"**: Check frontmatter against schema
**"Manual creation error"**: Use AgentQMS toolbelt, not `write`
**"Bug ID generation failed"**: Check `uv` is installed and working
**"State tracking disabled"**: Check `.agentqms/config.yaml` exists

## Quick Links

- [Complete System Instructions](system.md)
- [Development Protocol](protocols/development.md)
- [Governance Protocol](protocols/governance.md)
- [Data Contracts](../pipeline/data_contracts.md)
- [API Contracts](../api/pipeline-contract.md)
- [State Tracking Guide](../../.agentqms/USAGE_GUIDE.md)
