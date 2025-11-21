# State Tracking Tools

Tools for tracking and managing project state using the AgentQMS framework.

## Overview

The state tracking system provides:
- **State Persistence**: Track project context, artifacts, sessions, and progress
- **CLI Tools**: Query and display current project state
- **README Sync**: Automatically synchronize README.md with actual project state

## Tools

### 1. `initialize_state.py` - Initialize/Update State

Initializes or updates the AgentQMS state with current project data.

**Usage:**
```bash
# Initialize state (safe - won't overwrite if already exists)
uv run python scripts/agent_tools/utilities/initialize_state.py

# Force re-initialization
uv run python scripts/agent_tools/utilities/initialize_state.py --force

# Set specific phase
uv run python scripts/agent_tools/utilities/initialize_state.py --phase phase-5
```

**What it does:**
- Scans `artifacts/` directory for all markdown files
- Extracts artifact metadata (type, status, timestamps)
- Gets current git branch
- Auto-detects current project phase from README
- Updates state file (`.agentqms/state.json`)

### 2. `show_state.py` - Display Current State

Displays project state in a readable format.

**Usage:**
```bash
# Show all information
uv run python scripts/agent_tools/utilities/show_state.py

# Show only current context
uv run python scripts/agent_tools/utilities/show_state.py --context

# Show only artifacts (top 10)
uv run python scripts/agent_tools/utilities/show_state.py --artifacts

# Show only artifacts (top 5)
uv run python scripts/agent_tools/utilities/show_state.py --artifacts 5

# Show only project health
uv run python scripts/agent_tools/utilities/show_state.py --health

# Show only phase progress
uv run python scripts/agent_tools/utilities/show_state.py --phases

# Show only statistics
uv run python scripts/agent_tools/utilities/show_state.py --stats
```

**What it shows:**
- Current context (branch, phase, active artifacts, pending tasks)
- Artifact statistics (by type, by status)
- Recent artifacts (most recently updated)
- Phase progress (parsed from README.md)
- Project health metrics

### 3. `sync_readme_state.py` - Sync README with State

Synchronizes README.md project status section with current state.

**Usage:**
```bash
# Sync README (dry run to preview changes)
uv run python scripts/agent_tools/utilities/sync_readme_state.py --dry-run

# Actually sync README
uv run python scripts/agent_tools/utilities/sync_readme_state.py

# Use different README path
uv run python scripts/agent_tools/utilities/sync_readme_state.py --readme path/to/README.md
```

**What it does:**
- Reads current state from StateManager
- Parses README.md project status table
- Updates overall progress based on current phase
- (Future) Could update individual phase progress from implementation plans

### 4. `state_workflow.py` - Complete Workflow

Runs the complete state tracking workflow in one command.

**Usage:**
```bash
# Full workflow (init + show + optional sync)
uv run python scripts/agent_tools/utilities/state_workflow.py

# Include README sync
uv run python scripts/agent_tools/utilities/state_workflow.py --sync-readme

# Force re-initialization
uv run python scripts/agent_tools/utilities/state_workflow.py --force

# Skip initialization (just show state)
uv run python scripts/agent_tools/utilities/state_workflow.py --skip-init
```

## Quick Start

1. **Initialize state tracking** (first time):
   ```bash
   uv run python scripts/agent_tools/utilities/initialize_state.py --force
   ```

2. **Check current state**:
   ```bash
   uv run python scripts/agent_tools/utilities/show_state.py
   ```

3. **Sync README** (optional):
   ```bash
   uv run python scripts/agent_tools/utilities/sync_readme_state.py --dry-run
   uv run python scripts/agent_tools/utilities/sync_readme_state.py
   ```

Or use the complete workflow:
```bash
uv run python scripts/agent_tools/utilities/state_workflow.py --sync-readme
```

## State File Location

The state is stored in:
- `.agentqms/state.json` - Main state file (gitignored)
- `.agentqms/config.yaml` - Configuration (versioned in git)
- `.agentqms/sessions/` - Session snapshots (gitignored)
- `.agentqms/backups/` - State backups (gitignored)

## State Schema

The state tracks:
- **Current Context**: Active session, git branch, current phase, active artifacts
- **Artifacts**: Index of all artifacts with type, status, metadata
- **Sessions**: Session history and tracking
- **Statistics**: Aggregate metrics (total sessions, artifacts created, etc.)

See `.agentqms/STATE_SCHEMA.md` for detailed schema documentation.

## Integration with Agent Workflows

These tools integrate with the AgentQMS framework:

1. **Automatic State Updates**: When creating artifacts via `AgentQMSToolbelt`, state is automatically updated
2. **Session Tracking**: State tracks active sessions and context preservation
3. **Artifact Indexing**: All artifacts are automatically indexed in state

## Future Enhancements

Potential improvements:
- Parse implementation plan progress trackers to calculate phase progress more accurately
- Automatically update individual phase progress in README
- Generate project status reports
- Integration with CI/CD for automated status updates
- Export state to various formats (JSON, Markdown, HTML)

## Examples

### Check if state needs updating
```bash
# Show current state
uv run python scripts/agent_tools/utilities/show_state.py

# If artifacts are missing, re-initialize
uv run python scripts/agent_tools/utilities/initialize_state.py --force
```

### Regular status check
```bash
# Quick status check
uv run python scripts/agent_tools/utilities/show_state.py --context --stats
```

### Update README after major milestone
```bash
# Update state with latest artifacts
uv run python scripts/agent_tools/utilities/initialize_state.py

# Preview README changes
uv run python scripts/agent_tools/utilities/sync_readme_state.py --dry-run

# Apply changes
uv run python scripts/agent_tools/utilities/sync_readme_state.py
```

