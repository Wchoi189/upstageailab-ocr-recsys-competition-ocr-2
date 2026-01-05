# Shared MCP Architecture

This project now uses a unified Model Context Protocol (MCP) server that combines all project MCP functionality into a single process to reduce resource overhead and improve efficiency.

## Architecture Overview

The unified server combines functionality from:
- Project Compass (project navigation and context management)
- AgentQMS (artifact workflows and standards)
- Experiment Manager (experiment lifecycle tools)

## Resources

### Project Compass Resources
- `compass://compass.json` - Main project state
- `compass://session_handover.md` - Current session handover
- `compass://current_session.yml` - Active session context
- `compass://uv_lock_state.yml` - Environment lock state
- `compass://agents.yaml` - Agent configuration

### AgentQMS Resources
- `agentqms://standards/index` - Standards hierarchy
- `agentqms://standards/artifact_types` - Artifact types and locations
- `agentqms://standards/workflows` - Workflow requirements
- `agentqms://templates/list` - Available templates
- `agentqms://config/settings` - QMS settings

### Experiment Manager Resources
- `experiments://agent_interface` - Command reference
- `experiments://active_list` - List of active experiments
- `experiments://schemas/manifest` - Manifest JSON schema
- `experiments://schemas/artifact` - Artifact JSON schema

## Tools

### Project Compass Tools
- `env_check` - Validate environment
- `session_init` - Initialize session with objective
- `reconcile` - Synchronize experiment metadata
- `ocr_convert` - Convert datasets to LMDB
- `ocr_inspect` - Verify LMDB integrity

### AgentQMS Tools
- `create_artifact` - Create new artifact following standards
- `validate_artifact` - Validate artifact against standards
- `list_artifact_templates` - List available templates
- `check_compliance` - Check overall compliance status

### Experiment Manager Tools
- `init_experiment` - Initialize new experiment
- `get_experiment_status` - Get experiment status
- `add_task` - Add task to experiment
- `log_insight` - Log insight/decision/failure
- `sync_experiment` - Sync to database

## Configuration

The server is configured in `mcp/shared_config.json` and can be run with:

```bash
uv run python mcp/unified_server.py
```

## Benefits

- Single process instead of multiple MCP servers
- Reduced memory and CPU overhead
- Simplified maintenance
- Centralized logging and monitoring