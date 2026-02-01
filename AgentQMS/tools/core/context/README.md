# Context Bundling System

This directory contains the core logic for the AgentQMS Context Bundling System. The system is designed to provide AI agents with high-performance, token-budgeted, and highly relevant context for their tasks.

## Script Overview

| Script | Role | Description |
|--------|------|-------------|
| `context_bundle.py` | **Core Engine** | High-performance bundling logic, parallel token estimation, and budgeting. |
| `suggest_context.py`| **Intelligence** | Advanced task detection using keywords, regex patterns, and AST analysis. |
| `context_loader.py` | **Stateful Loader**| Manages context for chat-based interactions, including persistence and auto-loading. |
| `context_control.py`| **Admin/Control** | Globally enable/disable the system or put it in maintenance/degraded mode. |
| `context_inspector.py`| **Observability**| Inspects memory footprint, stale files, and collects AI feedback on context quality. |
| `get_context.py`    | **CLI Utility**  | Developer-friendly CLI wrapper for fetching and listing bundles. |

## Key Concepts

### Context Bundles
Bundles are defined in `AgentQMS/.agentqms/plugins/context_bundles/*.yaml`. They specify sets of files (grouped by Tiers) that provide context for specific task areas (e.g., `ocr-debugging`, `pipeline-development`).

### Token Budgeting
The system enforces a strict token limit (default: 32,000). Files are prioritied by Tier (Tier 1 is highest). If the budget is exceeded, lower-priority files are automatically dropped.

### Intelligent Detection
The `ContextSuggester` (in `suggest_context.py`) analyzes the task description to pick the best bundle. It can detect debugging tasks, refactoring needs, and even suggest specific AST tools for code analysis.

## Usage

### CLI
```bash
# Auto-detect context for a task
uv run python AgentQMS/tools/core/context/context_bundle.py --task "fix hydra config" --auto

# List all available bundles
uv run python AgentQMS/tools/core/context/context_bundle.py --list
```

### Python API
```python
from AgentQMS.tools.core.context.context_bundle import get_context_bundle

# Get files for a task (returns list of dicts with 'path', 'tokens', etc.)
files = get_context_bundle("Fix bug in OCR pipeline")
```

### MCP Integration
The system is exposed via the `unified_project` MCP server:
- `bundle://<bundle-name>`: Access bundle files.
- `bundle://list`: List available bundles.
- `get_context_bundle` tool: High-level task-to-context resolution.
