# Context Bundles

This directory contains YAML definitions for task-specific context bundles.
The context bundle system helps AI agents load relevant files based on their current task.

## Available Bundles

| Bundle | Description | Make Target |
|--------|-------------|-------------|
| `development` | Code implementation, features | `make context-development` |
| `documentation` | Writing/updating docs | `make context-docs` |
| `debugging` | Troubleshooting issues | `make context-debug` |
| `planning` | Design and implementation planning | `make context-plan` |
| `general` | Default fallback bundle | N/A (auto-selected) |

## Usage

### Via Make Targets

```bash
cd AgentQMS/interface/
make context-development     # Get development context
make context-docs           # Get documentation context
make context-debug          # Get debugging context
make context-plan           # Get planning context
make context TASK="desc"    # Auto-detect from task description
```

### Via Python

```python
from AgentQMS.agent_tools.core.context_bundle import get_context_bundle

# Auto-detect task type from description
files = get_context_bundle("implement new validation feature")

# Explicit task type
files = get_context_bundle("any task", task_type="development")
```

### Via CLI

```bash
PYTHONPATH=. python AgentQMS/agent_tools/utilities/get_context.py --task "implement feature"
PYTHONPATH=. python AgentQMS/agent_tools/utilities/get_context.py --type development
PYTHONPATH=. python AgentQMS/agent_tools/utilities/get_context.py --list-context-bundles
```

## Bundle Structure

Each bundle YAML file contains:

```yaml
name: bundle_name
title: Human-readable title
description: |
  Multi-line description of when to use this bundle.

tiers:
  tier1:
    name: Essential
    description: Must-read files
    max_files: 5
    files:
      - path: path/to/file.md
        priority: critical|high|medium|low
        description: Why this file is included
```

## Adding New Bundles

1. Create a new YAML file in this directory (e.g., `my_bundle.yaml`)
2. Follow the structure above
3. Add keywords to `TASK_KEYWORDS` in `context_bundle.py` if needed for auto-detection
4. Update this README with the new bundle

