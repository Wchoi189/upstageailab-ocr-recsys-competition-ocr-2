# AgentQMS Core Tools

## Context Engine 2.0 (`context_bundle.py`)

The Context Engine provides high-performance, token-budget-aware context bundling for AI agents.

### Key Features
- **Smart Caching**: Caches `os.stat` calls, keyword configurations, and parsed bundle YAMLs to minimize I/O and latency.
- **Parallel I/O**: Uses `ThreadPoolExecutor` for parallel token estimation, significantly speeding up large bundle processing.
- **Token Budgeting**: Enforces a strict token limit (default: 32,000 tokens). Automatically drops lower-priority files when over budget.
- **Task Detection**: Automatically analyzes task descriptions to suggest the most relevant context bundle.
- **Efficient Globbing**: Uses `heapq` and `iglob` to handle large file trees without excessive memory usage.

### Usage

**CLI**:
```bash
# Auto-detect task type and suggest context
python AgentQMS/tools/core/context_bundle.py --task "debug ocr" --auto

# Load specific bundle type
python AgentQMS/tools/core/context_bundle.py --task "..." --type debugging

# List available bundles
python AgentQMS/tools/core/context_bundle.py --list

# Set custom token budget
python AgentQMS/tools/core/context_bundle.py --task "..." --budget 16000
```

**Python API**:
```python
from AgentQMS.tools.core.context_bundle import get_context_bundle

# Get context files (respects budget)
files = get_context_bundle("Fix bug in OCR pipeline")

# Print paths
for f in files:
    print(f['path'])
```

## Workflow Detector (`workflow_detector.py`)
Analyzes task descriptions to suggest appropriate workflows (e.g., `create-bug-report`, `create-plan`). Supports artifact type detection.

## Context Loader (`context_loader.py`)
Stateful loader for chat-based integrations, managing context windows and auto-loading bundles based on conversation turns.
