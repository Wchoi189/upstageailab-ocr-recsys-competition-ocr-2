---
ads_version: "1.0"
type: "walkthrough"
category: "documentation"
status: "active"
version: "1.0"
tags: "adt, mcp, router, edits, implementation"
title: "ADT Router Pattern and Edit Tools Implementation Walkthrough"
date: "2026-01-09 23:42 (KST)"
branch: "main"
description: "Documents the implementation of edit tools and router pattern for agent-debug-toolkit"
---

# ADT Router Pattern and Edit Tools Implementation

## Summary

Implemented **Option B (Router/Meta-Tool Pattern)** and **adt_edits module** for the agent-debug-toolkit, reducing tool count while adding reliable editing capabilities.

## Deliverables

### Phase 1: Edit Tools Module ✅

| File | Description |
|------|-------------|
| [edits.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/edits.py) | 600+ lines with `apply_unified_diff`, `smart_edit`, `read_file_slice`, `format_code` |
| [test_edits.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/tests/test_edits.py) | 26 tests covering all functions |

**Key Features**:
- Fuzzy diff application (Aider-style with fallback matching)
- Multi-mode search/replace (exact, regex, fuzzy)
- Line-range reading for large file editing
- Code formatting wrapper (black/ruff/isort)

### Phase 2: Router Pattern ✅

| File | Description |
|------|-------------|
| [router.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/router.py) | Meta-tool routing logic |
| [mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/mcp_server.py) | +2 meta-tools, handlers |
| [unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py) | Synced meta-tools |

**Meta-Tools**:
- `adt_meta_query`: Routes to 10 analysis tools via `kind` parameter
- `adt_meta_edit`: Routes to 4 edit tools via `kind` parameter

## Verification Results

```
✅ All 149 tests passed
✅ unified_server.py: Syntax OK
✅ mcp_server.py: Syntax OK  
✅ router.py: Syntax OK
✅ edits.py: Syntax OK
```

## Tool Mapping

### adt_meta_query Kinds
| Kind | Routes To |
|------|-----------|
| config_access | analyze_config_access |
| merge_order | trace_merge_order |
| hydra_usage | find_hydra_usage |
| component_instantiations | find_component_instantiations |
| config_flow | explain_config_flow |
| dependency_graph | analyze_dependency_graph |
| imports | analyze_imports |
| complexity | analyze_complexity |
| context_tree | context_tree |
| symbol_search | intelligent_search |

### adt_meta_edit Kinds
| Kind | Routes To |
|------|-----------|
| apply_diff | apply_unified_diff |
| smart_edit | smart_edit |
| read_slice | read_file_slice |
| format | format_code |

## Usage

```python
# Analysis via meta-tool
adt_meta_query(kind="config_flow", target="src/model.py")

# Editing via meta-tool
adt_meta_edit(kind="smart_edit", target="src/utils.py", options={
    "search": "old_function",
    "replace": "new_function"
})
```

## Documentation Updated

- [AI_USAGE.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/AI_USAGE.yaml) - Edit tools documentation and best practices added
