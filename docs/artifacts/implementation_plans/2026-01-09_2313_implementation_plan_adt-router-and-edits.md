---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: "adt, mcp, edits, router-pattern"
title: "ADT Router Pattern and Edit Tools Implementation"
date: "2026-01-09 23:13 (KST)"
branch: "main"
description: "Implementation plan for adding adt_edits module and router pattern to agent-debug-toolkit"
---

# ADT Router Pattern and Edit Tools Implementation Plan

## Goal

Implement Option B (Router/Meta-Tool Pattern) and `adt_edits` module for the agent-debug-toolkit to:
1. Reduce tool proliferation (from 15+ to 2-3 visible tools)
2. Add reliable editing capabilities for AI agents on large files
3. Keep both `mcp_server.py` and `unified_server.py` in sync

## User Review Required

> [!IMPORTANT]
> **Phase 1 creates files that enable code modification.** The `apply_unified_diff` and `smart_edit` tools will write to disk. Please confirm this is acceptable.

> [!WARNING]
> **Phase 2 changes the tool schema.** The router pattern replaces individual tools with meta-tools. Any existing prompts/workflows referencing old tool names will need updating.

---

## Proposed Changes

### Phase 1: `adt_edits` Module

#### [NEW] [edits.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/edits.py)

New editing module with four functions:

| Function | Purpose |
|----------|---------|
| `apply_unified_diff(diff, strategy, project_root)` | Apply git-style diffs with fuzzy matching |
| `smart_edit(file, search, replace, mode)` | Search/replace with exact, regex, or fuzzy modes |
| `read_file_slice(file, start_line, end_line)` | Read specific line range to minimize context |
| `format_code(path, style, scope)` | Post-edit formatting (black/ruff) |

Key implementation details from Aider patterns:
- Fuzzy matching normalizes whitespace for drift tolerance
- Return `EditReport` dataclass with applied status, message, and failed hunks
- Multi-stage fallback: exact → whitespace-insensitive → fuzzy

---

#### [NEW] [test_edits.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/tests/test_edits.py)

Test cases following existing test patterns in ADT:
- Test exact diff application
- Test fuzzy matching with whitespace drift
- Test smart_edit in all modes  
- Test file slice reading
- Test format_code wrapper

---

#### [MODIFY] [mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/mcp_server.py)

Add 4 new tool definitions to `TOOLS` list:
- `apply_unified_diff`
- `smart_edit`  
- `read_file_slice`
- `format_code`

Add corresponding handlers in `call_tool()`.

---

#### [MODIFY] [unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py)

Mirror the 4 new tool definitions and handlers to keep servers in sync.

---

### Phase 2: Router Pattern

#### [MODIFY] [mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/mcp_server.py)

Replace individual tools with 2 meta-tools:

```python
# Before: 14 individual tools
# After: 2 meta-tools + original tools available via 'kind' dispatch

adt_meta_query:
  kinds: [config_access, merge_order, hydra_usage, component_instantiations, 
          config_flow, dependency_graph, imports, complexity, 
          context_tree, intelligent_search]

adt_meta_edit:
  kinds: [apply_diff, smart_edit, read_slice, format_code]
```

---

#### [MODIFY] [AI_USAGE.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/AI_USAGE.yaml)

Update documentation to reflect router pattern:
- Document `adt_meta_query` with all supported kinds
- Document `adt_meta_edit` with all supported kinds
- Keep backward-compatible individual tool docs as "direct access" section

---

## Verification Plan

### Automated Tests

**Existing tests** (must continue to pass):
```bash
cd agent-debug-toolkit && uv run pytest tests/ -v
```

**New tests** for Phase 1:
```bash
cd agent-debug-toolkit && uv run pytest tests/test_edits.py -v
```

Test coverage goals:
- `apply_unified_diff`: 3 test cases (exact, fuzzy, failure handling)
- `smart_edit`: 4 test cases (exact, regex, fuzzy, not-found)
- `read_file_slice`: 2 test cases (valid range, edge cases)
- `format_code`: 1 test case (black formatting)

### Manual Verification

1. **MCP Server Test** - Start server and invoke tools:
   ```bash
   cd agent-debug-toolkit && uv run python -m agent_debug_toolkit.mcp_server
   ```
   Then use MCP client to call `smart_edit` on a test file.

2. **Integration Test** - Use the unified_server.py and verify new tools appear:
   ```bash
   cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
   uv run python scripts/mcp/unified_server.py
   ```
   Call `get_server_info` and verify `adt_edits` group is listed.

---

## Dependencies

- `difflib` (stdlib) - For fuzzy matching
- `black` (optional) - For format_code; graceful degradation if missing

## Rollback Plan

All changes are additive in Phase 1. Simply delete `edits.py` and remove tool entries to rollback.

Phase 2 changes existing tool names - rollback requires restoring original tool definitions.
