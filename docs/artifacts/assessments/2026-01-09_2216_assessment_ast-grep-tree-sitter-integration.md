---
ads_version: "1.0"
type: "assessment"
category: "evaluation"
status: "active"
version: "1.0"
tags: "ast-grep, tree-sitter, mcp, agent-toolkit, code-editing"
title: "AST-Grep and Tree-Sitter Integration Assessment for Agent Debug Toolkit"
date: "2026-01-09 22:16 (KST)"
branch: "main"
description: "In-depth assessment of integrating ast-grep and tree-sitter into the agent-debug-toolkit MCP server to improve AI agent code editing capabilities"
---

# AST-Grep and Tree-Sitter Integration Assessment

## Executive Summary

This assessment evaluates integrating **ast-grep** and **tree-sitter** into the existing agent-debug-toolkit to address three key problems:
1. **Large codebase navigation** - Slow text-based searches
2. **Frequent failed edits** - LLM edits often fail on large files  
3. **MCP tool overload** - Risk of context degradation with many tools

**Recommendation**: Adopt a **layered architecture** with tree-sitter as the parsing foundation, ast-grep for structural search/lint, and new `adt_edits` tools for reliable editing—all behind a **router/meta-tool pattern** to minimize tool proliferation.

---

## Problem Analysis

### Current State
The agent-debug-toolkit provides excellent **reading** capabilities:
- `analyze_config_access` - Config pattern detection
- `intelligent_search` - Fuzzy symbol lookup
- `context_tree` - Semantic directory navigation
- `explain_config_flow` - Hydra/OmegaConf flow analysis

**Gap**: No tools for **editing** code reliably, especially in large files where LLM whole-file rewrites frequently fail.

### Root Causes of Edit Failures
1. **Context window limits** - LLMs lose accuracy on files >500 lines
2. **Whitespace drift** - Minor formatting changes break exact matching
3. **No recovery** - Failed edits provide no partial application or diagnostics

---

## Technology Overview

### Tree-Sitter (Foundation Layer)

| Aspect | Details |
|--------|---------|
| **What** | Parser generator + incremental parsing library |
| **Output** | Concrete Syntax Tree (CST) for 100+ languages |
| **Strength** | Fast, incremental, error-tolerant parsing |
| **Limitation** | Low-level; requires custom logic for search/refactor |

**MCP Servers Available**:
- `wrale/mcp-server-tree-sitter` - Python, `pip install mcp-server-tree-sitter`
- `pwno-io/treesitter-mcp` - Call graphs, symbol extraction
- `nendotools/tree-sitter-mcp` - Standalone CLI + MCP

### ast-grep (Application Layer)

| Aspect | Details |
|--------|---------|
| **What** | Structural search/lint/rewrite engine built on tree-sitter |
| **Output** | Pattern matches, lint diagnostics, code patches |
| **Strength** | Rule-based YAML patterns, LLM-friendly workflow |
| **Limitation** | Less semantic depth than custom Python analyzers |

**MCP Server**: `ast-grep/ast-grep-mcp` (official)
- Tools: `sg_search`, `sg_lint`, `sg_rewrite`
- Pattern language for structural queries

### Relationship

```
┌─────────────────────────────────────────────────┐
│            Your Agent (Claude/Gemini)           │
├─────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ ADT (Hydra) │  │  ast-grep   │  │adt_edits │ │
│  │  Semantic   │  │ Structural  │  │  Fuzzy   │ │
│  │  Analysis   │  │   Search    │  │  Diff    │ │
│  └──────┬──────┘  └──────┬──────┘  └────┬─────┘ │
│         │                │              │       │
│  ┌──────┴────────────────┴──────────────┘       │
│  │           Tree-Sitter (Parsing)              │
│  └──────────────────────────────────────────────┘
└─────────────────────────────────────────────────┘
```

---

## Integration Options

### Option A: Parallel MCP Servers (Current Risk)

```yaml
# Multiple servers = tool explosion
mcpServers:
  unified_project: 20+ tools
  ast_grep_mcp: 5 tools  
  tree_sitter_mcp: 8 tools
```

**Problem**: 33+ tools floods the model's context, degrading reasoning accuracy (cliff ~40 tools).

### Option B: Router/Meta-Tool Pattern (Recommended)

Expose **2-5 meta-tools** that internally dispatch to many concrete operations:

```yaml
# Model sees only:
adt_meta_query:
  kinds: [config_flow, dependency_graph, symbol_search, sg_search, ast_query]
  
adt_meta_edit:
  kinds: [apply_diff, smart_edit, format_code, sg_rewrite]
```

**Benefits**:
- Small tool surface for model
- Add capabilities without changing tool schema
- Profiles can enable/disable kinds dynamically

### Option C: Tool Group Profiles

Leverage existing `mcp_tools_config.yaml` to create task-specific profiles:

```yaml
enabled_groups:
  - compass
  - adt_core
  - adt_edits  # New group for editing
  # - ast_grep  # Enable when needed
```

---

## Proposed Implementation

### Phase 1: Add `adt_edits` Tool Group (Priority)

New tools to solve the failed-edit problem:

| Tool | Purpose | Implementation |
|------|---------|----------------|
| `apply_unified_diff` | Apply git-style diffs with fuzzy matching | Aider-style hunk normalization |
| `smart_edit` | Search/replace with exact, regex, fuzzy modes | difflib + LibCST fallback |
| `read_file_slice` | Read specific line ranges | Encourage slice-based editing |
| `format_code` | Normalize code after patches | black/ruff wrapper |

**Key Pattern (from Aider)**:
```
read_file_slice → LLM proposes diff → apply_unified_diff
```

This avoids whole-file rewrites that cause failures.

### Phase 2: Integrate ast-grep MCP

Add as a separate server or inline into unified_server:

```python
# New tool group in unified_server.py
adt_astgrep:
  - sg_search   # Structural pattern search
  - sg_lint     # Run YAML lint rules
  - sg_rewrite  # Generate patches from rules
```

**Use Cases**:
- "Find all `requests.get` calls without `timeout=`"
- "Migrate API from `old_func($ARGS)` to `new_func($ARGS)`"
- Anti-pattern detection across repo

### Phase 3: Tree-Sitter Foundation (Optional)

Only if you need:
- Non-Python language support
- Custom AST queries not covered by ast-grep
- Integration with existing `intelligent_search` for richer symbol tables

**Implementation**: Install `mcp-server-tree-sitter` as subprocess or embed directly.

---

## Router Implementation Sketch

```python
# In unified_server.py

@app.call_tool()
async def call_tool(name: str, arguments: Any):
    if name == "adt_meta_query":
        kind = arguments["kind"]
        target = arguments["target"]
        
        if kind == "config_flow":
            # Existing ADT tools
            return await _run_config_flow(target)
        elif kind == "sg_search":
            # Dispatch to ast-grep
            pattern = arguments.get("pattern")
            return await _run_astgrep_search(pattern, target)
        elif kind == "ast_query":
            # Dispatch to tree-sitter
            query = arguments.get("query")
            return await _run_treesitter_query(query, target)
    
    if name == "adt_meta_edit":
        kind = arguments["kind"]
        
        if kind == "apply_diff":
            return await _apply_unified_diff(arguments)
        elif kind == "sg_rewrite":
            return await _run_astgrep_rewrite(arguments)
```

---

## Coexistence Strategy

```
┌────────────────────────────────────────────────────────┐
│                  unified_server.py                      │
├────────────────────────────────────────────────────────┤
│  Tool Groups (mcp_tools_config.yaml)                   │
│  ├── compass: Session, env_check                       │
│  ├── agentqms: Artifacts, standards                    │
│  ├── adt_core: Config analysis (existing)              │
│  ├── adt_edits: Fuzzy diff, smart_edit (NEW)          │
│  ├── adt_astgrep: sg_search, sg_lint (NEW)            │
│  └── adt_treesitter: get_ast, run_query (OPTIONAL)    │
├────────────────────────────────────────────────────────┤
│  Router Layer (adt_meta_query, adt_meta_edit)          │
│  - Reduces visible tools from 30+ to ~5                │
│  - Dispatches based on 'kind' parameter                │
└────────────────────────────────────────────────────────┘
```

---

## Decision Matrix

| Solution | Editing | Speed | Multi-lang | Complexity | Recommendation |
|----------|---------|-------|------------|------------|----------------|
| `adt_edits` only | ✅ | ✅ | ❌ | Low | **Start here** |
| + ast-grep | ✅ | ✅ | ✅ | Medium | Add for structural search |
| + tree-sitter | ✅ | ✅ | ✅ | High | Only if needed |

---

## Next Steps

1. **Immediate**: Create `agent_debug_toolkit/edits.py` with `apply_unified_diff`, `smart_edit`
2. **Short-term**: Add `adt_edits` tool group to `unified_server.py`
3. **Medium-term**: Integrate ast-grep MCP for structural operations
4. **Optional**: Add tree-sitter if multi-language or custom queries needed

---

## References

- [ast-grep/ast-grep-mcp](https://github.com/ast-grep/ast-grep-mcp) - Official MCP server
- [wrale/mcp-server-tree-sitter](https://github.com/wrale/mcp-server-tree-sitter) - Python tree-sitter MCP
- [Aider Edit Formats](https://aider.chat/docs/more/edit-formats.html) - Fuzzy diff patterns
- Curated research: `agent-debug-toolkit/draft-research-snippets-*.md`
