# Session Handover: ADT Router & Edit Tools

## Current Session (2026-01-10)

Continuing agent-debug-toolkit enhancement: completed **Phase 3 (ast-grep integration)**, now starting **Phase 4 (Tree-Sitter Foundation)**.

## Completed Work

### Phase 1: Edit Tools ✅
- `edits.py` - 600+ lines with `apply_unified_diff`, `smart_edit`, `read_file_slice`, `format_code`
- `test_edits.py` - 26 tests, all passing
- Integrated into both `mcp_server.py` and `unified_server.py`

### Phase 2: Router Pattern ✅
- `router.py` - Meta-tool routing logic
- `adt_meta_query` - Routes to 10+ analysis tools
- `adt_meta_edit` - Routes to 4 edit tools
- Updated `AI_USAGE.yaml` with documentation

### Phase 3: ast-grep Integration ✅ (Just Completed)
- `astgrep.py` - 340+ lines with `sg_search`, `sg_lint`, `dump_syntax_tree`
- `test_astgrep.py` - 18 tests, all passing
- New kinds in router: `sg_search`, `sg_lint`, `ast_dump`
- `mcp_tools_config.yaml` updated with `adt_astgrep` group
- 167 total tests passing (no regressions)

## Next: Phase 4 - Tree-Sitter Foundation

### Phase 4 Scope (from roadmap)
- Add tree-sitter for multi-language support
- Dependencies: Phase 3 (ast-grep) complete ✅
- Priority: Low

### Key References
- Roadmap: `project_compass/roadmap/00_adt_router_and_edits.yaml`
- Assessment: `docs/artifacts/assessments/2026-01-09_2216_assessment_ast-grep-tree-sitter-integration.md`
- AI_USAGE.yaml: `agent-debug-toolkit/AI_USAGE.yaml` (v1.2)

## Files Modified This Session (Phase 3)
- `agent-debug-toolkit/src/agent_debug_toolkit/astgrep.py` (NEW)
- `agent-debug-toolkit/tests/test_astgrep.py` (NEW)
- `agent-debug-toolkit/src/agent_debug_toolkit/router.py`
- `scripts/mcp/unified_server.py`
- `scripts/mcp/mcp_tools_config.yaml`
- `agent-debug-toolkit/AI_USAGE.yaml`
