# Session 3 Preparation: Unified Server Integration

**Date Prepared**: 2026-01-10
**Status**: Ready for Session 3 start
**Estimated Work**: 2-3 hours

## Context

Phase 1 Sessions 1-2 completed MCP resource for artifact types in `AgentQMS/mcp_server.py`. Now need to integrate same resource into unified server at `scripts/mcp/unified_server.py`.

## Current State

### What's Done (Session 1-2)
- ✅ `AgentQMS/mcp_server.py`:
  - Resource registered: `agentqms://plugins/artifact_types`
  - Handler: `_get_plugin_artifact_types()` (120+ LOC)
  - Tests: 27/27 passing
  - Verified: All 11 types discovered correctly

### What Needs Doing (Session 3)
- [ ] Integrate resource into `scripts/mcp/unified_server.py`
- [ ] Add unified server tests for new resource
- [ ] Verify resource accessible via unified server
- [ ] Update unified server documentation

## Discovery: Unified Server Structure

**File**: `scripts/mcp/unified_server.py` (784 lines)

### Current Resources Pattern
```python
RESOURCES_CONFIG = [
    {
        "uri": "compass://compass.json",
        ...
    },
    {
        "uri": "agentqms://standards/artifact_types",  # RELATED: This exists!
        ...
    },
    ...
]
```

### Handler Pattern
```python
@app.read_resource()
async def read_resource(uri: str) -> list[ReadResourceContents]:
    """Handle resource read requests."""
    resource = next((r for r in RESOURCES_CONFIG if r["uri"] == uri), None)

    if not resource:
        raise ValueError(f"Unknown resource: {uri}")

    # Load resource content based on type
    if uri == "compass://compass.json":
        # Handle compass resource
    elif uri.startswith("agentqms://"):
        # Handle agentqms resources
    ...
```

## Session 3 Scope

### Task 1: Add Resource to RESOURCES_CONFIG
**File**: `scripts/mcp/unified_server.py` line ~142
**Work**: Add new entry after existing `agentqms://standards/artifact_types`:
```python
{
    "uri": "agentqms://plugins/artifact_types",
    "name": "Plugin Artifact Types",
    "description": "Discoverable artifact types with complete metadata",
    "mimeType": "application/json",
}
```

### Task 2: Add Handler Logic
**File**: `scripts/mcp/unified_server.py` in `read_resource()` handler
**Work**: Add conditional to route to AgentQMS handler:
```python
elif uri == "agentqms://plugins/artifact_types":
    # Import and call the AgentQMS handler
    from AgentQMS.mcp_server import _get_plugin_artifact_types
    content = await _get_plugin_artifact_types()
    return [ReadResourceContents(content=content, mime_type="application/json")]
```

### Task 3: Create Tests
**File**: Create `tests/test_unified_server_artifact_types.py`
**Tests needed**:
- Resource appears in unified server list_resources()
- read_resource() returns valid JSON
- All 11 types included
- Response schema matches design doc
- Works via unified server endpoint

### Task 4: Integration Verification
**File**: N/A (manual verification)
**Steps**:
- Start unified MCP server
- Query `agentqms://plugins/artifact_types`
- Verify response structure
- Confirm all types present

### Task 5: Documentation
**File**: `scripts/mcp/README.md` (if exists)
**Work**: Add section documenting new resource

## Key Files Reference

| File | Role | Status |
|------|------|--------|
| `AgentQMS/mcp_server.py` | Source implementation | ✅ Complete |
| `scripts/mcp/unified_server.py` | Integration target | 🟡 Needs update |
| `tests/test_mcp_plugin_resource.py` | Source tests | ✅ Complete (27/27) |
| `tests/test_unified_server_artifact_types.py` | New tests | ❌ To create |
| Design doc | Reference spec | ✅ Complete |

## Code Patterns to Reuse

### From AgentQMS Handler (Source)
- Function: `_get_plugin_artifact_types()` in AgentQMS/mcp_server.py
- Import needed: `from AgentQMS.mcp_server import _get_plugin_artifact_types`
- Response format: JSON string matching design schema

### From Unified Server (Target)
- List resources: `@app.list_resources()` pattern
- Handle resource: `@app.read_resource()` pattern
- Response type: `ReadResourceContents(content=..., mime_type="application/json")`

## Implementation Checklist for Session 3

- [ ] Read current `scripts/mcp/unified_server.py` structure
- [ ] Add resource entry to RESOURCES_CONFIG
- [ ] Add handler conditional in read_resource()
- [ ] Create test file for unified server integration
- [ ] Write 5-8 integration tests
- [ ] Manual verification with unified server
- [ ] Update documentation
- [ ] Verify backward compatibility
- [ ] All tests pass
- [ ] Document completion

## Success Criteria

✅ **Functional**
- Resource accessible via unified server
- Response includes all 11 artifact types
- JSON schema matches design doc

✅ **Quality**
- All new tests passing
- No regressions in existing tests
- Error handling working

✅ **Documentation**
- Changes documented in code comments
- Session completion report created

## Estimated Effort

| Task | Time |
|------|------|
| Code integration | 30-45 min |
| Test creation | 30-45 min |
| Verification | 15-20 min |
| Documentation | 15-20 min |
| **Total** | **90-130 min** |

## Notes for Next Session

1. **Imports**: May need to check if AgentQMS import path works in unified_server context
2. **Error Handling**: Unified server may need different error response format
3. **Testing**: Will need to mock or start actual unified server for integration tests
4. **Compatibility**: Check if unified server has different dependencies or versions

## Session Handover Summary

**Phase 1 Status**: ✅ Complete (Sessions 1-2)
- Discovery system: Working ✅
- MCP resource (AgentQMS): Working ✅
- 27 tests: All passing ✅

**Phase 2 Status**: 🟡 Starting (Session 3+)
- Unified server integration: TODO
- Plugin migration: TODO (later)

**Ready for Pickup**: Yes ✅
**Previous Session Docs**: All in `project_compass/active_context/`
