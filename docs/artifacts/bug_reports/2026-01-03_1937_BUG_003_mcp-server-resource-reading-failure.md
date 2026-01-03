title: "MCP Server resource/read Returns 'Unknown resource URI' Despite Correct Implementation"
author: "AI Assistant (Antigravity)"
timestamp: "2026-01-03 19:37 (KST)"
branch: "main"
type: "bug_report"
category: "infrastructure"
status: closed
severity: high
created: 2026-01-03 19:37
closed: 2026-01-03 20:05
tags: [mcp, agentqms, project_compass, bug, resolved]
---

# Bug Report - MCP Server Resource Reading Failure

Bug ID: BUG-003

## Summary

Both `project_compass` and `AgentQMS` MCP servers fail to read resources via the `resources/read` endpoint, consistently returning error `"Unknown resource URI: {uri}"` despite:
- Correct implementation of `read_resource` handler returning `ReadResourceResult`
- Valid URI matching logic verified through direct testing
- Proper MCP SDK types (v1.25.0) imported and used
- Successful module imports and server startup

## Environment

- **OS/Env**: Linux (Development Container)
- **Python**: 3.11.14
- **MCP SDK**: 1.25.0
- **Package Manager**: uv
- **Dependencies**:
  - `mcp` package installed via uv
  - Servers run via `uv run --directory /workspaces/upstageailab-ocr-recsys-competition-ocr-2 python {server_path}`

## Affected Components

1. **project_compass MCP Server** (`project_compass/mcp_server.py`)
   - Server name: `project_compass`
   - URIs affected: `compass://compass.json`, `compass://session_handover.md`, etc.

2. **AgentQMS MCP Server** (`AgentQMS/mcp_server.py`)
   - Server name: `agentqms`
   - URIs affected: `agentqms://standards/artifact_types`, etc.

3. **experiment_manager MCP Server** (`experiment_manager/mcp_server.py`)
   - Not tested but has identical pattern

## Reproduction Steps

1. Start MCP server via configured MCP client (Antigravity)
2. Call `list_resources` - **SUCCESS** (returns all resources correctly)
3. Call `read_resource` with valid URI (e.g., `compass://compass.json`) - **FAILS**
4. Error: `"calling 'resources/read': Unknown resource URI: compass://compass.json"`

## Comparison

### Expected Behavior
```python
read_resource(ServerName="project_compass", Uri="compass://compass.json")
# Should return: ReadResourceResult with TextResourceContents containing JSON
```

### Actual Behavior
```
Error: failed to read resource compass://compass.json:
       calling "resources/read": Unknown resource URI: compass://compass.json
```

The error is raised from line 145 of `read_resource` handler:
```python
if not resource:
    raise ValueError(f"Unknown resource URI: {uri}")
```

This indicates `next((r for r in RESOURCES if r["uri"] == uri), None)` returns `None`.

## Code Implementation (VERIFIED CORRECT)

### Handler Implementation
```python
@app.read_resource()
async def read_resource(uri: str) -> ReadResourceResult:
    """Read content of a compass resource."""
    resource = next((r for r in RESOURCES if r["uri"] == uri), None)

    if not resource:
        raise ValueError(f"Unknown resource URI: {uri}")

    path: Path = resource["path"]

    if not path.exists():
        raise FileNotFoundError(f"Resource file not found: {path}")

    content = path.read_text(encoding="utf-8")

    return ReadResourceResult(
        contents=[
            TextResourceContents(
                uri=uri,
                text=content,
                mimeType=resource["mimeType"]
            )
        ]
    )
```

### Verification Tests (ALL PASS)

```bash
# Test 1: Module import - SUCCESS
uv run python -c "from project_compass.mcp_server import app, RESOURCES; print('OK')"
# Output: OK

# Test 2: URI matching logic - SUCCESS
uv run python -c "
from project_compass.mcp_server import RESOURCES
test_uri = 'compass://compass.json'
match = next((r for r in RESOURCES if r['uri'] == test_uri), None)
print(f'Match found: {match is not None}')
"
# Output: Match found: True

# Test 3: Python syntax - SUCCESS
python -m py_compile project_compass/mcp_server.py
# Output: (no error)
```

## Investigation History

### Timeline of Debugging

1. **Initial Error**: "Unknown resource URI" from both servers
2. **Hypothesis 1**: Wrong return type → Fixed to `ReadResourceResult`
3. **Hypothesis 2**: Python cache → Cleared `__pycache__` and `.pyc` files
4. **Discovery**: Syntax error preventing server startup (extra `")` in FileNotFoundError)
5. **Fix Applied**: Removed syntax error
6. **Current State**: Module imports successfully, logic verified, but still fails

### Troubleshooting Actions Taken

- ✅ Fixed return type from `str` to `ReadResourceResult`
- ✅ Added proper imports (`ReadResourceResult`, `TextResourceContents`)
- ✅ Fixed syntax error on line 159
- ✅ Cleared all Python bytecode caches
- ✅ Killed and restarted MCP server processes (multiple times)
- ✅ Reloaded developer window (multiple times)
- ✅ Restarted development container
- ❌ **Issue persists**

## Logs

### Error Message
```
Encountered error in step execution: error executing cascade step:
CORTEX_STEP_TYPE_READ_RESOURCE:
failed to read resource compass://compass.json:
calling "resources/read": Unknown resource URI: compass://compass.json
```

### Server Process Status
```bash
$ ps aux | grep mcp_server.py
vscode  1343  uv run --directory /workspaces/... python AgentQMS/mcp_server.py
vscode  1472  /workspaces/.../python3 AgentQMS/mcp_server.py
vscode  2494  uv run python project_compass/mcp_server.py
vscode  2522  /workspaces/.../python3 project_compass/mcp_server.py
# Servers ARE running
```

### Direct Import Test
```python
>>> from project_compass.mcp_server import RESOURCES
>>> for r in RESOURCES:
...     print(r['uri'])
compass://compass.json
compass://session_handover.md
compass://current_session.yml
compass://uv_lock_state.yml
compass://agents.yaml
# URIs are correctly registered
```

## Impact

**Severity**: HIGH (blocks all MCP resource access)

### Blocked Functionality
- Cannot read project compass state via MCP
- Cannot access AgentQMS standards via MCP
- Cannot use MCP servers for their intended purpose

### Workarounds
- Direct file system access (works)
- Manual file reading (bypasses MCP)

### Affected Users
- AI assistants relying on MCP for context
- Automated tools using MCP client
- Project compass integrations

## Root Cause Analysis (CONFIRMED)

The issue was caused by two distinct problems interacting with the MCP SDK (v1.25.0):

1.  **URI Type Mismatch**: The `read_resource` handler receives `uri` as a Pydantic `AnyUrl` object, not a Python `str`, despite type hinting. This caused:
    *   Direct string comparisons (`r["uri"] == uri`) to fail.
    *   String methods (like `.strip()`) to raise `AttributeError`.
2.  **Return Type Mismatch**: The `read_resource` handler decorator in the SDK expects the function to return a `str`, `bytes`, or `Iterable[ReadResourceContents]`. It does *not* support returning a `ReadResourceResult` object directly. Returning `ReadResourceResult` caused the internal SDK code to try to access `.content` on what it thought was a `ReadResourceResult` but was actually treating as a tuple/object, leading to `'tuple' object has no attribute 'content'`.

## Resolution

1.  **Cast URI to String**: Updated `read_resource` handlers in both `AgentQMS/mcp_server.py` and `project_compass/mcp_server.py` to explicitly cast the input URI: `uri = str(uri).strip()`.
2.  **Correct Return Type**: Updated handlers to return `list[ReadResourceContents]` instead of `ReadResourceResult`.
3.  **Fix Imports**: Added `from mcp.server.lowlevel.helper_types import ReadResourceContents` to both server files to support the correct return type.

## Verification

*   Created `verify_fix.py` to simulate the SDK call pattern.
*   Confirmed that `AgentQMS` now successfully reads `agentqms://standards/artifact_types` and returns the correct `ReadResourceContents` object.
*   Confirmed that `project_compass` works identically.

## Proposed Solutions

### Immediate
1. Investigate MCP SDK source code for decorator caching behavior
2. Check Antigravity MCP client configuration and logs
3. Try alternative MCP SDK versions

### Long-term
1. Add integration tests for MCP server resource reading
2. Implement server-side debug logging to trace exact URIs received
3. Document MCP server development and caching gotchas

## Related Files

- [project_compass/mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/project_compass/mcp_server.py)
- [AgentQMS/mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/mcp_server.py)
- [experiment_manager/mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment_manager/mcp_server.py)
- [Investigation Walkthrough](file:///home/vscode/.gemini/antigravity/brain/156d7b16-5576-4bc6-8985-2f907945011d/walkthrough.md)

## Next Steps

1. ☐ Check MCP SDK GitHub issues for similar reports
2. ☐ Test with minimal reproducer outside Antigravity
3. ☐ Add comprehensive logging to track URI flow
4. ☐ Contact MCP SDK or Antigravity maintainers
5. ☐ Consider alternative MCP server implementation patterns
