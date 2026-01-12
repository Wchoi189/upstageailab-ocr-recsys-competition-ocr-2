# Walkthrough - Fix AgentQMS MCP Resource Reading

> [!IMPORTANT]
> Resolved "Method not found" error when reading `agentqms://` resources.

## Issue Description
User reported failure to read `agentqms://standards/artifact_types` with error:
`failed to read resource agentqms://standards/artifact_types: calling "resources/read": Method not found`

## Investigation
1. **Server Analysis**: Examined `AgentQMS/mcp_server.py`.
2. **Comparison**: Compared imports/setup with working `project_compass/mcp_server.py`.
3. **Environment**: Checked `mcp` package presence (`pip list`, `pip show` failed, but `uv run` worked).
4. **Import Check**: Verified `ReadResourceResult` and `ReadResourceContents` availability in `mcp` library using `check_mcp_imports.py`. Both were found, but `ReadResourceResult` was unused in `AgentQMS`.

## Changes Applied
- **Aligned Imports**: Modified `AgentQMS/mcp_server.py` to match `project_compass` imports exactly.
- **Removed Unused Types**: Removed `ReadResourceResult` and `TextResourceContents` from imports as they were unused and potentially causing version/environment specific issues.
- **Code Change**:
  ```python
  # Before
  from mcp.types import ReadResourceResult, Resource, TextContent, TextResourceContents, Tool

  # After
  from mcp.types import Resource, Tool, TextContent
  ```

## Verification
- **Resource Read**: Successfully read `agentqms://standards/artifact_types` using the `read_resource` tool after the fix.
- **Server Functionality**: Confirmed `list_resources` and `read_resource` are operational.

## Next Steps
- If the issue persists in the user's specific environment, ensure the `mcp-server-agentqms` configuration uses the same `uv` environment or correct python interpreter.
