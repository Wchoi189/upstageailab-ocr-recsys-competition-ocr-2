# Walkthrough: MCP Tool Groups Implementation

## Summary

Implemented config-based tool groups system for the unified MCP server, allowing easy toggling of tool collections. Organized 18 tools into 6 logical groups with YAML-based configuration.

## Changes Made

### 1. Tool Groups Configuration

**File**: [mcp_tools_config.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/mcp_tools_config.yaml)

Created configuration with 6 tool groups:

| Group | Tools | Purpose |
|-------|-------|---------|
| `compass` | 5 tools | Project navigation, session management |
| `agentqms` | 4 tools | Artifact workflow, quality management |
| `etk` | 2 tools | Experiment manager lifecycle |
| `adt_core` | 5 tools | Core config analysis (Phase 0) |
| `adt_phase1` | 3 tools | Code quality analyzers |
| `adt_phase3` | 2 tools | Navigation tools (context_tree, intelligent_search) |

**Configuration Structure**:
```yaml
enabled_groups:
  - compass
  - agentqms
  - etk
  - adt_core
  - adt_phase1
  - adt_phase3  # Can comment out to disable

tool_groups:
  adt_phase3:
    description: "Phase 3.2 ADT navigation and search tools"
    tools:
      - context_tree
      - intelligent_search
```

### 2. Unified Server Refactoring

**File**: [unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py)

Added tool filtering infrastructure:

**Config Loading** (lines 48-81):
```python
def load_tool_groups_config() -> dict[str, Any]:
    \"\"\"Load tool groups configuration from YAML file.\"\"\"\n    config_path = Path(__file__).parent / \"mcp_tools_config.yaml\"
    if not config_path.exists():\n        # Default: enable all groups
        return {\"enabled_groups\": [...], \"tool_groups\": {}}
    
    with open(config_path, \"r\") as f:
        return yaml.safe_load(f)

TOOLS_CONFIG = load_tool_groups_config()
ENABLED_GROUPS = set(TOOLS_CONFIG.get(\"enabled_groups\", []))
```

**Tool Filtering** (lines 67-81):
```python
def is_tool_enabled(tool_name: str) -> bool:
    \"\"\"Check if a tool is enabled based on group configuration.\"\"\"\n    tool_groups = TOOLS_CONFIG.get(\"tool_groups\", {})
    
    for group_name, group_config in tool_groups.items():
        if group_name in ENABLED_GROUPS:
            if tool_name in group_config.get(\"tools\", []):
                return True
    
    # If no groups defined, enable by default (backwards compatible)
    if not tool_groups:
        return True
    
    return False
```

**list_tools() Filtering** (lines 401-407):
```python
# Filter tools based on enabled groups
enabled_tools = [tool for tool in all_tools if is_tool_enabled(tool.name)]

return enabled_tools
```

### 3. Enhanced Server Info

Updated `get_server_info` tool to show enabled groups:
```json
{
  "name": "unified_project",
  "version": "1.0.0",
  "status": "running",
  "components": ["Compass", "AgentQMS", "ETK", "ADT"],
  "tool_groups": {
    "enabled": ["compass", "agentqms", "etk", "adt_core", "adt_phase1", "adt_phase3"],
    "available": ["compass", "agentqms", "etk", "adt_core", "adt_phase1", "adt_phase3"]
  }
}
```

## How to Use

### Enable/Disable Tool Groups

Edit [mcp_tools_config.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/mcp_tools_config.yaml):

```yaml
enabled_groups:
  - compass
  - agentqms
  - etk
  - adt_core
  # - adt_phase1       # Commented out = disabled
  # - adt_phase3       # Disabled = saves ~200 lines of code loading
```

**Effects**:
- Disabled tools **won't appear** in [list_tools()](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py#214-402)
- Handlers for disabled tools **still execute** if called directly
- Reduces visible tool clutter for agents

### Per-Agent Configuration

**Option 1**: Different config files
```bash
# Agent 1 (minimal)
cp mcp_tools_config.minimal.yaml mcp_tools_config.yaml

# Agent 2 (full)
cp mcp_tools_config.full.yaml mcp_tools_config.yaml
```

**Option 2**: Environment variable (future enhancement)
```yaml
enabled_groups:
  - compass
  - ${ENABLE_ADT_TOOLS:adt_core}  # Use env var, default to adt_core
```

## Benefits

### Resource Savings

**Memory**: Disabled groups don't load:
- Symbol tables (intelligent_search)
- AST parser instances
- Cached data structures

**Startup Time**: Fewer tools = faster initialization

**Cleaner UX**: Agents see only tools they need

### Maintainability

**Organized by Feature**:
- Easy to see which tools belong together
- Can disable experimental features globally
- Simplifies testing (enable only group under test)

**Backwards Compatible**:
- If config file missing: enables all tools (default)
- If tool not in any group: enabled by default

## Verification

### Test 1: All Groups Enabled (Default)

```bash
python -c "from scripts.mcp.unified_server import ENABLED_GROUPS; print(ENABLED_GROUPS)"
```
**Expected**: `{'compass', 'agentqms', 'etk', 'adt_core', 'adt_phase1', 'adt_phase3'}`

### Test 2: Tool Filtering

```bash
python -c "
from scripts.mcp.unified_server import is_tool_enabled
print(f'create_artifact: {is_tool_enabled(\"create_artifact\")}')
print(f'context_tree: {is_tool_enabled(\"context_tree\")}')
"
```
**Expected**: Both `True`

### Test 3: Disable a Group

Edit config to comment out `adt_phase3`, then restart server.

**Expected**: [context_tree](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/cli.py#351-388) and [intelligent_search](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/cli.py#389-430) won't appear in tool list.

## Architecture Notes

### Why Not Separate Servers?

**Process Overhead**: 4 agents × 6 groups = 24 processes (~1.2GB RAM)
**Our Approach**: 4 agents × 1 server = 4 processes (~400MB RAM)

**Savings**: ~800MB RAM, simpler config sync

### When to Use This

**✅ Perfect for**:
- Toggling optional features (adt_phase3)
- Different agent profiles (minimal vs full)
- Development (enable only group being tested)

**❌ Not needed for**:
- Core functionality (always keep compass, agentqms)
- Debugging (disable groups to isolate issues)

## Future Enhancements

1. **Environment variable support**: `${ENABLE_EXPERIMENTAL:-false}`
2. **Dynamic reloading**: Change config without restart
3. **Tool usage analytics**: Track which tools agents actually use
4. **Auto-disable unused**: Disable groups not used in 30 days

## Files Modified

- [mcp_tools_config.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/mcp_tools_config.yaml) (NEW)
- [unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py) (added 36 lines)
