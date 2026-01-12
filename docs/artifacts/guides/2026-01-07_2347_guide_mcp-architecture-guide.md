---
title: MCP Server Architecture Guide
date: 2026-01-07 23:47 (KST)
type: guide
category: architecture
status: completed
version: 1.0
ads_version: 1.0
---

# MCP Server Architecture Guide

## TL;DR Recommendation

✅ **Keep your unified server** - It's the right choice for your use case
⚠️ **Extract handlers at 1000+ lines** - Use module pattern when file grows
📋 **Config-based toggles** - Better than separate processes for optional features

---

## Unified vs Separate Servers

### When to Use Unified (Your Current Approach) ✅

**Benefits**:
- **Single process**: Lower memory/CPU overhead
- **Easier sync**: One config file across 4 agents
- **Shared infrastructure**: SymbolTable, caches, utilities
- **Simpler deployment**: One server to start/monitor

**Best For**:
- Related tools from same codebase (ADT tools) ✅
- Tools sharing state/infrastructure ✅
- Core project functionality ✅
- Teams with multiple agents using same tools ✅

### When to Separate

**Use separate servers for**:
1. **External integrations**: Perplexity, GitHub API, Jira
   - Different auth/credentials
   - Independent lifecycle
   - Third-party dependencies

2. **Optional heavy features**: ML model serving, image processing
   - Not all agents need them
   - Resource-intensive
   - Can be toggled off globally

3. **Different tech stacks**: Node.js tools, system utilities
   - Can't share Python infrastructure
   - Different packaging/deployment

---

## Managing File Growth

### Current State: ~550 Lines ✅ Manageable

Your [mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/mcp_server.py) is well-structured at 550 lines. No action needed yet.

### At 1000+ Lines: Extract Handlers

**Pattern 1: Module-based handlers**
```python
# mcp_server.py (main file)
from .handlers import adt_core, adt_advanced, agentqms

TOOL_GROUPS = {
    **adt_core.TOOLS,
    **adt_advanced.TOOLS,  # context_tree, intelligent_search
    **agentqms.TOOLS,
}

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name in adt_core.TOOLS:
        return await adt_core.handle(name, arguments)
    if name in adt_advanced.TOOLS:
        return await adt_advanced.handle(name, arguments)
    ...
```

**Pattern 2: Registry pattern**
```python
# registry.py
HANDLERS = {}

def register_tool(name: str):
    def decorator(func):
        HANDLERS[name] = func
        return func
    return decorator

# adt_handlers.py
@register_tool("intelligent_search")
async def handle_intelligent_search(arguments):
    ...
```

### At 2000+ Lines: Split Servers

Only if handlers grow unmaintainably large. Consider:
- `adt_core_server.py` - Basic analysis tools
- `adt_advanced_server.py` - Phase 3+ tools
- Both can still run in same process if needed

---

## Config-Based Feature Toggles

Better than separate processes for optional features:

```yaml
# mcp_config.yaml
tool_groups:
  enabled:
    - adt_core
    - adt_advanced
    - agentqms
  disabled:
    - adt_experimental  # Not loaded at all
```

```python
# mcp_server.py
import yaml

with open("mcp_config.yaml") as f:
    config = yaml.safe_load(f)

enabled_groups = set(config["tool_groups"]["enabled"])

TOOLS = []
if "adt_core" in enabled_groups:
    TOOLS.extend(ADT_CORE_TOOLS)
if "adt_advanced" in enabled_groups:
    TOOLS.extend(ADT_ADVANCED_TOOLS)
```

**Advantages**:
- No process overhead
- Easy to sync across agents (one YAML file)
- Can enable/disable per environment
- Simpler than maintaining multiple servers

---

## Your Architecture Assessment

### Current Setup

- **unified_server.py**: ~600 lines, all project tools
- **adt mcp_server.py**: ~550 lines, ADT-specific
- **Total tools**: ~18 tools across both

### Recommendation

**Short term (now)**: ✅ Status quo is fine
- File sizes manageable
- Clear separation (unified vs ADT-specific)
- Good for 4 agents sharing config

**Medium term (1000+ lines)**: Extract handlers
```
agent-debug-toolkit/
  src/agent_debug_toolkit/
    mcp_server.py          # Main entry point (~150 lines)
    mcp_handlers/
      __init__.py
      core.py              # analyze_config, trace_merge, etc.
      advanced.py          # context_tree, intelligent_search
      phase1.py            # dependency_graph, imports, complexity
```

**Long term (if needed)**: Config toggles
```yaml
# .mcp_config.json
{
  "mcpServers": {
    "unified_project": {
      "command": "python",
      "args": ["scripts/mcp/unified_server.py"],
      "env": {
        "TOOL_GROUPS": "compass,agentqms,etk,adt_core"
        # Omit adt_advanced if not needed
      }
    }
  }
}
```

---

## Process Overhead Analysis

### Your Concern: "Forking separate processes multiplies MCP servers"

**Reality**:
- Each MCP server = 1 Python process (~50-100MB RAM)
- 4 agents × 3 servers = 12 processes (~600-1200MB)
- Your unified approach: 4 agents × 1 server = 4 processes (~200-400MB)

**Savings**: ~400-800MB RAM, simpler architecture ✅

### Sharing Between Agents

**Your approach** (unified):
```json
// Same mcp_config.json for all 4 agents
{
  "mcpServers": {
    "unified_project": {
      "command": "python",
      "args": ["scripts/mcp/unified_server.py"]
    }
  }
}
```

**Alternative** (separate):
```json
// Need to sync 3× files across 4 agents = 12 config entries
{
  "mcpServers": {
    "compass": {...},
    "agentqms": {...},
    "adt": {...}
  }
}
```

**Winner**: Unified for your use case ✅

---

## Summary

| Aspect        | Unified (You)               | Separate              |
| ------------- | --------------------------- | --------------------- |
| Process count | 1                           | N                     |
| Memory        | ~50-100MB                   | ~50-100MB × N         |
| Config sync   | 1 file                      | N files               |
| File length   | Can grow large              | Smaller per server    |
| Best for      | Related tools, shared infra | External integrations |

**Verdict**: Your unified approach is **optimal** for:
- ✅ 4 agents sharing same tools
- ✅ Related ADT/AgentQMS/Compass tools
- ✅ Minimizing resource overhead

**Action items**:
- ✅ Keep unified for now
- 📋 Extract handlers when file hits 1000 lines
- 💡 Consider config toggles for optional tool groups
