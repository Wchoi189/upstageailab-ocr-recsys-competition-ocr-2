#!/usr/bin/env python3
"""
Refresh MCP Tools Configuration
Aggregates tool definitions from component servers and generates unified config.
"""
import asyncio
import sys
import yaml
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# We rely on PYTHONPATH being set correctly by the caller (Makefile)


# Definition of component servers to aggregate
COMPONENT_SERVERS = [
    {
        "name": "agentqms",
        "module": "AgentQMS.mcp_server",
        "description": "Agent Quality Management System"
    },
    {
        "name": "project_compass",
        "module": "project_compass.mcp_server",
        "description": "Project Compass & Session Management"
    },
    {
        "name": "experiment_manager",
        "module": "experiment_manager.mcp_server",
        "description": "Experiment Lifecycle Manager"
    },
    {
        "name": "agent_debug_toolkit",
        "module": "agent_debug_toolkit.mcp_server",
        "description": "Agent Debug Toolkit (ADT)"
    }
]

async def get_tools_from_module(module_name: str) -> list[dict]:
    """Import module and get tools list."""
    try:
        if module_name not in sys.modules:
            # Dynamic import
            import importlib
            mod = importlib.import_module(module_name)
        else:
            mod = sys.modules[module_name]

        # Check for list_tools function
        if hasattr(mod, "list_tools"):
            # It might be an async generator or function
            tools = await mod.list_tools()

            # Convert mcp.types.Tool to dict
            tool_dicts = []
            for t in tools:
                # Handle Pydantic model dump if available, or direct attr access
                if hasattr(t, "model_dump"):
                    t_dict = t.model_dump()
                elif hasattr(t, "dict"):
                    t_dict = t.dict()
                else:
                    t_dict = {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.inputSchema
                    }
                tool_dicts.append(t_dict)
            return tool_dicts

    except Exception as e:
        print(f"❌ Error loading {module_name}: {e}")
        import traceback
        traceback.print_exc()
        return []
    return []

async def main():
    print(f"🔄 Refreshing MCP Tools from {len(COMPONENT_SERVERS)} components...")

    aggregated_tools = []

    for server in COMPONENT_SERVERS:
        module_name = server["module"]
        print(f"   scanning {server['name']} ({module_name})...")

        tools = await get_tools_from_module(module_name)
        print(f"     -> found {len(tools)} tools")

        for tool in tools:
            # Inject "force" argument into inputSchema if properties exist
            if "inputSchema" in tool and "properties" in tool["inputSchema"]:
                tool["inputSchema"]["properties"]["force"] = {
                    "type": "boolean",
                    "description": "Force execution, bypassing some compliance checks (use with caution)",
                    "default": False
                }

            # Inject implementation metadata
            tool["implementation"] = {
                "module": module_name,
                "function": "call_tool"
            }
            # Add to list
            aggregated_tools.append(tool)

    # Dedup by name (last wins logic if needed, or error)
    # We'll use a dict to check duplicates
    seen = {}
    final_list = []
    for t in aggregated_tools:
        name = t["name"]
        if name in seen:
            print(f"⚠️  Duplicate tool reference: {name} from {t['implementation']['module']} overwriting {seen[name]}")
        seen[name] = t["implementation"]["module"]
        final_list.append(t)

    # Write to tools.yaml
    output_path = PROJECT_ROOT / "scripts" / "mcp" / "config" / "tools.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(final_list, f, sort_keys=False, indent=2)

    print(f"\n✅ Successfully wrote {len(final_list)} tools to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
