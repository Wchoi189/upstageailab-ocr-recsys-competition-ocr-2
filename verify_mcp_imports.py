import sys
import importlib

modules = [
    "AgentQMS.mcp_server",
    "project_compass.mcp_server",
    "experiment_manager.mcp_server",
    "agent_debug_toolkit.mcp_server",
    "AgentQMS.tools.utils.config.loader",
    "AgentQMS.tools.utils.system.paths"
]

print("Verifying imports...")
for mod_name in modules:
    try:
        importlib.import_module(mod_name)
        print(f"✅ {mod_name} imported successfully")
    except ImportError as e:
        print(f"❌ {mod_name} FAILED: {e}")
    except Exception as e:
        print(f"❌ {mod_name} ERROR: {e}")
