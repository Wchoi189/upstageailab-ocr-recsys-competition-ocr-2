#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

# Paths
WORKSPACE_ROOT = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2")
SHARED_CONFIG_PATH = WORKSPACE_ROOT / "scripts/mcp/shared_config.json"
GEMINI_CONFIG_PATH = Path("/home/vscode/.gemini/antigravity/mcp_config.json")
CLAUDE_CONFIG_DIR = Path("/home/vscode/.claude")
CLAUDE_CONFIG_PATH = CLAUDE_CONFIG_DIR / "config.json"
QWEN_CONFIG_PATH = WORKSPACE_ROOT / ".qwen/settings.json"
ENV_LOCAL_PATH = WORKSPACE_ROOT / ".env.local"


def load_env_local():
    """Load variables from .env.local into os.environ."""
    if not ENV_LOCAL_PATH.exists():
        return

    with open(ENV_LOCAL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                os.environ[key.strip()] = value


def expand_env_vars(obj):
    """Recursively expand environment variables in a dictionary/list."""
    if isinstance(obj, str):
        # Match ${VAR_NAME}
        def replace(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(r"\$\{([^}]+)\}", replace, obj)
    elif isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_env_vars(i) for i in obj]
    return obj


def load_json(path):
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}


def save_json(path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Successfully updated {path}")
    except Exception as e:
        print(f"Error saving {path}: {e}")


def main():
    shared_config = load_json(SHARED_CONFIG_PATH)
    if not shared_config or "mcpServers" not in shared_config:
        print(f"Error: Invalid or empty shared config at {SHARED_CONFIG_PATH}")
        return

    # Update Gemini Config
    gemini_config = load_json(GEMINI_CONFIG_PATH)
    if "mcpServers" not in gemini_config:
        gemini_config["mcpServers"] = {}

    # Merge servers (preserving placeholders)
    for server_name, server_config in shared_config["mcpServers"].items():
        gemini_config["mcpServers"][server_name] = server_config

    # Remove deprecated servers
    for deprecated in ["agent_debug_toolkit", "project_compass"]:
        if deprecated in gemini_config["mcpServers"]:
            del gemini_config["mcpServers"][deprecated]

    save_json(GEMINI_CONFIG_PATH, gemini_config)

    # Update Claude Config
    claude_config = load_json(CLAUDE_CONFIG_PATH)
    if "mcpServers" not in claude_config:
        claude_config["mcpServers"] = {}

    # Merge servers (preserving placeholders)
    for server_name, server_config in shared_config["mcpServers"].items():
        claude_config["mcpServers"][server_name] = server_config

    # Remove deprecated servers
    for deprecated in ["agent_debug_toolkit", "project_compass"]:
        if deprecated in claude_config["mcpServers"]:
            del claude_config["mcpServers"][deprecated]

    save_json(CLAUDE_CONFIG_PATH, claude_config)

    # Update Qwen Config (Project level)
    qwen_config = load_json(QWEN_CONFIG_PATH)
    if not qwen_config:
        qwen_config = {"$version": 2}  # Default structure if missing
    if "mcpServers" not in qwen_config:
        qwen_config["mcpServers"] = {}

    # Merge servers (preserving placeholders)
    for server_name, server_config in shared_config["mcpServers"].items():
        qwen_config["mcpServers"][server_name] = server_config

    # Remove deprecated servers
    for deprecated in ["agent_debug_toolkit", "project_compass"]:
        if deprecated in qwen_config["mcpServers"]:
            del qwen_config["mcpServers"][deprecated]

    save_json(QWEN_CONFIG_PATH, qwen_config)


if __name__ == "__main__":
    main()
