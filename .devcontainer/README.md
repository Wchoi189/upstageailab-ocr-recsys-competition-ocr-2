# DevContainer Configurations

## Default: Cloud-Optimized
**File**: `devcontainer.json`
**Use**: Codespaces, Gitpod, cloud environments
**Features**: No GPU, build fallback, auto-discovered

## Local: GPU-Enabled
**File**: `local/devcontainer.json`
**Use**: Docker Desktop + NVIDIA GPU
**Features**: GPU support, local data mounts
**Selection**: Manual via VS Code UI

## MCP Servers
**Config**: `mcp_config.json` (portable paths)
**Schema**: `mcp_servers.yaml` (machine-parseable)
**Startup**: `scripts/start-mcp-servers.sh`

## Usage

### Cloud (Default)
Opens automatically in Codespaces/cloud environments.

### Local GPU
1. Open folder in VS Code
2. Command Palette: "Dev Containers: Reopen in Container"
3. Select: "From .devcontainer/local/devcontainer.json"

Or via CLI:
```bash
devcontainer open --workspace-folder . --config .devcontainer/local/devcontainer.json
```
