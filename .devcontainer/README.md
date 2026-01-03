# Development Container Configurations

This directory contains multiple devcontainer configurations optimized for different environments.

## Available Configurations

### 1. `devcontainer.json` (Local Docker Development)
**Use when**: Running locally with Docker Desktop

**Features**:
- GPU support (`--gpus all`)
- Local data directory mount
- Full development environment
- Optimized for local machine with GPU

**Requirements**:
- Docker Desktop with GPU support
- NVIDIA drivers (for GPU)
- Local data directory at `./data`

### 2. `devcontainer.codespaces.json` (GitHub Codespaces)
**Use when**: Running in GitHub Codespaces

**Features**:
- Cloud-optimized (no GPU requirements)
- No local mounts (cloud environment)
- MCP server auto-start
- Claude Dev extension pre-installed
- Free tier optimized (4 CPUs, 16GB RAM, 15GB storage)

**Automatic Selection**: GitHub Codespaces automatically uses this config when available

## MCP Servers

This project includes 3 MCP (Model Context Protocol) servers:

1. **project_compass** - Project state and configuration
2. **agentqms** - Artifact management and quality
3. **experiments** - Experiment tracking

### MCP Configuration

- **Config file**: `mcp_config.json`
- **Startup script**: `scripts/start-mcp-servers.sh`
- **Auto-start**: Enabled in Codespaces via `postStartCommand`

### Using MCP Servers

**In Codespaces with Claude Dev**:
1. MCP servers are automatically detected
2. Use MCP tools directly in Claude Dev
3. Example: `list_resources(ServerName="project_compass")`

**Manual Testing**:
```bash
# Test individual servers
uv run python project_compass/mcp_server.py
uv run python AgentQMS/mcp_server.py
uv run python experiment_manager/mcp_server.py
```

## Scripts

### `scripts/start-mcp-servers.sh`
Verifies MCP servers exist and creates usage documentation at `/tmp/mcp_servers_info.md`

**Run manually**:
```bash
bash .devcontainer/scripts/start-mcp-servers.sh
cat /tmp/mcp_servers_info.md
```

## Switching Between Configurations

### Local Development → Codespaces
1. Commit and push your changes
2. Go to GitHub repo → Code → Codespaces
3. Click "Create codespace on main"
4. Codespaces will use `devcontainer.codespaces.json` automatically

### Codespaces → Local Development
1. Clone the repo locally
2. Open in VS Code
3. VS Code will use `devcontainer.json` for local Docker

## Troubleshooting

### Codespace won't start
- Check image is public: `ghcr.io/wchoi189/upstageailab-ocr-recsys-competition-ocr-2:latest`
- Verify GitHub Actions built the image successfully
- Check Codespaces logs for errors

### MCP servers not detected
- Run startup script manually: `bash .devcontainer/scripts/start-mcp-servers.sh`
- Check `/tmp/mcp_servers_info.md` for server status
- Verify Claude Dev extension is installed

### Data not available in Codespaces
- Codespaces doesn't have access to local `./data` directory
- Options:
  - Commit sample data to repo (small datasets only)
  - Download data in `postCreateCommand`
  - Use cloud storage (S3/GCS) with credentials

## Resources

- [Devcontainer Specification](https://containers.dev/)
- [GitHub Codespaces Docs](https://docs.github.com/en/codespaces)
- [MCP Protocol](https://modelcontextprotocol.io/)
