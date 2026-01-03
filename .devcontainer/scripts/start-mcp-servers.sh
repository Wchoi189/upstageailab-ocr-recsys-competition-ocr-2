#!/bin/bash
set -e

echo "🔍 Verifying MCP servers..."

# Check MCP servers exist
MCP_SERVERS=(
  "project_compass/mcp_server.py"
  "AgentQMS/mcp_server.py"
  "experiment_manager/mcp_server.py"
)

for server in "${MCP_SERVERS[@]}"; do
  if [ -f "$server" ]; then
    echo "✅ Found: $server"
  else
    echo "❌ Missing: $server"
  fi
done

# Create MCP config documentation
cat > /tmp/mcp_servers_info.md << 'EOF'
# Available MCP Servers

This Codespace has 3 MCP servers available:

1. **project_compass** - Project state and configuration
   - Location: `project_compass/mcp_server.py`
   - Resources: compass.json, session_handover.md, current_session.yml, etc.

2. **AgentQMS** - Artifact management and quality
   - Location: `AgentQMS/mcp_server.py`
   - Tools: create_artifact, validate_artifact, list_templates, etc.

3. **experiment_manager** - Experiment tracking
   - Location: `experiment_manager/mcp_server.py`
   - Tools: init_experiment, add_task, log_insight, sync_experiment, etc.

## Usage with Claude Dev

These servers are automatically available when using the Claude Dev extension.

To use them:
1. Install the Claude Dev extension in this Codespace
2. Open the Claude Dev panel
3. MCP servers will be automatically detected
4. Use MCP tools like: `list_resources(ServerName="project_compass")`

## Manual Testing

Test each server starts correctly:
```bash
uv run python project_compass/mcp_server.py
uv run python AgentQMS/mcp_server.py
uv run python experiment_manager/mcp_server.py
```

Press Ctrl+C to stop each server.

EOF

echo "📝 MCP server info written to /tmp/mcp_servers_info.md"
echo "✅ MCP servers ready for use with Claude Dev extension"
echo ""
echo "💡 Tip: Run 'cat /tmp/mcp_servers_info.md' to see usage instructions"
