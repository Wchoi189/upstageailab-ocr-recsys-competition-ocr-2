#!/bin/bash
set -e

echo "🔍 Verifying MCP servers..."

# Check MCP servers exist
MCP_SERVERS=(
  "project_compass/mcp_server.py"
  "AgentQMS/mcp_server.py"
  "experiment_manager/mcp_server.py"
)

ALL_FOUND=true
for server in "${MCP_SERVERS[@]}"; do
  if [ -f "$server" ]; then
    echo "✅ $server"
  else
    echo "❌ $server"
    ALL_FOUND=false
  fi
done

if [ "$ALL_FOUND" = true ]; then
  echo "✅ All MCP servers found"
  exit 0
else
  echo "❌ Some MCP servers missing"
  exit 1
fi
