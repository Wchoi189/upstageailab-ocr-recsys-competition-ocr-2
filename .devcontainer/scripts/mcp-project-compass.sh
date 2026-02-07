#!/bin/bash
cd /workspaces
exec uv run python dev_tools/project_compass/project_compass/mcp_server.py
