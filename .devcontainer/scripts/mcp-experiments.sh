#!/bin/bash
cd /workspaces
exec uv run python dev_tools/experiment_manager/src/etk/mcp_server.py
