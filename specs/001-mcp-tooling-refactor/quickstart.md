# Quickstart

## Prerequisites
- Python 3.11
- `uv` installed and environment synced

## Run unified MCP server (stdio)
```bash
uv run python scripts/mcp/unified_server.py --transport stdio
```

## Run unified MCP server (SSE)
```bash
uv run python scripts/mcp/unified_server.py --transport sse --host 0.0.0.0 --port 8000
```

## Smoke test
```bash
uv run pytest tests/test_unified_server_artifact_types.py
```
