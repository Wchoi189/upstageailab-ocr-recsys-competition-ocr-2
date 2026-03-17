#!/usr/bin/env python3
"""Smoke checks for MCP dispatch registry and unknown tool behavior."""

from __future__ import annotations

import asyncio
import json

from AgentQMS import mcp_server
from AgentQMS.tools.core.mcp.handlers import TOOL_HANDLERS as EXTRACTED_HANDLERS


def _decode_payload(text: str) -> dict:
    return json.loads(text)


async def _run_async_checks() -> None:
    expected = {
        "create_artifact",
        "validate_artifact",
        "list_artifact_templates",
        "check_compliance",
        "get_standard",
        "get_context_bundle",
    }

    assert set(EXTRACTED_HANDLERS.keys()) == expected, "handler registry keys mismatch"
    assert set(mcp_server.TOOL_HANDLERS.keys()) == expected, "mcp_server dispatch keys mismatch"

    unknown_result = await mcp_server.call_tool("unknown_tool_name", {})
    unknown_payload = _decode_payload(unknown_result[0].text)
    assert "Unknown tool" in unknown_payload.get("error", ""), "unknown tool response must include error"

    known_result = await mcp_server.call_tool("list_artifact_templates", {})
    known_payload = _decode_payload(known_result[0].text)
    assert "templates" in known_payload, "known handler dispatch must return expected payload shape"


def main() -> int:
    asyncio.run(_run_async_checks())
    print("mcp_dispatch_smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
