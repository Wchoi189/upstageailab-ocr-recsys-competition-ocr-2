#!/usr/bin/env python3
"""
Validation harness for MCP batch dispatcher behavior.

Runs 1,000 mixed tool calls concurrently to verify:
- failure isolation (one induced failure does not block others)
- telemetry includes ordering fields (request_id, sequence_index)
"""

import asyncio
import json
import time
from pathlib import Path

from scripts.mcp import unified_server

TOTAL_CALLS = 1000
FAILURE_INDEX = 37
MAX_CONCURRENCY = 50
MAX_RUNTIME_S = 480


def _telemetry_path() -> Path:
    return unified_server.TELEMETRY_FILE


def _read_new_telemetry_lines(path: Path, start_pos: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(start_pos)
        return [line.rstrip("\n") for line in handle if line.strip()]


def _is_error_payload(text: str) -> bool:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(payload, dict) and payload.get("status") == "error"


async def _run_batch() -> list[list[unified_server.TextContent]]:
    unified_server.TOOLS_DEFINITIONS = await unified_server.load_tools_from_servers()

    base_calls = [
        ("list_artifact_templates", {}),
        ("list_artifact_templates", {}),
        ("compass_meta_pulse", {"kind": "status"}),
        (
            "adt_meta_query",
            {
                "kind": "ast_dump",
                "target": "x = 1\nprint(x)\n",
                "options": {"lang": "python"},
            },
        ),
        ("list_artifact_templates", {}),
    ]

    specs: list[tuple[str, dict]] = []
    for idx in range(TOTAL_CALLS - 1):
        specs.append(base_calls[idx % len(base_calls)])

    specs.insert(
        FAILURE_INDEX,
        (
            "adt_meta_query",
            {
                "kind": "config_access",
                "target": "scripts/mcp/DOES_NOT_EXIST.py",
            },
        ),
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _invoke(spec: tuple[str, dict]) -> list[unified_server.TextContent]:
        name, args = spec
        async with semaphore:
            return await unified_server.call_tool(name, args)

    tasks = [asyncio.create_task(_invoke(spec)) for spec in specs]
    return await asyncio.gather(*tasks)


def test_mcp_mixed_batch_validation():
    telemetry_path = _telemetry_path()
    start_pos = telemetry_path.stat().st_size if telemetry_path.exists() else 0

    start_time = time.perf_counter()
    results = asyncio.run(_run_batch())
    duration_s = time.perf_counter() - start_time
    assert duration_s <= MAX_RUNTIME_S, (
        f"Batch runtime exceeded budget: {duration_s:.2f}s > {MAX_RUNTIME_S:.2f}s"
    )

    error_payloads = 0
    for result in results:
        if result and hasattr(result[0], "text") and _is_error_payload(result[0].text or ""):
            error_payloads += 1

    assert error_payloads == 1, f"Expected 1 error payload, got {error_payloads}"

    new_lines = _read_new_telemetry_lines(telemetry_path, start_pos)
    assert len(new_lines) >= TOTAL_CALLS, (
        f"Expected at least {TOTAL_CALLS} telemetry events, got {len(new_lines)}"
    )

    parsed = [json.loads(line) for line in new_lines[:TOTAL_CALLS]]
    missing_fields = [
        event
        for event in parsed
        if "request_id" not in event or "sequence_index" not in event
    ]
    assert not missing_fields, "Telemetry events missing request_id or sequence_index"


if __name__ == "__main__":
    test_mcp_mixed_batch_validation()
