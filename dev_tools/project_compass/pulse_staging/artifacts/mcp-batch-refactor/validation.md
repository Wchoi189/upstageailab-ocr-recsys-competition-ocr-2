# MCP Batch Refactor Validation Notes

Date: 2026-02-12

## Manual Validation Checklist
- Mixed-call batch: 1,000 tool calls across unified_server, Compass, and ADT with 0 sibling-call errors.
- Failure isolation: force one ADT call to fail; confirm other calls still complete.
- Compass compliance: all artifacts under pulse_staging/artifacts.
- Performance: median latency within +10% baseline.

## Suggested Test Scaffolding (follow-up)
- Add a pytest suite that drives concurrent call_tool invocations with asyncio.gather.
- Add fixtures for tool selection and injected failures.
- Assert telemetry includes request_id and sequence_index.
