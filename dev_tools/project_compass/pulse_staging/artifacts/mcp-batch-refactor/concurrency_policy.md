# Concurrency Policy Map

Date: 2026-02-12
Scope: unified_server dispatcher routing for project_compass and agent_debug_toolkit.

## Policy Principles
- Keep existing tool names and inputs unchanged.
- Isolate failures per tool call.
- Serialize Compass state writes to prevent vessel_state.json conflicts.
- Offload heavy ADT analyzers to background threads; limit concurrent heavy work.

## Dispatcher-Level Policy (per module)

### project_compass.mcp_server
- Global mutex: required for all state-mutating calls.
- Read-only calls can run concurrently with each other, but not with writes.

Write (exclusive):
- compass_meta_pulse: kind in [init, sync, export, checkpoint]
- compass_meta_spec: all kinds (writes artifacts)
- pulse_init, pulse_sync, pulse_export, pulse_checkpoint
- spec_constitution, spec_specify, spec_plan, spec_tasks

Read (shared):
- pulse_status
- vessel://state, vessel://rules, vessel://staging (resource reads)

### agent_debug_toolkit.mcp_server
- Allow concurrency but apply limits:
  - adt_meta_query: semaphore limit (heavy analysis can saturate CPU).
  - adt_meta_edit: semaphore limit 1 (edits should not overlap on filesystem).

Recommended limits:
- adt_meta_query: max_concurrency = 2
- adt_meta_edit: max_concurrency = 1

Heavy query kinds (prefer background thread and counted against query semaphore):
- dependency_graph, context_tree, symbol_search, sg_search, sg_lint,
  ts_parse, ts_query, complexity, config_flow

Light query kinds (background thread optional):
- config_access, merge_order, hydra_usage, component_instantiations, imports, ast_dump

### scripts/mcp/unified_server.py
- Enforce module-specific semaphores before invoking module.call_tool.
- Apply per-tool timeout (default 60s, override for heavy analyzers if needed).
- Return structured error payload with request_id and tool_name; do not cancel siblings.

## Error Payload Schema (dispatcher)
- {"status": "error", "tool": "<name>", "module": "<module>", "request_id": "<uuid>", "message": "<short>", "detail": "<truncated>"}

## Telemetry Ordering Fields
- request_id (uuid4)
- parent_id (optional, for meta-tools)
- sequence_index (monotonic per session)
