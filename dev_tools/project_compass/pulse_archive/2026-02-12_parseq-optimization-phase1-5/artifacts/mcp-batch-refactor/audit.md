# MCP Tooling Refactor Audit

Date: 2026-02-12
Scope: unified_server dispatcher, Project Compass MCP server, Agent Debug Toolkit MCP server.

## Survey: Tool Handlers

### Unified dispatcher (scripts/mcp/unified_server.py)
- Entry: call_tool() routes by tool name to module call_tool.
- Current behavior: no concurrency policy, no timeouts, returns text error payloads on exceptions.
- Risk: sibling-call failures when batched; error propagation unclear.

### Project Compass MCP (dev_tools/project_compass/project_compass/mcp_server.py)
- Meta-tools: compass_meta_pulse, compass_meta_spec.
- Pulse handlers: pulse_init, pulse_sync, pulse_export, pulse_status, pulse_checkpoint.
- Spec handlers: spec_constitution, spec_specify, spec_plan, spec_tasks.
- Writes: vessel_state.json, staging artifacts, history exports.
- Risk: concurrent writes to vessel_state.json and staging artifacts.

### Agent Debug Toolkit MCP (dev_tools/agent_debug_toolkit/agent_debug_toolkit/mcp_server.py)
- Meta-tools exposed: adt_meta_query, adt_meta_edit (tools.yaml).
- Query kinds: config_access, merge_order, hydra_usage, component_instantiations, config_flow,
  dependency_graph, imports, complexity, context_tree, symbol_search, sg_search, sg_lint,
  ast_dump, ts_parse, ts_query.
- Edit kinds: apply_diff, smart_edit, read_slice, format.
- Risk: heavy analyzers run on event loop; edit tools mutate files; exceptions bubble out.

## Baseline Classification

### Read-only / low risk
- Compass: pulse_status (read), vessel://state, vessel://rules, vessel://staging.
- ADT: read-only analyzers and read_slice (query side).

### Write / stateful
- Compass: pulse_init, pulse_sync, pulse_export, pulse_checkpoint, spec_* (artifact writes).
- ADT: apply_diff, smart_edit, format.

### Heavy / long-running
- ADT: dependency_graph, context_tree, symbol_search, sg_search/sg_lint, ts_parse/ts_query,
  complexity (large trees), config_flow (multi-analyzer).

## Refactor Targets
- Dispatcher: add per-module concurrency limits and structured error isolation.
- Compass: serialize state writes and staging exports to avoid concurrent mutation.
- ADT: offload heavy analyzer work to background threads.

## Validation Targets
- Mixed tool-call batch (1,000 calls) with 0 sibling-call errors.
- Failure isolation: injected failure does not block unrelated calls.
- Compass compliance: artifacts only under pulse_staging/artifacts.
