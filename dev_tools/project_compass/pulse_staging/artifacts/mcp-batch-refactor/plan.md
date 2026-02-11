# MCP Batch Refactor Plan

Date: 2026-02-12
Owner: Copilot
Scope: Unified MCP dispatcher, Project Compass MCP server, Agent Debug Toolkit MCP server.

## Goals
- Eliminate async sibling tool call errors under mixed tool-call loads.
- Preserve tool names, inputs, and existing client workflows.
- Preserve Compass artifact rules and auditability.

## Non-Goals
- No new MCP tools or client-facing API changes.
- No changes to artifact types or external workflows.

## Audit Candidates and Resolutions
- Unified dispatcher concurrency: add per-server concurrency limits and task-group isolation.
- Failure isolation: wrap each call in a dedicated error boundary and return structured errors.
- Ordering telemetry: add request ids and sequence indexes to telemetry for deterministic ordering.
- Compass state mutations: serialize write operations to vessel_state.json and staging export paths.
- ADT heavy analyzers: run expensive analyzers off the event loop (thread pool).

## Phased Plan
1) Survey and baseline
   - Catalog all tool handlers and classify by read vs write and expected runtime.
   - Capture baseline tool latency and current error rate.
2) Dispatcher design
   - Define concurrency policy map (per tool or per module).
   - Define structured error payload schema.
3) Implementation
   - Add dispatcher-level concurrency controls and error isolation.
   - Serialize Compass state writes.
   - Run ADT analyzers in background threads.
4) Validation
   - Mixed-tool concurrent calls: 1,000-call batch with zero sibling errors.
   - Failure isolation: one tool failure does not block others.
   - Compass compliance: all artifacts under pulse_staging/artifacts/mcp-batch-refactor.
   - Performance: median latency within +10% baseline.
5) Rollout
   - Enable telemetry ordering fields.
   - Monitor for regressions under real workflows.

## Deliverables
- Updated dispatcher with concurrency and error isolation controls.
- Compass write-path serialization guard.
- ADT analyzer offload to background threads.
- Test coverage for mixed-call concurrency and failure isolation.
