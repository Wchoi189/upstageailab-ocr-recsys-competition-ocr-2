# Implementation Plan: MCP Tooling Refactor

**Branch**: `001-mcp-tooling-refactor` | **Date**: 2026-02-12 | **Spec**: /specs/001-mcp-tooling-refactor/spec.md
**Input**: Feature specification from `/specs/001-mcp-tooling-refactor/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Refactor the unified MCP dispatcher to eliminate async sibling tool call errors while preserving existing tool names, error semantics, and Compass artifact compliance. The plan introduces explicit concurrency controls per server, isolates failures to individual calls, and adds structured telemetry to trace call ordering without slowing baseline performance beyond 10%.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11
**Primary Dependencies**: `mcp`, `asyncio`, AgentQMS middleware, `project_compass`, `agent_debug_toolkit`, `starlette`/`uvicorn` (SSE transport)
**Storage**: Filesystem (telemetry JSONL, Compass artifacts)
**Testing**: `pytest`, `asyncio`-driven tests
**Target Platform**: Linux server / local dev (stdio + SSE)
**Project Type**: Monorepo with shared scripts and dev_tools packages
**Performance Goals**: 0 sibling-call errors in 1,000 mixed tool calls; median latency within +10% baseline
**Constraints**: Artifact writes limited to `project_compass/pulse_staging/artifacts/`; no manual artifact creation; preserve tool names and inputs
**Scale/Scope**: Dispatcher-level refactor touching unified server and two tool servers; no client-facing API changes

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- No ratified constitution detected (file contains placeholders). Proceeding with repository constraints: UV-only execution, Compass artifact write restrictions, no manual artifact creation.
- Gate Status: PASS (no enforceable constitution gates present)

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
scripts/
└── mcp/
    └── unified_server.py

dev_tools/
├── project_compass/
│   └── project_compass/
│       └── mcp_server.py
└── agent_debug_toolkit/
    └── agent_debug_toolkit/
        └── mcp_server.py

tests/
└── test_unified_server_artifact_types.py
```

**Structure Decision**: Targeted refactor within existing dispatcher and MCP server modules; no new project roots introduced.

## Complexity Tracking

No constitution violations requiring justification.

## Phase 0: Outline & Research

- Extracted unknowns: concurrency control strategy, error isolation boundaries, audit-friendly ordering logs, and constitution gates.
- Completed research in `/specs/001-mcp-tooling-refactor/research.md` resolving all unknowns with concrete decisions and alternatives.

## Phase 1: Design & Contracts

- Data model defined in `/specs/001-mcp-tooling-refactor/data-model.md` (tool calls, results, dispatch context, compliance events).
- API contracts captured in `/specs/001-mcp-tooling-refactor/contracts/openapi.yaml` for dispatcher-facing endpoints used by MCP transport.
- Quickstart documented in `/specs/001-mcp-tooling-refactor/quickstart.md`.
- Agent context updated via `.specify/scripts/bash/update-agent-context.sh copilot`.
- Constitution re-check: PASS (no constitution gates defined; repository constraints retained).

## Phase 2: Implementation Plan

1. **Dispatcher Concurrency Controls**: Introduce per-server `asyncio.Semaphore` limits and a dispatcher-level task group to avoid sibling-call reentrancy while preserving overlapping calls.
2. **Error Isolation & Reporting**: Wrap each call in a per-request execution context that captures exceptions, records telemetry, and returns structured error payloads without cancelling unrelated calls.
3. **Ordering & Telemetry**: Add monotonic ordering metadata (request id, parent id, sequence index) and emit to telemetry log to support troubleshooting without exposing sensitive data.
4. **Compass Compliance Guards**: Ensure dispatcher pathways never write artifacts directly; all artifact creation remains within `project_compass` tool handlers and is audited for path compliance.
5. **Compatibility Checks**: Keep tool names, input schemas, and responses intact; add regression tests for mixed-tool concurrency and failure isolation.
