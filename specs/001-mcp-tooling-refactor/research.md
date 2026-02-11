# Research

## Concurrency Controls

Decision: Use per-server `asyncio.Semaphore` limits plus a dispatcher-level task group to run calls concurrently without sibling reentrancy.
Rationale: Bounded concurrency prevents event-loop overload while still allowing overlaps across servers; task groups simplify lifecycle management for multiple concurrent calls.
Alternatives considered: A single global lock (too restrictive); unbounded `asyncio.gather` (risk of overload and sibling-call errors).

## Error Isolation

Decision: Wrap each tool execution in its own try/except boundary and return structured error payloads without cancelling unrelated tasks.
Rationale: Ensures failure isolation and preserves high success rate under mixed workloads.
Alternatives considered: Fail-fast cancellation of sibling tasks (violates isolation requirement); shared exception handler (opaque error attribution).

## Ordering and Telemetry

Decision: Add monotonic request identifiers and sequence indexes to telemetry events to reconstruct call ordering across servers.
Rationale: Troubleshooting requires deterministic ordering without leaking sensitive data or coupling to transport.
Alternatives considered: Reliance on timestamp-only ordering (non-deterministic under concurrency); verbose payload logging (privacy and size risks).

## Compliance and Artifact Rules

Decision: Enforce a dispatcher-level guard that never writes artifacts directly; all artifact writes remain inside `project_compass` tool handlers and must target `project_compass/pulse_staging/artifacts/`.
Rationale: Compass rules require a single write location and forbid manual artifacts.
Alternatives considered: Dispatcher-level artifact creation (violates Compass constraints); client-side artifact writes (breaks auditability).

## Constitution Gates

Decision: Proceed with repository constraints only, because the constitution file is a template with no enforceable gates.
Rationale: There are no ratified principles to enforce; existing operational constraints remain mandatory.
Alternatives considered: Blocking until a constitution is ratified (not required by the feature spec).
