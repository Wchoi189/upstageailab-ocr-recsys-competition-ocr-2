# Feature Specification: MCP Tooling Refactor

**Feature Branch**: `001-mcp-tooling-refactor`
**Created**: 2026-02-12
**Status**: Draft
**Input**: User description: "Create a concise spec for refactoring MCP tooling to avoid async sibling tool call errors. Focus on unified_server dispatcher plus project_compass and agent_debug_toolkit servers. Include goals, non-goals, constraints (Compass artifact rules, no manual artifacts), and success criteria."

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.

  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Reliable Cross-Server Tool Calls (Priority: P1)

As an operator running MCP tooling, I need tool calls dispatched by the unified_server to complete without async sibling call errors when invoking project_compass and agent_debug_toolkit so that workflows remain reliable.

**Why this priority**: The dispatcher is the shared entry point for critical tooling; failures here block multiple workflows.

**Independent Test**: Trigger mixed tool calls to unified_server, project_compass, and agent_debug_toolkit and verify all complete without sibling call errors while returning their expected results.

**Acceptance Scenarios**:

1. **Given** a mixed sequence of tool calls across the three servers, **When** the dispatcher processes them with overlaps, **Then** each call completes and no async sibling tool call error is raised.
2. **Given** a single tool call to project_compass via the dispatcher, **When** it runs to completion, **Then** the result is returned without delays or errors introduced by sibling calls.

---

### User Story 2 - Compass Artifact Compliance Preserved (Priority: P2)

As a maintainer, I need the refactor to preserve Compass artifact rules so that artifacts remain compliant and audit-friendly.

**Why this priority**: Artifact compliance is a non-negotiable operational requirement.

**Independent Test**: Run Compass-related tool calls and confirm all artifacts are created only in the allowed location with no manual artifact creation.

**Acceptance Scenarios**:

1. **Given** a Compass tool call that produces artifacts, **When** it completes, **Then** all artifacts appear only under pulse_staging/artifacts and none are written elsewhere.

---

### User Story 3 - Failure Isolation (Priority: P3)

As an operator, I need a failed tool call to be isolated so that other tool calls are not blocked or corrupted.

**Why this priority**: Refactors must not reduce reliability or cascade failures across servers.

**Independent Test**: Induce a failure in one server call and confirm that subsequent calls in the dispatcher still succeed.

**Acceptance Scenarios**:

1. **Given** a tool call that fails in agent_debug_toolkit, **When** the dispatcher continues to process new calls, **Then** new calls still execute and return results.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- Concurrent tool calls arrive for project_compass and agent_debug_toolkit while unified_server is already processing another request.
- A tool call times out or returns an error mid-flight while sibling calls are still in progress.
- The dispatcher restarts during a burst of tool calls and must resume without duplicating results.
- A server responds out of order relative to request submission.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: The dispatcher MUST support tool calls to unified_server, project_compass, and agent_debug_toolkit without async sibling tool call errors.
- **FR-002**: The dispatcher MUST allow overlapping tool calls and return each result to the correct caller.
- **FR-003**: The system MUST surface tool call failures without preventing unrelated tool calls from completing.
- **FR-004**: Project Compass tool calls MUST create artifacts only under pulse_staging/artifacts.
- **FR-005**: The system MUST avoid manual artifact creation and rely only on automated outputs.
- **FR-006**: The system MUST preserve existing tool names and inputs so current workflows remain usable.
- **FR-007**: The system MUST provide a human-readable record of tool call ordering and outcomes for troubleshooting.

### Key Entities *(include if feature involves data)*

- **Tool Call**: A request routed by the dispatcher with an identifier, target server, and expected outcome.
- **Tool Result**: The completed outcome of a tool call, including success or failure status.
- **Dispatch Context**: The grouping of concurrent tool calls and their sequencing metadata.
- **Compass Artifact**: A generated output stored under pulse_staging/artifacts.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: 0 async sibling tool call errors occur during 1,000 mixed tool calls across the three servers.
- **SC-002**: Median tool call completion time does not exceed the current baseline by more than 10% under the same workload.
- **SC-003**: 100% of artifacts produced by Compass tool calls are located under pulse_staging/artifacts.
- **SC-004**: At least 99% of tool calls complete successfully even when a sibling call fails.

## Goals

- Eliminate async sibling tool call errors in the dispatcher flows.
- Keep project_compass and agent_debug_toolkit tool behavior intact while improving reliability.
- Preserve compliance with Compass artifact rules and auditing needs.

## Non-Goals

- Introducing new MCP tools or changing existing tool semantics.
- Changing external client workflows or user-facing interfaces.
- Adding new artifact types or manual artifact workflows.

## Constraints

- Compass artifacts must be written only to pulse_staging/artifacts.
- Manual artifact creation is not allowed.
- The refactor must preserve existing tool names and inputs.

## Assumptions

- The current async sibling tool call error is reproducible via mixed tool call workloads.
- Existing monitoring or logs can be used to verify call ordering and outcomes.
