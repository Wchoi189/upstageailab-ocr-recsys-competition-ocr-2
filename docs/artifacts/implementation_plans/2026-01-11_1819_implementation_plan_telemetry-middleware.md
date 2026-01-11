---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: "None"
title: "Telemetry Middleware Implementation Plan"
date: "2026-01-11 18:19 (KST)"
branch: "main"
description: "None"
---

# Telemetry Middleware & Contextual Gating System

## Goal Description
Implement a **Telemetry-driven Feedback Loop** (Middleware Layer) to resolve "Architectural Redundancy" between the custom **AgentQMS** (explicit tool calls) and the **Antigravity Managed Service** (opaque, internal protocol).

The system will act as a **Contextual Gatekeeper**, inspecting agent intent before execution to prevent:
1.  **Double-Taxing**: Redundant artifact creation.
2.  **State Mismatch**: Conflicts between workspace state and `.gemini/` shadow state.
3.  **Policy Violations**: Usage of forbidden patterns (e.g., bare `python`, `sys.path`).

## User Review Required
> [!IMPORTANT]
> This introduces a "Circuit Breaker" pattern. Tool calls may be rejected programmatically with feedback instructions, requiring the Agent to self-correct.

## Proposed Architecture

### 1. The Middleware Pipeline (`TelemetryPipeline`)
A centralized pipeline that intercepts tool calls in `unified_server.py`.

```python
class TelemetryPipeline:
    def validate(self, tool_name: str, arguments: dict) -> Optional[Feedback]:
        # Runs a chain of Interceptors
        pass
```

### 2. Interceptors (The Policies)

#### A. `RedundancyInterceptor` (The Observer)
- **Goal**: Detect if AgentQMS is duplicating work already handled by the Managed Service.
- **Logic**:
    - Triggers on `create_artifact` calls.
    - Checks the **Shadow State** (`.gemini/` directory) for corresponding artifacts (e.g., if `implementation_plan` exists in `.gemini/`, reject the tool call).
    - **Metric**: Contextual Overlap or File Existence.

#### B. `ComplianceInterceptor` (The Guardrail)
- **Goal**: Enforce coding standards at runtime by inspecting code arguments in tool calls.
- **Policies**:
    1.  **Python Execution Policy**:
        -   **Trigger**: usage of bare `python` command in `run_command` or similar tools.
        -   **Feedback**: "Internal Violation: Plain 'python' used. You MUST use 'uv run python' for environment consistency."
    2.  **Path Utility Policy**:
        -   **Trigger**: usage of `sys.path`, `os.path.join`, or excessive `.parent` chaining in code generation.
        -   **Feedback**: "PROTOCOL ERROR: Do not use sys.path or os.path for project paths. Use 'AgentQMS.tools.utils.paths' instead to ensure portability."

### 3. Feedback Injection Strategy
- **Mechanism**: Instead of executing the tool, the MCP server returns a "Soft Failure" response (TextContent).
- **Message Format**: "NOTICE: [Policy Violation]. [Correction Instruction]."
    - Example: *"NOTICE: The 'implementation_plan' is already managed by the internal Antigravity service in .gemini/. Please reference the managed version."*

## Proposed Changes

### Core Framework
#### [NEW] [AgentQMS/middleware/telemetry.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/middleware/telemetry.py)
- Defines `TelemetryPipeline`, `Interceptor` abstract base class, and concrete implementations (`RedundancyInterceptor`, `ComplianceInterceptor`).

#### [NEW] [AgentQMS/middleware/policies.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/middleware/policies.py)
- Specific policy logic (regex patterns, shadow directory resolution).

### Integration points
#### [MODIFY] [scripts/mcp/unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py)
- Inject `TelemetryPipeline` into the `call_tool` handler.
- Catch `PolicyViolation` exceptions and return formatted feedback to the Agent.

## Verification Plan

### Automated Tests
- **Unit Tests**: Test `TelemetryPipeline` with mock tool calls and mock `.gemini/` state.
- **Integration Tests**: Extend `verify_server.py` to trigger a redundancy policy violation and assert the returned feedback message.

### Manual Verification
- Attempt to create a redundant implementation plan and verify the system rejects it with the correct "Circuit Breaker" message.
