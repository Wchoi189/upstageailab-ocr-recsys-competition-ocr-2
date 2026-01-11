---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: "None"
title: "Agent Process Feedback System"
date: "2026-01-11 16:30 (KST)"
branch: "main"
description: "Plan to implement the log_process_feedback tool."
---

# Implementation Plan: Agent Process Feedback System

## Goal
Implement a "Metacognitive Feedback Loop" (Agent-In-the-Loop) by adding a `log_process_feedback` tool to the Unified MCP Server. This tool allows agents to report friction, redundancy, or cognitive overload without derailing the current task.

## User Review Required
> [!NOTE]
> **Derailment Protection**: The tool is designed to be "fire-and-forget". It logs the issue to a file and returns a confirmation to the agent, allowing the agent to feel heard and proceed immediately without attempting to fix the system architecture mid-task.

## Proposed Changes

### 1. Tool Configuration

#### [MODIFY] [tools.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/config/tools.yaml)
- Add `log_process_feedback` definition.
    - **Inputs**: `severity`, `category`, `observation`, `suggestion`.
    - **Description**: Emphasize that this is for *logging* issues to be addressed later, not for immediate interactive debugging.

### 2. Server Logic

#### [MODIFY] [unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py)
- Implement `log_process_feedback` handler.
- **Dual Output Mechanism**:
    1.  **File Output**: Append the feedback event to `project_compass/feedback_log.md` (creating it if it doesn't exist).
        - Format: Timestamp, Agent Name (if available), Severity, Category, Observation.
    2.  **Chat Output**: Return a concise confirmation message to the agent (e.g., "Feedback logged. Proceeding with task."). This satisfies the agent's need for strict output while keeping the chat clean.

### 3. Future Work (Not in Scope)
- **Mechanical Artifact Sync**: As discussed, a future script could auto-convert internal artifacts (Antigravity/Cursor specific) into AgentQMS standard templates. This requires cross-platform research.

## Verification Plan

### Manual Verification
1.  **Trigger**: Manually invoke the `log_process_feedback` tool via a test script or MCP inspector.
2.  **Verify File**: Check that `project_compass/feedback_log.md` contains the entry.
3.  **Verify Output**: Check that the tool returns the expected confirmation text.
