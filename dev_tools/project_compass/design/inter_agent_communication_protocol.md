# Inter-Agent Communication Protocol (IACP) v0.1

**Status:** Draft
**Target Framework:** AutoGen + RabbitMQ

## Overview
The IACP defines a strict message exchange standard for autonomous agents in the "Multi-Agent Collaboration Environment". It ensures that all agents, regardless of their internal implementation (LLM-based, deterministic, or hybrid), can communicate reliably using a common schema.

## 1. Message Envelope
All messages **MUST** be wrapped in a standard JSON envelope. This allows the routing layer (Service Bus / RabbitMQ) to manage delivery without parsing the payload.

```json
{
  "id": "uuid-v4",
  "version": "1.0",
  "metadata": {
    "source": "string (agent_id)",
    "target": "string (agent_id | topic)",
    "correlation_id": "uuid-v4 (tracks conversation thread)",
    "timestamp": "iso8601-utc",
    "priority": "int (1-10, default 5)",
    "ttl": "int (seconds, optional)"
  },
  "type": "string (command | event | query | response | error)",
  "payload": { ... }
}
```

## 2. Message Types

| Type         | Semantics                           | Expects Response?     | Example                |
| :----------- | :---------------------------------- | :-------------------- | :--------------------- |
| **command**  | "Do this work." Imperative.         | Yes (Result or Error) | `cmd.lint_code`        |
| **event**    | "Something happened." Notification. | No (Fire & forget)    | `evt.file_changed`     |
| **query**    | "Tell me info." Side-effect free.   | Yes (Result)          | `qry.get_agent_status` |
| **response** | Answer to a command/query.          | No                    | `res.lint_result`      |
| **error**    | Failure signal.                     | No                    | `err.timeout`          |

## 3. Payload Schemas
Payloads **MUST** validate against a Pydantic model registered in the `AgentQMS/standards/schemas/iacp/` registry.

### Example: `cmd.lint_code`
```json
{
  "files": ["path/to/file.py"],
  "linter": "ruff",
  "fix": true
}
```

### Example: `res.lint_result`
```json
{
  "status": "success",
  "violations": [],
  "fixed_files": ["path/to/file.py"]
}
```

## 4. Operational Semantics

### 4.1. Handover
When Agent A tasks Agent B:
1.  Agent A sends `cmd.execute_task` to `queue:agent_b`.
2.  Agent B ACKs the message.
3.  Agent B processes.
4.  Agent B sends `res.task_result` to `queue:agent_a` (or `reply_to` queue).

### 4.2. Error Handling
*   **Retryable**: Application errors (e.g., "LLM overloaded") -> NACK + Requeue (with exponential backoff).
*   **Fatal**: Schema violations, 4xx logic errors -> PUBLISH `err.fatal` -> Dead Letter Queue (DLQ).

## 5. Transport Binding (RabbitMQ)
*   **Exchange**: `iacp.topic` (Topic Exchange)
*   **Routing Keys**: `{type}.{source}.{target}` (e.g., `cmd.planner.worker_1`)
*   **Queues**: One queue per Agent ID (e.g., `q.agent.worker_1`).
