# Data Model

## Entities

### ToolCall
- Fields: `id` (uuid), `name` (string), `arguments` (object), `target_server` (string), `requested_at` (timestamp), `session_id` (string), `sequence` (int)
- Validation: `name` must match known tool names; `arguments` must satisfy the tool input schema.

### ToolResult
- Fields: `call_id` (uuid), `status` (enum: success|error|policy_violation), `output` (list), `error_message` (string, optional), `duration_ms` (number), `completed_at` (timestamp)
- Validation: `status` required; `duration_ms` >= 0.

### DispatchContext
- Fields: `context_id` (uuid), `session_id` (string), `calls` (list of ToolCall), `concurrency_group` (string), `started_at` (timestamp)
- Validation: `calls` must be non-empty for active contexts.

### ConcurrencyGroup
- Fields: `name` (string), `limit` (int), `active` (int)
- Validation: `limit` >= 1; `active` <= `limit`.

### ComplianceEvent
- Fields: `call_id` (uuid), `rule` (string), `status` (enum: pass|fail), `details` (string), `recorded_at` (timestamp)
- Validation: `rule` must reference a known compliance rule.

## Relationships
- `DispatchContext` has many `ToolCall` entries.
- `ToolCall` has one `ToolResult`.
- `ToolCall` can emit many `ComplianceEvent` entries.

## State Transitions
- ToolCall: `queued` -> `running` -> (`success` | `error` | `policy_violation`)
- DispatchContext: `active` -> `completed`
