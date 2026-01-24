# Walkthrough - Multi-Agent IACP Migration

## Changes Key
- **State Virtualization**: `effective.yaml` is now virtual-only/on-demand via [generate_virtual_config](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/config_loader.py#288-333).
- **IACP Validation**: Implemented strict Pydantic schemas for RabbitMQ transport.
- **Maintenance**: Added [janitor.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/bin/janitor.py) for automated cleanup.

## Verification Results

### 1. Unified CLI Compliance
The `aqms` CLI (via [AgentQMS/cli.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/cli.py)) now supports virtual configuration generation suitable for AI ingestion.

```bash
uv run AgentQMS/cli.py generate-config --json --path ocr/inference
```

**Result**:
```json
{
  "metadata": {
    "session_id": "local",
    "path": "ocr/inference",
    "virtual": true
  },
  "resolved": { ... }
}
```

### 2. IACP Transport Layer
Created [ocr/core/infrastructure/communication/iacp_schemas.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/infrastructure/communication/iacp_schemas.py) and [AgentQMS/tools/multi_agent/rabbitmq_transport.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/multi_agent/rabbitmq_transport.py).

**Schema Validation**:
- Enforced [IACPEnvelope](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/multi_agent/rabbitmq_transport.py#14-16) model.
- Prevents loose dictionary passing.

### 3. Automated Cleanup (Janitor)
The [janitor.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/bin/janitor.py) script identifies stale artifacts to reduce token noise.

```bash
uv run AgentQMS/bin/janitor.py --dry-run
```

**Status**: Verified dry run execution.
