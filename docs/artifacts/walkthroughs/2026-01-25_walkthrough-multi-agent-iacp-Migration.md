# Walkthrough - Multi-Agent IACP Migration

## Changes Key
- **State Virtualization**: `effective.yaml` is now virtual-only via [generate_virtual_config](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/config_loader.py#340-384).
- **IACP Protocol**:
    - **Schema**: Enforced strict [IACPEnvelope](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/multi_agent/rabbitmq_transport.py#14-16) via Pydantic.
    - **Transport**: [RabbitMQTransport](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/multi_agent/rabbitmq_transport.py#19-145) now validates all ingress/egress messages.
    - **Execution**: Agents (`OCRInferenceAgent`, [ValidationAgent](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/infrastructure/agents/validation_agent.py#21-349)) updated to handle [IACPEnvelope](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/multi_agent/rabbitmq_transport.py#14-16) objects.
- **Local LLM**:
    - **Client**: [QwenClient](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/infrastructure/agents/llm/qwen_client.py#20-268) updated to support separate Ollama instance.
    - **Endpoint**: `http://host.docker.internal:11434` (Ollama Native / OpenAI Compatible)
    - **Default Model**: `qwen3:4b-instruct`
- **Distributed Caching**:
    - **ConfigLoader**: Now checks Redis (`config:{path}`) before disk. 
    - **Fallback**: Gracefully degrades to memory/disk if Redis is offline.

## Verification Results

### 1. IACP Compliance
Agents now communicate using strictly typed envelopes.

```python
# ValidationAgent handler signature (updated)
def _handle_validate_ocr_result(self, envelope: IACPEnvelope) -> dict[str, Any]:
    payload = envelope.payload
    # ...
```

### 2. Local Inference
[QwenClient](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/infrastructure/agents/llm/qwen_client.py#20-268) now routes to the local RTX 3090 via `host.docker.internal`.

```python
client = QwenClient(model_name="qwen3:4b-instruct")
response = client.generate("Validate this text...")
```

### 3. Cleanup
[janitor.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/bin/janitor.py) successfully moved stale artifacts to `.archive/`.

### 4. Infrastructure Verification
Verified connectivity to core services using [AgentQMS/bin/verify_infra.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/bin/verify_infra.py).
- **Redis**: ✅ Connected (host: redis, port: 6379)
- **RabbitMQ**: ✅ Connected (host: rabbitmq)

### 5. Redis Config Caching
[ConfigLoader](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/config_loader.py#61-436) logic verified via internal test script:
1. First Load: Fetches from disk (IO) → Sets in Redis.
2. Second Load: Fetches from Redis (Cache Hit).

### 6. Integration Testing (IACP Flow)
Verified end-to-end message flow using [tests/integration/run_client_only.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/tests/integration/run_client_only.py):
- **Flow**: Client -> RabbitMQ -> ValidationAgent -> QwenClient -> Ollama -> Response.
- **Payload**: `cmd.detect_errors` with sample text.
- **Result**: [ValidationAgent](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/infrastructure/agents/validation_agent.py#21-349) successfully processed the request and returned LLM-generated errors/suggestions via IACP Envelope.
- **Fixes**: Resolved `ModuleNotFoundError` by correcting `ocr.core` package imports to avoid heavy dependencies (Torch) and fixed relative imports in [llm](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/infrastructure/agents/base_agent.py#304-320) module.
