# Multi-Agent Collaboration Environment for OCR

A professional-grade multi-agent system for OCR processing using RabbitMQ, AutoGen patterns, and LLM-powered intelligence.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Components](#components)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Overview

This multi-agent collaboration environment provides a scalable, distributed system for OCR processing with the following features:

- **Specialized Agents**: Preprocessing, Inference, Validation, and Orchestration
- **LLM Integration**: QwenCLI, Grok4, and OpenAI support
- **Job Queue System**: RabbitMQ-based task distribution with priorities and retries
- **Background Workers**: Scalable worker processes for async job processing
- **IACP Protocol**: Inter-Agent Communication Protocol for standardized messaging

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Application                       │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                  RabbitMQ Message Broker                     │
│                  (IACP Topic Exchange)                       │
└───┬────────┬────────┬────────┬────────┬────────┬───────────┘
    │        │        │        │        │        │
    ▼        ▼        ▼        ▼        ▼        ▼
┌────────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐
│Preproc.│ │Infer.│ │Valid.│ │Orch. │ │Lint. │ │Workers   │
│Agent   │ │Agent │ │Agent │ │Agent │ │Agent │ │(1, 2...)│
└────────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────────┘
    │                   │        │
    │                   ▼        ▼
    │              ┌──────────────────┐
    └─────────────►│  LLM Providers   │
                   │ (Qwen/Grok/GPT)  │
                   └──────────────────┘
```

### Key Components

1. **Base Agent Framework** (`ocr/agents/base_agent.py`)
   - `BaseAgent`: Foundation for all agents
   - `LLMAgent`: Base class for LLM-powered agents
   - IACP message routing and handling

2. **Specialized OCR Agents**
   - **Preprocessing Agent**: Image normalization, enhancement, background removal
   - **Inference Agent**: Text detection and recognition
   - **Validation Agent**: Quality assurance with LLM-powered validation
   - **Orchestrator Agent**: Workflow coordination and task delegation

3. **LLM Clients** (`ocr/agents/llm/`)
   - QwenClient: Local Qwen model integration
   - Grok4Client: xAI Grok API integration
   - OpenAIClient: OpenAI GPT integration

4. **Job Queue System** (`ocr/workers/`)
   - JobQueue: RabbitMQ-based job queuing with priorities
   - OCRWorker: Background worker for async processing

5. **Communication Layer** (`ocr/communication/`)
   - RabbitMQTransport: IACP transport implementation

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- RabbitMQ (or use Docker Compose)
- Optional: GPU for OCR inference
- Optional: API keys for Grok4 or OpenAI

### Installation

1. **Clone the repository**:
```bash
git clone <repo-url>
cd upstageailab-ocr-recsys-competition-ocr-2
```

2. **Set up environment variables**:
```bash
cp docker/multi-agent/.env.example docker/multi-agent/.env
# Edit .env and add your API keys
```

3. **Start the multi-agent system**:
```bash
cd docker/multi-agent
docker-compose up -d
```

4. **Verify agents are running**:
```bash
docker-compose ps
docker-compose logs -f agent-orchestrator
```

### Running Without Docker

1. **Start RabbitMQ**:
```bash
docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3.12-management
```

2. **Install dependencies**:
```bash
pip install pika opencv-python-headless rembg httpx openai
```

3. **Start agents** (in separate terminals):
```bash
# Terminal 1: Preprocessing Agent
python -m ocr.agents.ocr_preprocessing_agent

# Terminal 2: Inference Agent
python -m ocr.agents.ocr_inference_agent

# Terminal 3: Validation Agent
python -m ocr.agents.ocr_validation_agent

# Terminal 4: Orchestrator Agent
python -m ocr.agents.orchestrator_agent

# Terminal 5: Background Worker
python -m ocr.workers.ocr_worker
```

## Usage Examples

### Example 1: Simple Preprocessing

```python
from ocr.communication.rabbitmq_transport import RabbitMQTransport

transport = RabbitMQTransport(host="rabbitmq", agent_id="client.app")
transport.connect()

response = transport.send_command(
    target="agent.ocr.preprocessor",
    command="normalize_image",
    payload={
        "image_path": "/data/receipt.jpg",
        "options": {
            "enhance_contrast": True,
            "denoise": True
        }
    },
    timeout=30
)

print(response["payload"])
transport.close()
```

### Example 2: Full OCR Workflow

```python
from ocr.communication.rabbitmq_transport import RabbitMQTransport

transport = RabbitMQTransport(host="rabbitmq", agent_id="client.app")
transport.connect()

response = transport.send_command(
    target="agent.orchestrator",
    command="execute_ocr_workflow",
    payload={
        "image_paths": ["/data/receipt1.jpg", "/data/receipt2.jpg"],
        "workflow_config": {
            "stages": ["preprocess", "inference", "validation"],
            "validation_options": {
                "use_llm": True,
                "min_confidence": 0.7
            }
        },
        "output_dir": "/data/output"
    },
    timeout=300
)

print(response["payload"]["summary"])
transport.close()
```

### Example 3: Batch Processing with Job Queue

```python
from ocr.workers.job_queue import JobQueue, JobPriority

queue = JobQueue(queue_name="jobs.ocr", rabbitmq_host="rabbitmq")
queue.connect()

job_id = queue.submit_job(
    job_type="ocr.batch",
    payload={
        "image_dir": "/data/receipts",
        "output_dir": "/data/output",
        "batch_size": 10
    },
    priority=JobPriority.HIGH,
    max_retries=2
)

print(f"Job submitted: {job_id}")
queue.close()
```

### Example 4: LLM-Powered Validation

```python
from ocr.communication.rabbitmq_transport import RabbitMQTransport

transport = RabbitMQTransport(host="rabbitmq", agent_id="client.app")
transport.connect()

ocr_result = {
    "results": [
        {"text": "Total: $45.99", "confidence": 0.88, "bbox": [10, 100, 200, 120]}
    ],
    "full_text": "Total: $45.99"
}

response = transport.send_command(
    target="agent.ocr.validator",
    command="validate_ocr_result",
    payload={
        "ocr_result": ocr_result,
        "validation_rules": {"min_confidence": 0.7},
        "use_llm": True
    },
    timeout=60
)

print(response["payload"])
transport.close()
```

## Configuration

### Environment Variables

```bash
# RabbitMQ
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672

# LLM APIs
XAI_API_KEY=your_grok_api_key
OPENAI_API_KEY=your_openai_api_key
QWEN_API_ENDPOINT=http://localhost:8000

# Worker Settings
WORKER_PREFETCH_COUNT=1
WORKER_LOG_LEVEL=INFO
```

### Agent Configuration

Agents can be configured via environment variables or constructor parameters:

```python
from ocr.agents.ocr_validation_agent import OCRValidationAgent

agent = OCRValidationAgent(
    agent_id="agent.ocr.validator.1",
    rabbitmq_host="rabbitmq",
    llm_provider="qwen"  # or "grok4", "openai"
)
agent.start()
```

### Workflow Configuration

```python
workflow_config = {
    "stages": ["preprocess", "inference", "validation"],
    "preprocess_options": {
        "resize": True,
        "target_size": [1024, 1024],
        "enhance_contrast": True,
        "denoise": False
    },
    "inference_options": {
        "detection_model": "db_resnet50",
        "recognition_model": "crnn_vgg",
        "confidence_threshold": 0.5
    },
    "validation_options": {
        "use_llm": True,
        "min_confidence": 0.7,
        "min_quality_score": 0.8
    }
}
```

## Development

### Adding a New Agent

1. **Create agent class**:
```python
from ocr.agents.base_agent import BaseAgent, AgentCapability

class MyCustomAgent(BaseAgent):
    def __init__(self, agent_id="agent.custom", rabbitmq_host="rabbitmq"):
        capabilities = [
            AgentCapability(
                name="my_capability",
                description="Description of what it does",
                input_schema={"param": "type"},
                output_schema={"result": "type"}
            )
        ]

        super().__init__(agent_id, "custom", rabbitmq_host, capabilities)
        self.register_handler("cmd.my_command", self._handle_my_command)

    def get_binding_keys(self):
        return ["cmd.my_command.#"]

    def _handle_my_command(self, envelope):
        payload = envelope.get("payload", {})
        # Process command
        return {"status": "success", "result": "data"}

if __name__ == "__main__":
    agent = MyCustomAgent()
    agent.start()
```

2. **Add to Docker Compose**:
```yaml
my-agent:
  build:
    context: ../..
    dockerfile: docker/multi-agent/Dockerfile.agent
  container_name: my-custom-agent
  environment:
    - RABBITMQ_HOST=rabbitmq
  command: python -m ocr.agents.my_custom_agent
  networks:
    - ocr-network
```

### Testing Agents

Use the example script:
```bash
python examples/multi_agent/ocr_workflow_example.py
```

Or test individual agents:
```bash
# Test preprocessing
python scripts/prototypes/multi_agent/test_preprocessing.py

# Test full workflow
python scripts/prototypes/multi_agent/test_workflow.py
```

## Monitoring

### RabbitMQ Management UI

Access at `http://localhost:15672` (default credentials: admin/admin123)

- View queues and message rates
- Monitor agent connections
- Inspect dead letter queues

### Agent Health Checks

```python
from ocr.communication.rabbitmq_transport import RabbitMQTransport

transport = RabbitMQTransport(host="rabbitmq", agent_id="monitor")
transport.connect()

response = transport.send_command(
    target="agent.ocr.preprocessor",
    command="get_status",
    payload={},
    timeout=5
)

print(response["payload"])  # Agent status and metrics
```

### Queue Statistics

```python
from ocr.workers.job_queue import JobQueue

queue = JobQueue(queue_name="jobs.ocr", rabbitmq_host="rabbitmq")
queue.connect()

stats = queue.get_queue_stats()
print(stats)
# {'queue_name': 'jobs.ocr', 'pending_jobs': 5, 'active_jobs': 2, ...}
```

## Troubleshooting

### Common Issues

**1. Agent not receiving messages**

- Check RabbitMQ is running: `docker ps | grep rabbitmq`
- Verify binding keys match routing keys
- Check RabbitMQ logs: `docker logs ocr-rabbitmq`

**2. LLM API errors**

- Verify API keys are set correctly
- Check network connectivity to API endpoints
- Review agent logs for specific error messages

**3. Job queue errors**

- Check dead letter queue: RabbitMQ UI → Queues → `jobs.ocr.dlq`
- Review worker logs: `docker logs ocr-worker-1`
- Verify job payload schema is correct

**4. High latency**

- Increase worker count in docker-compose.yml
- Adjust prefetch_count for workers
- Monitor RabbitMQ message rates

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

View agent logs:
```bash
docker-compose logs -f agent-orchestrator
docker-compose logs -f worker-1
```

## Performance Tuning

### Scaling Workers

Add more workers in docker-compose.yml:
```yaml
worker-3:
  build:
    context: ../..
    dockerfile: docker/multi-agent/Dockerfile.agent
  environment:
    - WORKER_ID=worker.ocr.3
  command: python -m ocr.workers.ocr_worker
```

### Optimizing RabbitMQ

```bash
# In .env file
RABBITMQ_VM_MEMORY_HIGH_WATERMARK=0.6
RABBITMQ_CHANNEL_MAX=2048
```

### GPU Utilization

For inference agent with GPU:
```yaml
agent-inference:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## References

- [IACP Protocol Specification](../../project_compass/design/inter_agent_communication_protocol.md)
- [AutoGen vs CrewAI Research](../../project_compass/design/research_crewai_vs_autogen.md)
- [Multi-Agent Roadmap](../../project_compass/roadmap/00_multi_agent_infrastructure.yaml)
- [RabbitMQ Documentation](https://www.rabbitmq.com/documentation.html)

## Contributing

When contributing to the multi-agent system:

1. Follow the IACP protocol specification
2. Add comprehensive docstrings to new agents
3. Include example usage in documentation
4. Add tests for new capabilities
5. Update this README with new features

## License

See project LICENSE file.
