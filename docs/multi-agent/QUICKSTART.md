# Multi-Agent System Quick Start

Get up and running with the multi-agent OCR system in 5 minutes.

## Option 1: Using Docker Compose (Recommended)

### Step 1: Setup Environment

```bash
cd docker/multi-agent
cp .env.example .env
# Edit .env and add your API keys (optional for basic functionality)
```

### Step 2: Start the System

```bash
docker-compose up -d
```

### Step 3: Verify Agents are Running

```bash
docker-compose ps
```

You should see:
- rabbitmq
- agent-preprocessor
- agent-inference
- agent-validator
- agent-orchestrator
- agent-linter
- worker-1
- worker-2

### Step 4: Run Example

```bash
# From project root
python examples/multi_agent/ocr_workflow_example.py
```

### Step 5: Monitor

Open RabbitMQ Management UI: http://localhost:15672
- Username: `admin`
- Password: `admin123`

View agent logs:
```bash
docker-compose logs -f agent-orchestrator
```

### Stop the System

```bash
docker-compose down
```

## Option 2: Local Development Setup

### Step 1: Install Dependencies

```bash
# Install RabbitMQ (or use Docker)
docker run -d -p 5672:5672 -p 15672:15672 \
    -e RABBITMQ_DEFAULT_USER=admin \
    -e RABBITMQ_DEFAULT_PASS=admin123 \
    rabbitmq:3.12-management

# Install Python dependencies
pip install pika opencv-python-headless rembg httpx openai
```

### Step 2: Start Agents

Using the startup script:
```bash
./scripts/multi_agent/start_agents.sh
```

Or manually (in separate terminals):
```bash
# Terminal 1
python -m ocr.agents.ocr_preprocessing_agent

# Terminal 2
python -m ocr.agents.ocr_inference_agent

# Terminal 3
python -m ocr.agents.ocr_validation_agent

# Terminal 4
python -m ocr.agents.orchestrator_agent

# Terminal 5
python -m ocr.workers.ocr_worker
```

### Step 3: Run Example

```bash
python examples/multi_agent/ocr_workflow_example.py
```

### Stop Agents

```bash
./scripts/multi_agent/stop_agents.sh
```

Or press Ctrl+C in each terminal.

## Your First Workflow

### Simple OCR Request

Create a file `my_first_ocr.py`:

```python
from ocr.communication.rabbitmq_transport import RabbitMQTransport
import json

# Connect to RabbitMQ
transport = RabbitMQTransport(
    host="localhost",  # or "rabbitmq" if using Docker
    agent_id="my.client"
)
transport.connect()

# Send OCR request
response = transport.send_command(
    target="agent.orchestrator",
    command="execute_ocr_workflow",
    payload={
        "image_paths": ["/path/to/your/image.jpg"],
        "workflow_config": {
            "stages": ["preprocess", "inference"],
            "preprocess_options": {
                "enhance_contrast": True
            }
        }
    },
    timeout=120
)

# Print result
result = response["payload"]
print(json.dumps(result, indent=2))

transport.close()
```

Run it:
```bash
python my_first_ocr.py
```

## Testing the System

### Health Check

```python
from ocr.communication.rabbitmq_transport import RabbitMQTransport

transport = RabbitMQTransport(host="localhost", agent_id="health.check")
transport.connect()

# Check each agent
agents = [
    "agent.ocr.preprocessor",
    "agent.ocr.inference",
    "agent.ocr.validator",
    "agent.orchestrator"
]

for agent in agents:
    try:
        response = transport.send_command(
            target=agent,
            command="get_status",
            payload={},
            timeout=5
        )
        status = response["payload"]
        print(f"✓ {agent}: {status['status']} ({status['tasks_completed']} tasks)")
    except Exception as e:
        print(f"✗ {agent}: {e}")

transport.close()
```

### Submit Test Job

```python
from ocr.workers.job_queue import JobQueue, JobPriority

queue = JobQueue(queue_name="jobs.ocr", rabbitmq_host="localhost")
queue.connect()

# Submit a test job
job_id = queue.submit_job(
    job_type="ocr.workflow",
    payload={
        "image_paths": ["/path/to/test/image.jpg"],
        "workflow_config": {"stages": ["preprocess", "inference"]}
    },
    priority=JobPriority.NORMAL
)

print(f"Job submitted: {job_id}")

# Check queue stats
stats = queue.get_queue_stats()
print(f"Pending jobs: {stats['pending_jobs']}")

queue.close()
```

## Common Use Cases

### 1. Batch Processing

```python
from ocr.workers.job_queue import JobQueue, JobPriority

queue = JobQueue("jobs.ocr", rabbitmq_host="localhost")
queue.connect()

job_id = queue.submit_job(
    job_type="ocr.batch",
    payload={
        "image_dir": "/data/receipts",
        "output_dir": "/data/output",
        "batch_size": 10
    },
    priority=JobPriority.HIGH
)

print(f"Batch job: {job_id}")
queue.close()
```

### 2. LLM Validation

```python
transport = RabbitMQTransport(host="localhost", agent_id="validator.test")
transport.connect()

response = transport.send_command(
    target="agent.ocr.validator",
    command="detect_errors",
    payload={
        "text": "Tota1 Amount: $99.OO",  # OCR errors: 1 vs l, OO vs 00
        "context": "Receipt total field"
    },
    timeout=30
)

result = response["payload"]
print(f"Errors found: {result['errors_found']}")
print(f"Corrected: {result.get('corrected_text')}")

transport.close()
```

### 3. Workflow Planning

```python
transport = RabbitMQTransport(host="localhost", agent_id="planner.test")
transport.connect()

response = transport.send_command(
    target="agent.orchestrator",
    command="plan_workflow",
    payload={
        "requirements": "Process 100 receipts with high accuracy validation",
        "constraints": {"max_time_hours": 1}
    },
    timeout=20
)

plan = response["payload"]["workflow_plan"]
print(f"Recommended stages: {plan['stages']}")
print(f"Reasoning: {plan['reasoning']}")

transport.close()
```

## Troubleshooting

### RabbitMQ Connection Failed

```bash
# Check if RabbitMQ is running
docker ps | grep rabbitmq

# Or check port
nc -zv localhost 5672

# Start RabbitMQ if not running
docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3.12-management
```

### Agent Not Responding

```bash
# Check agent logs
docker-compose logs agent-orchestrator

# Or local logs
tail -f logs/agents/orchestrator.log

# Test agent connection
python -c "
from ocr.communication.rabbitmq_transport import RabbitMQTransport
t = RabbitMQTransport(host='localhost', agent_id='test')
t.connect()
print('✓ Connected to RabbitMQ')
t.close()
"
```

### Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/path/to/project

# Or install in development mode
pip install -e .
```

## Next Steps

- Read the [Full Documentation](README.md)
- Check out [Developer Guide](DEVELOPER_GUIDE.md)
- Explore more [Examples](../../examples/multi_agent/)
- View the [Architecture Diagram](README.md#architecture)

## Getting Help

- Check RabbitMQ Management UI: http://localhost:15672
- View agent logs in `logs/agents/` directory
- Review queue statistics for bottlenecks
- Check the [Troubleshooting Guide](README.md#troubleshooting)
