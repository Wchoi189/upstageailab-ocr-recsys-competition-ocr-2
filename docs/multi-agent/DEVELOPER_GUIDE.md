# Multi-Agent System Developer Guide

## Architecture Deep Dive

### IACP (Inter-Agent Communication Protocol)

All agents communicate using a standardized JSON message envelope:

```json
{
  "id": "uuid-v4",
  "version": "1.0",
  "metadata": {
    "source": "agent.sender",
    "target": "agent.receiver",
    "correlation_id": "uuid-v4",
    "timestamp": "iso8601-utc",
    "priority": 5,
    "ttl": 300
  },
  "type": "cmd.action_name",
  "payload": { "key": "value" }
}
```

### Message Types

- **command**: Imperative actions requiring responses (e.g., `cmd.process_image`)
- **event**: Fire-and-forget notifications (e.g., `evt.job_completed`)
- **query**: Information requests (e.g., `qry.get_status`)
- **response**: Replies to commands/queries (e.g., `res.result`)
- **error**: Failure notifications (e.g., `err.timeout`)

### Routing Keys

Format: `{type}.{source}.{target}`

Examples:
- `cmd.normalize_image.client.agent.ocr.preprocessor`
- `evt.file_changed.watcher.all`
- `qry.health_check.monitor.agent.ocr.inference`

## Agent Lifecycle

### 1. Initialization

```python
class MyAgent(BaseAgent):
    def __init__(self, agent_id, rabbitmq_host):
        # Define capabilities
        capabilities = [...]

        # Call parent constructor
        super().__init__(agent_id, "my_agent_type", rabbitmq_host, capabilities)

        # Register custom handlers
        self.register_handler("cmd.my_action", self._handle_my_action)
```

### 2. Connection

```python
agent = MyAgent()
# Transport connects to RabbitMQ
# Declares exchanges and queues
# Sets up bindings
```

### 3. Message Processing

```python
def _handle_my_action(self, envelope):
    payload = envelope.get("payload", {})

    # Validate input
    if not payload.get("required_field"):
        return {"status": "error", "message": "Missing field"}

    # Process
    result = self._do_work(payload)

    # Return response
    return {"status": "success", "result": result}
```

### 4. Shutdown

```python
def stop(self):
    # Cleanup resources
    self.transport.close()
    logger.info("Agent stopped")
```

## Creating Custom Agents

### Stateless Agent Example

```python
from ocr.agents.base_agent import BaseAgent, AgentCapability

class DataTransformAgent(BaseAgent):
    """Transforms data formats."""

    def __init__(self, agent_id="agent.transform"):
        capabilities = [
            AgentCapability(
                name="json_to_xml",
                description="Convert JSON to XML",
                input_schema={"data": "dict"},
                output_schema={"xml": "str"}
            )
        ]

        super().__init__(agent_id, "transformer", "rabbitmq", capabilities)
        self.register_handler("cmd.json_to_xml", self._handle_transform)

    def get_binding_keys(self):
        return ["cmd.json_to_xml.#"]

    def _handle_transform(self, envelope):
        import dicttoxml
        data = envelope["payload"]["data"]
        xml = dicttoxml.dicttoxml(data)
        return {"status": "success", "xml": xml.decode()}
```

### Stateful Agent Example

```python
from ocr.agents.base_agent import BaseAgent

class CacheAgent(BaseAgent):
    """Caches results with TTL."""

    def __init__(self, agent_id="agent.cache"):
        super().__init__(agent_id, "cache", "rabbitmq")

        # Internal state
        self._cache = {}
        self._timestamps = {}

        self.register_handler("cmd.cache_set", self._handle_set)
        self.register_handler("cmd.cache_get", self._handle_get)

    def get_binding_keys(self):
        return ["cmd.cache_set.#", "cmd.cache_get.#"]

    def _handle_set(self, envelope):
        payload = envelope["payload"]
        key = payload["key"]
        value = payload["value"]
        ttl = payload.get("ttl", 300)

        self._cache[key] = value
        self._timestamps[key] = time.time() + ttl

        return {"status": "success"}

    def _handle_get(self, envelope):
        key = envelope["payload"]["key"]

        if key in self._cache:
            if time.time() < self._timestamps[key]:
                return {"status": "success", "value": self._cache[key]}
            else:
                # Expired
                del self._cache[key]
                del self._timestamps[key]

        return {"status": "error", "message": "Key not found"}
```

### LLM-Powered Agent Example

```python
from ocr.agents.base_agent import LLMAgent, AgentCapability

class SummarizationAgent(LLMAgent):
    """Summarizes text using LLM."""

    def __init__(self, agent_id="agent.summarize"):
        capabilities = [
            AgentCapability(
                name="summarize_text",
                description="Generate text summary",
                input_schema={"text": "str", "max_length": "int"},
                output_schema={"summary": "str"}
            )
        ]

        super().__init__(
            agent_id=agent_id,
            agent_type="summarizer",
            llm_provider="grok4",  # or "qwen", "openai"
            capabilities=capabilities
        )

        self.register_handler("cmd.summarize_text", self._handle_summarize)

    def get_binding_keys(self):
        return ["cmd.summarize_text.#"]

    def _handle_summarize(self, envelope):
        payload = envelope["payload"]
        text = payload["text"]
        max_length = payload.get("max_length", 100)

        prompt = f"Summarize the following text in {max_length} words or less:\n\n{text}"

        summary = self.generate_response(
            prompt=prompt,
            system_prompt="You are a professional text summarizer.",
            temperature=0.3,
            max_tokens=max_length * 2
        )

        return {"status": "success", "summary": summary}
```

## Job Queue Patterns

### Simple Job Submission

```python
from ocr.workers.job_queue import JobQueue, JobPriority

queue = JobQueue("jobs.ocr")
queue.connect()

job_id = queue.submit_job(
    job_type="ocr.inference",
    payload={"image_path": "/data/img.jpg"},
    priority=JobPriority.NORMAL
)
```

### Custom Job Handler

```python
def process_custom_job(job):
    """Custom job processing logic."""
    payload = job.payload

    # Do work
    result = process_data(payload)

    return {"status": "success", "result": result}

# Register handler
queue.register_handler("custom.job", process_custom_job)

# Start worker
queue.start_worker()
```

### Job with Retries

```python
job_id = queue.submit_job(
    job_type="ocr.batch",
    payload={"batch": [...]},
    priority=JobPriority.HIGH,
    max_retries=3,  # Retry up to 3 times
    timeout=600     # 10 minute timeout
)
```

## Error Handling

### Agent-Level Error Handling

```python
def _handle_risky_operation(self, envelope):
    try:
        result = self._do_risky_thing(envelope["payload"])
        return {"status": "success", "result": result}

    except ValueError as e:
        # Client error - don't retry
        logger.error(f"Invalid input: {e}")
        return {"status": "error", "message": str(e), "error_type": "validation"}

    except ConnectionError as e:
        # Transient error - can retry
        logger.warning(f"Connection failed: {e}")
        raise  # Re-raise to trigger retry logic

    except Exception as e:
        # Unexpected error
        logger.exception(f"Unexpected error: {e}")
        return {"status": "error", "message": "Internal error", "error_type": "internal"}
```

### Job-Level Error Handling

Jobs that raise exceptions will be automatically retried based on `max_retries`:

```python
def process_job(job):
    if not validate(job.payload):
        # Don't retry validation errors
        return {"status": "error", "message": "Invalid payload"}

    try:
        result = process(job.payload)
        return {"status": "success", "result": result}
    except TemporaryError:
        # Retry by raising exception
        raise
```

### Dead Letter Queue

Failed jobs (exhausted retries) go to DLQ:

```python
# Inspect DLQ
dlq_name = "jobs.ocr.dlq"
channel.queue_declare(queue=dlq_name, passive=True)

# Reprocess from DLQ
def reprocess_dlq():
    def callback(ch, method, props, body):
        job = Job.from_dict(json.loads(body))
        # Reprocess or log
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=dlq_name, on_message_callback=callback)
    channel.start_consuming()
```

## Testing

### Unit Testing Agents

```python
import unittest
from unittest.mock import Mock, patch

class TestMyAgent(unittest.TestCase):
    def setUp(self):
        self.agent = MyAgent(agent_id="test.agent")
        self.agent.transport = Mock()  # Mock transport

    def test_handle_command(self):
        envelope = {
            "type": "cmd.my_action",
            "payload": {"key": "value"}
        }

        result = self.agent._handle_my_action(envelope)

        self.assertEqual(result["status"], "success")
        self.assertIn("result", result)
```

### Integration Testing

```python
def test_workflow_integration():
    transport = RabbitMQTransport(host="localhost", agent_id="test.client")
    transport.connect()

    # Submit request
    response = transport.send_command(
        target="agent.ocr.preprocessor",
        command="normalize_image",
        payload={"image_path": "/test/image.jpg"},
        timeout=30
    )

    # Verify response
    assert response["payload"]["status"] == "success"

    transport.close()
```

## Performance Optimization

### Batching

```python
class BatchProcessor(BaseAgent):
    def __init__(self):
        super().__init__("batch.processor", "batch")
        self._batch = []
        self._batch_size = 10

    def _handle_item(self, envelope):
        self._batch.append(envelope["payload"])

        if len(self._batch) >= self._batch_size:
            results = self._process_batch(self._batch)
            self._batch = []
            return {"status": "success", "results": results}

        return {"status": "queued", "batch_size": len(self._batch)}
```

### Caching

```python
from functools import lru_cache

class CachedAgent(BaseAgent):
    @lru_cache(maxsize=1000)
    def _expensive_operation(self, key):
        # Expensive computation
        return compute(key)

    def _handle_request(self, envelope):
        key = envelope["payload"]["key"]
        result = self._expensive_operation(key)
        return {"status": "success", "result": result}
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

class ParallelAgent(BaseAgent):
    def __init__(self):
        super().__init__("parallel", "parallel")
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _handle_batch(self, envelope):
        items = envelope["payload"]["items"]

        # Process in parallel
        futures = [
            self._executor.submit(self._process_item, item)
            for item in items
        ]

        results = [f.result() for f in futures]
        return {"status": "success", "results": results}
```

## Security Considerations

### Authentication

```python
class SecureAgent(BaseAgent):
    def __init__(self):
        super().__init__("secure", "secure")
        self._api_keys = load_api_keys()

    def _route_message(self, envelope):
        # Verify sender
        sender = envelope["metadata"]["source"]
        api_key = envelope["payload"].get("api_key")

        if not self._verify_sender(sender, api_key):
            return {"status": "error", "message": "Unauthorized"}

        return super()._route_message(envelope)
```

### Input Validation

```python
from pydantic import BaseModel, ValidationError

class ImagePayload(BaseModel):
    image_path: str
    options: dict = {}

class ValidatingAgent(BaseAgent):
    def _handle_process(self, envelope):
        try:
            payload = ImagePayload(**envelope["payload"])
            # Process validated payload
            return self._do_process(payload)
        except ValidationError as e:
            return {"status": "error", "message": str(e)}
```

## Monitoring and Observability

### Metrics

```python
import time

class MetricsAgent(BaseAgent):
    def __init__(self):
        super().__init__("metrics", "metrics")
        self._metrics = {
            "requests": 0,
            "successes": 0,
            "errors": 0,
            "total_time": 0.0
        }

    def _route_message(self, envelope):
        start = time.time()
        self._metrics["requests"] += 1

        try:
            result = super()._route_message(envelope)

            if result.get("status") == "success":
                self._metrics["successes"] += 1
            else:
                self._metrics["errors"] += 1

            return result
        finally:
            self._metrics["total_time"] += time.time() - start
```

### Logging

```python
import logging
import json

# Structured logging
logger = logging.getLogger(__name__)

class LoggingAgent(BaseAgent):
    def _route_message(self, envelope):
        logger.info("processing_message", extra={
            "agent_id": self.metadata.agent_id,
            "message_type": envelope["type"],
            "correlation_id": envelope["metadata"]["correlation_id"]
        })

        result = super()._route_message(envelope)

        logger.info("message_processed", extra={
            "status": result.get("status"),
            "correlation_id": envelope["metadata"]["correlation_id"]
        })

        return result
```

## Best Practices

1. **Keep Agents Focused**: Each agent should do one thing well
2. **Use Timeouts**: Always set reasonable timeouts for commands
3. **Handle Failures Gracefully**: Distinguish between retriable and fatal errors
4. **Log Extensively**: Include correlation IDs for request tracing
5. **Monitor Queues**: Watch queue depths and dead letter queues
6. **Version Your Messages**: Include version in envelope for compatibility
7. **Validate Inputs**: Use Pydantic or similar for payload validation
8. **Test Thoroughly**: Unit test handlers, integration test workflows
9. **Document Capabilities**: Keep capability schemas up to date
10. **Use Priorities Wisely**: Reserve high priorities for truly urgent tasks

## Troubleshooting

### Agent Not Receiving Messages

1. Check binding keys match routing keys
2. Verify agent is connected: `docker logs agent-name`
3. Check RabbitMQ bindings in management UI
4. Ensure exchange and queue exist

### High Latency

1. Check queue depth - may need more workers
2. Review agent processing time in logs
3. Consider caching expensive operations
4. Use batch processing for bulk operations

### Memory Issues

1. Monitor agent memory usage: `docker stats`
2. Clear caches periodically
3. Process large payloads in streaming fashion
4. Use job queue for heavy workloads

## Additional Resources

- [RabbitMQ Best Practices](https://www.rabbitmq.com/best-practices.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [AutoGen Framework](https://microsoft.github.io/autogen/)
