import json
import uuid
import time
import pika
import logging
from typing import Any
from collections.abc import Callable

# Configure logging
logger = logging.getLogger(__name__)

class RabbitMQTransport:
    """
    Implements the Inter-Agent Communication Protocol (IACP) transport layer using RabbitMQ.
    """

    def __init__(self, host: str, exchange: str = "iacp.topic", agent_id: str = None):
        """
        Initialize the transport.

        Args:
            host: RabbitMQ host address.
            exchange: Exchange name to use (default: iacp.topic).
            agent_id: Unique identifier for this agent. If None, one is generated.
        """
        self.host = host
        self.exchange = exchange
        self.agent_id = agent_id or f"agent.{uuid.uuid4().hex[:8]}"
        self.connection: pika.BlockingConnection | None = None
        self.channel: pika.adapters.blocking_connection.BlockingChannel | None = None
        self.callback_queue: str | None = None
        self.listening_queue: str | None = None
        self.response_futures: dict[str, Any] = {} # Helper for blocking RPC calls

    def connect(self):
        """Establishes connection to RabbitMQ."""
        try:
            logger.info(f"Connecting to RabbitMQ at {self.host} as {self.agent_id}...")
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
            self.channel = self.connection.channel()
            self.channel.exchange_declare(exchange=self.exchange, exchange_type='topic')

            # Setup callback queue for RPC responses
            result = self.channel.queue_declare(queue='', exclusive=True)
            self.callback_queue = result.method.queue
            self.channel.basic_consume(
                queue=self.callback_queue,
                on_message_callback=self._on_rpc_response,
                auto_ack=True
            )

            logger.info("Connected to RabbitMQ.")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def close(self):
        """Closes the connection."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("Connection closed.")

    def _wrap_envelope(self, target: str, msg_type: str, payload: dict[str, Any], correlation_id: str = None, priority: int = 5, ttl: int = None) -> dict[str, Any]:
        """Creates a standardized IACP envelope."""
        return {
            "id": str(uuid.uuid4()),
            "version": "1.0",
            "metadata": {
                "source": self.agent_id,
                "target": target,
                "correlation_id": correlation_id or str(uuid.uuid4()),
                "timestamp": time.time(),
                "priority": priority,
                "ttl": ttl
            },
            "type": msg_type,
            "payload": payload
        }

    def send_command(self, target: str, type_suffix: str, payload: dict[str, Any], timeout: int = 10) -> dict[str, Any]:
        """
        Sends a command and waits for a response (RPC pattern).

        Args:
            target: Target agent ID or routing key.
            type_suffix: Specific command type (e.g. 'lint_code' -> 'cmd.lint_code').
            payload: Command arguments.
            timeout: Timeout in seconds.

        Returns:
            Review the response payload.
        """
        if not self.connection:
            raise RuntimeError("Not connected to RabbitMQ")

        full_type = f"cmd.{type_suffix}" if not type_suffix.startswith("cmd.") else type_suffix
        correlation_id = str(uuid.uuid4())

        envelope = self._wrap_envelope(target, full_type, payload, correlation_id=correlation_id)
        routing_key = f"{full_type}.{self.agent_id}.{target}"

        logger.info(f"Sending command {full_type} to {target} (corr_id={correlation_id})")

        self.response_futures[correlation_id] = None

        self.channel.basic_publish(
            exchange=self.exchange,
            routing_key=routing_key,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=correlation_id,
            ),
            body=json.dumps(envelope)
        )

        # Wait for response
        start_time = time.time()
        while self.response_futures[correlation_id] is None:
            self.connection.process_data_events()
            if time.time() - start_time > timeout:
                del self.response_futures[correlation_id]
                raise TimeoutError(f"Timeout waiting for response to {full_type}")
            time.sleep(0.01)

        response = self.response_futures.pop(correlation_id)
        return response

    def publish_event(self, type_suffix: str, payload: dict[str, Any]):
        """Publish a fire-and-forget event."""
        if not self.connection:
            raise RuntimeError("Not connected to RabbitMQ")

        full_type = f"evt.{type_suffix}" if not type_suffix.startswith("evt.") else type_suffix
        envelope = self._wrap_envelope("broadcast", full_type, payload)
        routing_key = f"{full_type}.{self.agent_id}.all"

        self.channel.basic_publish(
            exchange=self.exchange,
            routing_key=routing_key,
            body=json.dumps(envelope)
        )
        logger.info(f"Published event {full_type}")

    def _on_rpc_response(self, ch, method, props, body):
        """Internal callback for RPC responses."""
        if props.correlation_id in self.response_futures:
            self.response_futures[props.correlation_id] = json.loads(body)

    def start_listening(self, binding_keys: list[str], handler: Callable[[dict[str, Any]], dict[str, Any]]):
        """
        Starts listening for messages.

        Args:
            binding_keys: List of routing keys to bind to (e.g. 'cmd.lint_code.#').
            handler: Function that takes an envelope and returns a response payload (or None).
        """
        if not self.connection:
            self.connect()

        # Create queue for this agent
        queue_name = f"q.{self.agent_id}"
        self.channel.queue_declare(queue=queue_name)
        self.listening_queue = queue_name

        for key in binding_keys:
            self.channel.queue_bind(exchange=self.exchange, queue=queue_name, routing_key=key)

        logger.info(f"Listening on {queue_name} for keys: {binding_keys}")

        def on_message(ch, method, props, body):
            try:
                envelope = json.loads(body)
                logger.info(f"Received {envelope.get('type')} from {envelope.get('metadata', {}).get('source')}")

                # Execute handler
                result_payload = handler(envelope)

                # Send response if requested
                if props.reply_to and result_payload is not None:
                    response_type = "res.success" # Should ideally be derived

                    response_envelope = {
                        "id": str(uuid.uuid4()),
                        "version": "1.0",
                        "metadata": {
                            "source": self.agent_id,
                            "target": envelope['metadata']['source'],
                            "correlation_id": props.correlation_id,
                            "timestamp": time.time()
                        },
                        "type": response_type,
                        "payload": result_payload
                    }

                    ch.basic_publish(
                        exchange='',
                        routing_key=props.reply_to,
                        properties=pika.BasicProperties(correlation_id=props.correlation_id),
                        body=json.dumps(response_envelope)
                    )
                    logger.info("Sent response.")

                ch.basic_ack(delivery_tag=method.delivery_tag)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False) # Dead letter?

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=queue_name, on_message_callback=on_message)
        self.channel.start_consuming()
