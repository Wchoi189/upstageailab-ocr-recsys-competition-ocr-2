import json
import uuid
import pika
import logging
from typing import Any, Optional, Callable
from pydantic import ValidationError

# Import the IACP Envelope Schema
try:
    from ocr.core.infrastructure.communication.iacp_schemas import IACPEnvelope
except ImportError:
    # Fallback/Mock for initial setup if package path isn't perfect yet
    from pydantic import BaseModel, Field
    class IACPEnvelope(BaseModel):
        pass # Should be proper import in production

logger = logging.getLogger(__name__)

class RabbitMQTransport:
    """
    Implements the Inter-Agent Communication Protocol (IACP) with strict Pydantic validation.
    """

    def __init__(self, host: str, exchange: str = "iacp.topic", agent_id: str = None):
        self.host = host
        self.exchange = exchange
        self.agent_id = agent_id or f"agent.{uuid.uuid4().hex[:8]}"
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None
        self.callback_queue: Optional[str] = None
        self.response_futures: dict[str, Optional[IACPEnvelope]] = {}

    def connect(self):
        """Establishes connection and declares IACP exchange."""
        try:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))
            self.channel = self.connection.channel()
            self.channel.exchange_declare(exchange=self.exchange, exchange_type='topic')

            result = self.channel.queue_declare(queue='', exclusive=True)
            self.callback_queue = result.method.queue
            self.channel.basic_consume(
                queue=self.callback_queue,
                on_message_callback=self._on_rpc_response,
                auto_ack=True
            )
            logger.info(f"Connected as {self.agent_id}")
        except Exception as e:
            logger.error(f"RabbitMQ Connection failed: {e}")
            raise

    def _create_envelope(self, target: str, msg_type: str, payload: dict, correlation_id: str = None) -> IACPEnvelope:
        """Creates and validates an outgoing IACP envelope."""
        envelope = IACPEnvelope(
            source_agent=self.agent_id,
            target_agent=target,
            correlation_id=correlation_id or str(uuid.uuid4()),
            type=msg_type,
            payload=payload
        )
        return envelope

    def send_command(self, target: str, type_suffix: str, payload: dict, timeout: int = 10) -> IACPEnvelope:
        """Sends a validated command and waits for a validated response."""
        full_type = f"cmd.{type_suffix}" if not type_suffix.startswith("cmd.") else type_suffix
        envelope = self._create_envelope(target, full_type, payload)
        corr_id = envelope.correlation_id

        routing_key = f"{full_type}.{self.agent_id}.{target}"
        self.response_futures[corr_id] = None

        self.channel.basic_publish(
            exchange=self.exchange,
            routing_key=routing_key,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=corr_id,
            ),
            body=envelope.model_dump_json() # Use Pydantic's optimized JSON export
        )

        # Wait loop with timeout
        import time
        start_time = time.time()
        while self.response_futures[corr_id] is None:
            self.connection.process_data_events()
            if time.time() - start_time > timeout:
                del self.response_futures[corr_id]
                raise TimeoutError(f"IACP Timeout: {full_type}")
            time.sleep(0.01)

        return self.response_futures.pop(corr_id)

    def _on_rpc_response(self, ch, method, props, body):
        """Validates incoming RPC responses against the schema."""
        try:
            response = IACPEnvelope.model_validate_json(body)
            if props.correlation_id in self.response_futures:
                self.response_futures[props.correlation_id] = response
        except ValidationError as e:
            logger.error(f"Invalid IACP Response received: {e.json()}")

    def start_listening(self, binding_keys: list[str], handler: Callable[[IACPEnvelope], dict]):
        """Starts the listening loop with strict schema enforcement for all incoming messages."""
        queue_name = f"q.{self.agent_id}"
        self.channel.queue_declare(queue=queue_name)

        for key in binding_keys:
            self.channel.queue_bind(exchange=self.exchange, queue=queue_name, routing_key=key)

        def on_message(ch, method, props, body):
            try:
                # 1. Validate incoming envelope
                envelope = IACPEnvelope.model_validate_json(body)

                # 2. Execute agent handler
                result_payload = handler(envelope)

                # 3. Send validated response if requested
                if props.reply_to and result_payload is not None:
                    res_envelope = self._create_envelope(
                        target=envelope.source_agent,
                        msg_type="res.success",
                        payload=result_payload,
                        correlation_id=props.correlation_id
                    )
                    ch.basic_publish(
                        exchange='',
                        routing_key=props.reply_to,
                        properties=pika.BasicProperties(correlation_id=props.correlation_id),
                        body=res_envelope.model_dump_json()
                    )
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except ValidationError as e:
                logger.error(f"IACP Schema Violation: Rejecting message. Error: {e.json()}")
                # NACK without requeue to move to DLQ or drop invalid traffic
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception as e:
                logger.error(f"Handler error: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=queue_name, on_message_callback=on_message)
        self.channel.start_consuming()
