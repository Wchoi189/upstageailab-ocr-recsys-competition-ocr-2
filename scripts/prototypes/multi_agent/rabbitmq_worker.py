import json
import uuid
import time
import os
import pika

# Configuration
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
QUEUE_NAME = "q.agent.worker_1"
EXCHANGE_NAME = "iacp.topic"
BINDING_KEYS = ["cmd.lint_code.#"]

def process_task(payload):
    """Mock task processing."""
    print(f" [x] Processing task: {payload}")
    time.sleep(2)  # Simulate work
    return {"status": "success", "checked_files": payload.get("files", []), "violations": []}

def on_request(ch, method, properties, body):
    envelope = json.loads(body)
    print(f" [.] Received {envelope['type']} from {envelope['metadata']['source']}")

    # Process
    try:
        result = process_task(envelope['payload'])
        response_payload = result
        response_type = "res.lint_result"
    except Exception as e:
        response_payload = {"error": str(e)}
        response_type = "err.processing_failed"

    # Reply if needed
    if properties.reply_to:
        response_envelope = {
            "id": str(uuid.uuid4()),
            "metadata": {
                "source": "agent.worker_1",
                "target": envelope['metadata']['source'],
                "correlation_id": envelope['metadata'].get('correlation_id'),
                "timestamp": time.time()
            },
            "type": response_type,
            "payload": response_payload
        }

        ch.basic_publish(
            exchange='',
            routing_key=properties.reply_to,
            properties=pika.BasicProperties(correlation_id=properties.correlation_id),
            body=json.dumps(response_envelope)
        )
        print(f" [>] Sent response to {properties.reply_to}")

    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    print(f" [*] Connecting to RabbitMQ at {RABBITMQ_HOST}...")
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()

        channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='topic')
        result = channel.queue_declare(queue=QUEUE_NAME, exclusive=False)
        queue_name = result.method.queue

        for binding_key in BINDING_KEYS:
            channel.queue_bind(exchange=EXCHANGE_NAME, queue=queue_name, routing_key=binding_key)

        print(f" [*] Waiting for messages on {queue_name}. To exit press CTRL+C")
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=queue_name, on_message_callback=on_request)
        channel.start_consuming()
    except Exception as e:
        print(f" [!] Error: {e}")
        print(" [!] Ensure RabbitMQ is running (e.g., 'docker run -d -p 5672:5672 rabbitmq')")


if __name__ == "__main__":
    main()
