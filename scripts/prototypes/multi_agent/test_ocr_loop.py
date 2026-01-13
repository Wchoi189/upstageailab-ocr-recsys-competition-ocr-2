import json
import uuid
import time
import os
import pika

# Configuration
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
EXCHANGE_NAME = "iacp.topic"
TARGET_ROUTING_KEY = "cmd.ocr.process"

def main():
    print(f" [*] Connecting to RabbitMQ at {RABBITMQ_HOST}...")
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
        channel = connection.channel()

        channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type='topic')

        # Create callback queue
        result = channel.queue_declare(queue='', exclusive=True)
        callback_queue = result.method.queue

        # Payload
        # We need a real image to test.
        # Let's search for one or use a dummy path that will trigger "File not found" error handled gracefully.
        task_payload = {
            "files": ["data/datasets/images/test_sample.jpg"], # Expecting file not found or need to provide real one
            "options": {
                "enable_extraction": False
            }
        }

        # Envelope
        correlation_id = str(uuid.uuid4())
        envelope = {
            "id": str(uuid.uuid4()),
            "metadata": {
                "source": "agent.client",
                "target": "agent.ocr",
                "correlation_id": correlation_id,
                "timestamp": time.time()
            },
            "type": "cmd.ocr.process",
            "payload": task_payload
        }

        print(f" [x] Sending: {envelope['type']}")

        def on_response(ch, method, props, body):
            if props.correlation_id == correlation_id:
                response = json.loads(body)
                print(f" [.] Got response: {json.dumps(response, indent=2)}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                # Close connection after response
                ch.stop_consuming()

        channel.basic_consume(queue=callback_queue, on_message_callback=on_response)

        channel.basic_publish(
            exchange=EXCHANGE_NAME,
            routing_key=TARGET_ROUTING_KEY,
            properties=pika.BasicProperties(
                reply_to=callback_queue,
                correlation_id=correlation_id,
            ),
            body=json.dumps(envelope)
        )

        print(" [*] Waiting for response...")
        # connection.process_data_events(time_limit=10) # process for a bit
        channel.start_consuming()

    except Exception as e:
        print(f" [!] Error: {e}")
        print(" [!] Ensure RabbitMQ is running")

    finally:
        if 'connection' in locals() and connection.is_open:
            connection.close()

if __name__ == "__main__":
    main()
