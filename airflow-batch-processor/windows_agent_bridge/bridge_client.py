import pika
import uuid
import json
import sys
import os

# Configuration
# "host.docker.internal" works on Docker Desktop (Mac/Win).
# On pure Linux or some setups, use "172.17.0.1" (gateway).
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "host.docker.internal")
QUEUE_REQUEST = "agent_exec_request"

class AgentBridgeClient:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True
        )

        self.response = None
        self.corr_id = None

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = json.loads(body)

    def call(self, command):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        
        request = {
            "cmd": command,
            "id": self.corr_id
        }

        self.channel.basic_publish(
            exchange='',
            routing_key=QUEUE_REQUEST,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=json.dumps(request)
        )
        
        # Blocking wait
        while self.response is None:
            self.connection.process_data_events()
            
        return self.response

    def close(self):
        self.connection.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python bridge_client.py \"<command>\"")
        sys.exit(1)

    cmd = sys.argv[1]
    client = None
    try:
        client = AgentBridgeClient()
        print(f"Sending command: {cmd}")
        response = client.call(cmd)
        
        if response.get("status") == "ok":
            print("SUCCESS:")
            print(response.get("stdout"))
            if response.get("stderr"):
                print("STDERR:", response.get("stderr"))
        else:
            print(f"FAILURE ({response.get('status')}):")
            print(response.get("output", ""))
            print(response.get("stderr", ""))
            
    except pika.exceptions.AMQPConnectionError:
        print(f"Error: Could not connect to RabbitMQ at {RABBITMQ_HOST}.")
        print("Ensure RabbitMQ is running and accessible.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if client:
            client.close()

if __name__ == "__main__":
    main()
