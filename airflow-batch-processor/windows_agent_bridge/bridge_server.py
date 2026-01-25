import pika
import json
import subprocess
import os
import sys

# Configuration
RABBITMQ_HOST = "localhost"
QUEUE_REQUEST = "agent_exec_request"
QUEUE_REPLY = "agent_exec_reply"

ALLOWED_COMMANDS = {
    "docker ps": "List running containers",
    "docker images": "List images",
    "docker logs": "Fetch logs",
    "docker exec": "Execute command in container",
    "docker compose": "Manage stack",
    "airflow tasks test": "Run airflow debug command"  # Specific allowance
}

def validate_command(cmd_str):
    """
    Very basic validation. Ensure command starts with an allowed prefix.
    In production, use strict parsing.
    """
    for allowed in ALLOWED_COMMANDS:
        if cmd_str.startswith(allowed):
            return True
    return False

def on_request(ch, method, props, body):
    try:
        payload = json.loads(body)
        cmd = payload.get("cmd")
        req_id = payload.get("id")
        
        print(f" [.] Received request {req_id}: {cmd}")

        response = {}
        
        if not cmd:
            response = {"status": "error", "output": "No command provided"}
        elif not validate_command(cmd):
            response = {"status": "denied", "output": f"Command not allowed: {cmd}"}
        else:
            # Execute
            try:
                # shell=True required for complex args, but dangerous. 
                # Since we validate strictly (in theory), acceptable for dev bridge.
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                response = {
                    "status": "ok" if result.returncode == 0 else "failed",
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            except Exception as e:
                response = {"status": "error", "output": str(e)}

        # Reply
        ch.basic_publish(
            exchange='',
            routing_key=props.reply_to,
            properties=pika.BasicProperties(correlation_id=props.correlation_id),
            body=json.dumps(response)
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f" [x] Sent reply for {req_id}")

    except Exception as e:
        print(f" [!] Error processing message: {e}")
        # Ack anyway to avoid loop
        ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    print(" [*] Connecting to RabbitMQ at localhost...")
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        channel = connection.channel()

        channel.queue_declare(queue=QUEUE_REQUEST)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=QUEUE_REQUEST, on_message_callback=on_request)

        print(f" [*] Waiting for commands in queue '{QUEUE_REQUEST}'. To exit press CTRL+C")
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
