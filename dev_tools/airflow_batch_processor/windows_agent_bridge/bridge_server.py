"""Windows Agent Bridge Server.

Routes RabbitMQ messages to appropriate handlers:
- agent_exec_request → Docker/Airflow command execution
- kiwoom_request → Kiwoom API via pykiwoom
"""

import json
import logging
import os

import pika

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bridge-server")

# Configuration
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")

# ---------------------------------------------------------------------------
# Handler 1: Docker/Airflow command execution (original)
# ---------------------------------------------------------------------------

QUEUE_AGENT_EXEC = "agent_exec_request"

ALLOWED_COMMANDS = {
    "docker ps": "List running containers",
    "docker images": "List images",
    "docker logs": "Fetch logs",
    "docker exec": "Execute command in container",
    "docker compose": "Manage stack",
    "airflow tasks test": "Run airflow debug command",
}


def validate_command(cmd_str):
    for allowed in ALLOWED_COMMANDS:
        if cmd_str.startswith(allowed):
            return True
    return False


def on_agent_exec_request(ch, method, props, body):
    """Handle Docker/Airflow command execution requests."""
    try:
        payload = json.loads(body)
        cmd = payload.get("cmd")
        req_id = payload.get("id")

        logger.info("Exec request %s: %s", req_id, cmd)

        response = {}
        if not cmd:
            response = {"status": "error", "output": "No command provided"}
        elif not validate_command(cmd):
            response = {"status": "denied", "output": f"Command not allowed: {cmd}"}
        else:
            import subprocess  # noqa: PLC0415

            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            response = {
                "status": "ok" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        ch.basic_publish(
            exchange="",
            routing_key=props.reply_to,
            properties=pika.BasicProperties(correlation_id=props.correlation_id),
            body=json.dumps(response),
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("Sent exec reply for %s", req_id)

    except Exception as e:
        logger.error("Exec handler error: %s", e, exc_info=True)
        ch.basic_ack(delivery_tag=method.delivery_tag)


# ---------------------------------------------------------------------------
# Handler 2: Kiwoom API (from kiwoom_handler.py)
# ---------------------------------------------------------------------------

try:
    from kiwoom_handler import QUEUE_KIWOOM_REQUEST, register_kiwoom_handler

    KIWOOOM_AVAILABLE = True
except ImportError as e:
    logger.warning("Kiwoom handler not available: %s", e)
    KIWOOOM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Main: Register handlers and start consuming
# ---------------------------------------------------------------------------


def main():
    handlers = []

    logger.info("Connecting to RabbitMQ at %s...", RABBITMQ_HOST)
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    logger.info("Connected to RabbitMQ at %s", RABBITMQ_HOST)

    # Register Docker/Airflow handler
    ch1 = connection.channel()
    ch1.queue_declare(queue=QUEUE_AGENT_EXEC, durable=True)
    ch1.basic_qos(prefetch_count=1)
    ch1.basic_consume(queue=QUEUE_AGENT_EXEC, on_message_callback=on_agent_exec_request)
    handlers.append(("Docker command execution handler", QUEUE_AGENT_EXEC))
    logger.info("Registered handler: %s (queue: %s)", handlers[-1][0], handlers[-1][1])

    # Register Kiwoom handler (if available)
    if KIWOOOM_AVAILABLE:
        handler_name, queue_name = register_kiwoom_handler(connection)
        handlers.append((handler_name, queue_name))
        logger.info("Registered handler: %s (queue: %s)", handler_name, queue_name)

    logger.info(
        "Bridge server started with %d handler(s). Press CTRL+C to exit.",
        len(handlers),
    )

    try:
        connection.start_consuming()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        connection.close()


if __name__ == "__main__":
    main()
