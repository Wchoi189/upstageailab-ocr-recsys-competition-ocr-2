import os
import sys
import logging
import redis
import pika

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InfraCheck")

def check_redis():
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", 6379))
    logger.info(f"Checking Redis at {host}:{port}...")
    try:
        r = redis.Redis(host=host, port=port, socket_connect_timeout=2)
        r.ping()
        logger.info("✅ Redis Connection Successful")
        return True
    except Exception as e:
        logger.error(f"❌ Redis Connection Failed: {e}")
        return False

def check_rabbitmq():
    host = os.getenv("RABBITMQ_HOST", "rabbitmq")
    logger.info(f"Checking RabbitMQ at {host}...")
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, connection_attempts=1, socket_timeout=2))
        if connection.is_open:
            logger.info("✅ RabbitMQ Connection Successful")
            connection.close()
            return True
    except Exception as e:
        logger.error(f"❌ RabbitMQ Connection Failed: {e}")
        return False

if __name__ == "__main__":
    r_ok = check_redis()
    q_ok = check_rabbitmq()
    
    if r_ok and q_ok:
        sys.exit(0)
    else:
        sys.exit(1)
