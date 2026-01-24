import sys
import json
import logging
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ocr.core.infrastructure.communication.rabbitmq_transport import RabbitMQTransport
from ocr.core.infrastructure.communication.iacp_schemas import IACPEnvelope

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ClientRunner")

def run_client():
    client_transport = RabbitMQTransport(host=os.getenv("RABBITMQ_HOST", "rabbitmq"), agent_id="agent.test.client")
    client_transport.connect()
    
    test_text = "Th1s is a t3st messag3 with s0me err0rs."
    payload = {
        "text": test_text,
        "context": "Integration test for error detection."
    }
    
    logger.info(f"Sending request: {test_text}")
    
    try:
        response_envelope = client_transport.send_command(
            target="agent.ocr.validator.test",
            type_suffix="detect_errors",
            payload=payload,
            timeout=30
        )
        
        result = response_envelope.payload
        logger.info(f"Result Payload: {json.dumps(result, indent=2)}")
        
        if result["status"] == "success":
            logger.info("✅ SUCCESS")
        else:
            logger.error("❌ FAILURE: Status not success")
            
    except Exception as e:
        logger.error(f"❌ FAILURE: {e}")
    finally:
        client_transport.close()

if __name__ == "__main__":
    run_client()
