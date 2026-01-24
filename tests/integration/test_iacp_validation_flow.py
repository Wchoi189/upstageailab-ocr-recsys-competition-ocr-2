import os
import sys
import json
import time
import threading
import logging
import uuid
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ocr.core.infrastructure.agents.validation_agent import ValidationAgent
from ocr.core.infrastructure.communication.rabbitmq_transport import RabbitMQTransport
from ocr.core.infrastructure.communication.iacp_schemas import IACPEnvelope

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationTest")

def run_agent():
    """Run ValidationAgent in a separate thread."""
    try:
        agent = ValidationAgent(agent_id="agent.ocr.validator.test")
        logger.info("Starting ValidationAgent...")
        agent.start()
    except Exception as e:
        logger.error(f"Agent crashed: {e}")

def test_validation_flow():
    """Test full IACP flow ValidationAgent -> QwenClient -> Ollama."""
    
    # 1. Start Agent in Thread
    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    
    # Wait for agent to bind
    time.sleep(2)
    
    # 2. Setup Test Client
    client_transport = RabbitMQTransport(host=os.getenv("RABBITMQ_HOST", "rabbitmq"), agent_id="agent.test.client")
    client_transport.connect()
    
    # 3. Create Payload
    test_text = "Th1s is a t3st messag3 with s0me err0rs."
    payload = {
        "text": test_text,
        "context": "Integration test for error detection."
    }
    
    logger.info(f"Sending request: {test_text}")
    
    # 4. Send Command (RPC)
    try:
        # We manually construct and send to match what BaseAgent.send_command does
        # But here we use the raw transport to verify the envelope structure too
        response_envelope = client_transport.send_command(
            target="agent.ocr.validator.test",
            type_suffix="detect_errors", # becomes cmd.detect_errors
            payload=payload,
            timeout=30 # Give LLM time to think
        )
        
        # 5. Verify Response
        logger.info("Received response envelope")
        assert isinstance(response_envelope, IACPEnvelope)
        assert response_envelope.source_agent == "agent.ocr.validator.test"
        
        result = response_envelope.payload
        logger.info(f"Result Payload: {json.dumps(result, indent=2)}")
        
        assert result["status"] == "success"
        assert result["original_text"] == test_text
        # Qwen should find errors
        assert "errors_found" in result, "LLM should return errors_found list"
        
        logger.info("✅ Integration Test Passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Integration Test Failed: {e}", exc_info=True)
        return False
    finally:
        client_transport.close()
        # Thread will die with main process due to daemon=True

if __name__ == "__main__":
    success = test_validation_flow()
    sys.exit(0 if success else 1)
