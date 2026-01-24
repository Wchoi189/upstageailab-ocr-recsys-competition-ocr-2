import sys
import logging
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ocr.core.infrastructure.agents.validation_agent import ValidationAgent

# Configure logging to stdout explicitly
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("AgentRunner")

if __name__ == "__main__":
    try:
        logger.info("Initializing ValidationAgent...")
        agent = ValidationAgent(agent_id="agent.ocr.validator.test")
        logger.info("Starting ValidationAgent...")
        agent.start() # Blocking call
    except KeyboardInterrupt:
        logger.info("Stopping agent...")
        agent.stop()
    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        sys.exit(1)
