import os
import logging
import signal
import sys
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

from ocr.communication.rabbitmq_transport import RabbitMQTransport
from ocr.core.utils.path_utils import PROJECT_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.
    Handles RabbitMQ connection, lifecycle, and common path utilities.
    """

    def __init__(self, agent_id: str, binding_keys: list[str]):
        """
        Initialize the BaseAgent.

        Args:
            agent_id: Unique identifier for the agent (e.g., 'agent.ocr').
            binding_keys: List of routing keys to listen for (e.g., ['cmd.ocr.#']).
        """
        self.agent_id = agent_id
        self.binding_keys = binding_keys
        self.project_root = PROJECT_ROOT

        host = os.getenv("RABBITMQ_HOST", "rabbitmq")
        self.transport = RabbitMQTransport(host=host, agent_id=agent_id)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def run(self):
        """Starts the agent loop."""
        self.logger.info(f"Starting {self.__class__.__name__} (Root: {self.project_root})...")
        try:
            self.transport.start_listening(
                binding_keys=self.binding_keys,
                handler=self.handle_request
            )
        except Exception as e:
            self.logger.error(f"Agent failed: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of resources."""
        self.logger.info("Stopping agent...")
        self.transport.close()

    def _handle_signal(self, signum, frame):
        """Handle system signals for graceful exit."""
        self.logger.info(f"Received signal {signum}")
        self.shutdown()
        sys.exit(0)

    @abstractmethod
    def process_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Process the payload of a request.

        Args:
            payload: The content of the request.

        Returns:
            Dictionary containing result data.
        """
        pass

    def handle_request(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """
        Wrapper to handle incoming requests.
        Decouples transport logic from processing logic.
        """
        try:
            payload = envelope.get('payload', {})
            self.logger.debug(f"Received payload: {payload}")

            result = self.process_payload(payload)

            # Ensure status is present if not provided by subclass
            if "status" not in result:
                result["status"] = "success"

            return result
        except Exception as e:
            self.logger.exception(f"Error processing request: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def resolve_path(self, relative_path: str) -> Path:
        """Helper to resolve paths relative to project root."""
        return self.project_root / relative_path
