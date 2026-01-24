"""
Base Agent Framework for Multi-Agent Collaboration Environment.

This module provides the foundation for AutoGen-based agents that communicate
via RabbitMQ using the Inter-Agent Communication Protocol (IACP).
"""

import logging
import sys
import signal
from typing import Any
from collections.abc import Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Updated import path for IACP Transport
from ocr.core.infrastructure.communication.rabbitmq_transport import RabbitMQTransport
from ocr.core.infrastructure.communication.iacp_schemas import IACPEnvelope
from ocr.core.utils.path_utils import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Defines what an agent can do."""
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


@dataclass
class AgentMetadata:
    """Metadata about an agent."""
    agent_id: str
    agent_type: str
    capabilities: list[AgentCapability] = field(default_factory=list)
    status: str = "idle"  # idle, busy, error
    started_at: datetime | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent collaboration environment.

    Agents communicate via RabbitMQ using IACP and can be orchestrated
    using AutoGen patterns while maintaining message-based architecture.
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        rabbitmq_host: str = "rabbitmq",
        capabilities: list[AgentCapability] | None = None
    ):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for this agent instance
            agent_type: Type/role of the agent (e.g., 'ocr.preprocessor')
            rabbitmq_host: RabbitMQ server hostname
            capabilities: List of capabilities this agent provides
        """
        self.metadata = AgentMetadata(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities or [],
            started_at=datetime.now()
        )
        self.project_root = PROJECT_ROOT

        self.transport = RabbitMQTransport(
            host=rabbitmq_host,
            agent_id=agent_id
        )

        self._handlers: dict[str, Callable[[IACPEnvelope], dict[str, Any]]] = {}
        self._register_default_handlers()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(f"Initialized {agent_type} agent: {agent_id} (Root: {self.project_root})")

    def _register_default_handlers(self):
        """Register default message handlers."""
        self._handlers["cmd.get_status"] = self._handle_get_status
        self._handlers["cmd.get_capabilities"] = self._handle_get_capabilities
        self._handlers["qry.health_check"] = self._handle_health_check

    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a custom message handler.

        Args:
            message_type: Type of message to handle (e.g., 'cmd.process_image')
            handler: Callable that takes IACPEnvelope and returns response payload
        """
        self._handlers[message_type] = handler
        logger.info(f"Registered handler for {message_type}")

    def _route_message(self, envelope: IACPEnvelope) -> dict[str, Any]:
        """
        Route incoming message to appropriate handler.

        Args:
            envelope: IACP message envelope

        Returns:
            Response payload
        """
        msg_type = envelope.type

        if msg_type not in self._handlers:
            # Try partial matching for wildcards if we supported them, but strict matching is safer for now
            # Only exact match for now as per registry
            logger.warning(f"No handler for message type: {msg_type}")
            return {
                "status": "error",
                "message": f"Unsupported message type: {msg_type}"
            }

        try:
            self.metadata.status = "busy"
            result = self._handlers[msg_type](envelope)
            self.metadata.tasks_completed += 1
            self.metadata.status = "idle"
            return result
        except Exception as e:
            logger.error(f"Error handling {msg_type}: {e}", exc_info=True)
            self.metadata.tasks_failed += 1
            self.metadata.status = "error"
            return {
                "status": "error",
                "message": str(e),
                "error_type": type(e).__name__
            }

    def _handle_get_status(self, envelope: IACPEnvelope) -> dict[str, Any]:
        """Handle status query."""
        return {
            "agent_id": self.metadata.agent_id,
            "agent_type": self.metadata.agent_type,
            "status": self.metadata.status,
            "tasks_completed": self.metadata.tasks_completed,
            "tasks_failed": self.metadata.tasks_failed,
            "uptime_seconds": (datetime.now() - self.metadata.started_at).total_seconds()
        }

    def _handle_get_capabilities(self, envelope: IACPEnvelope) -> dict[str, Any]:
        """Handle capabilities query."""
        return {
            "agent_id": self.metadata.agent_id,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "input_schema": cap.input_schema,
                    "output_schema": cap.output_schema
                }
                for cap in self.metadata.capabilities
            ]
        }

    def _handle_health_check(self, envelope: IACPEnvelope) -> dict[str, Any]:
        """Handle health check query."""
        return {
            "status": "healthy",
            "agent_id": self.metadata.agent_id,
            "timestamp": datetime.now().isoformat()
        }

    @abstractmethod
    def get_binding_keys(self) -> list[str]:
        """
        Return list of routing keys this agent should listen to.

        Example: ['cmd.process_image.#', 'cmd.validate_ocr.#']
        """
        pass

    def start(self):
        """Start the agent and begin listening for messages."""
        logger.info(f"Starting {self.metadata.agent_type} agent...")

        try:
            self.transport.connect()
            binding_keys = self.get_binding_keys()

            # Add default binding keys (unique per agent instance)
            binding_keys.extend([
                f"cmd.get_status.*.{self.metadata.agent_id}",
                f"cmd.get_capabilities.*.{self.metadata.agent_id}",
                f"qry.health_check.*.{self.metadata.agent_id}"
            ])

            logger.info(f"Listening on keys: {binding_keys}")
            self.transport.start_listening(
                binding_keys=binding_keys,
                handler=self._route_message
            )
        except KeyboardInterrupt:
            logger.info("Shutting down agent...")
            self.stop()
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            self.stop()
            raise

    def stop(self):
        """Stop the agent and cleanup resources."""
        logger.info(f"Stopping {self.metadata.agent_type} agent...")
        self.transport.close()

    def _handle_signal(self, signum, frame):
        """Handle system signals for graceful exit."""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)

    def send_command(
        self,
        target: str,
        command: str,
        payload: dict[str, Any],
        timeout: int = 30
    ) -> dict[str, Any]:
        """
        Send a command to another agent.

        Args:
            target: Target agent ID
            command: Command type (e.g., 'process_image')
            payload: Command payload
            timeout: Timeout in seconds

        Returns:
            Response payload from target agent
        """
        response_envelope = self.transport.send_command(
            target=target,
            type_suffix=command,
            payload=payload,
            timeout=timeout
        )
        # Return only payload to maintain abstraction level for callers
        return response_envelope.payload

    def publish_event(self, event_type: str, payload: dict[str, Any]):
        """
        Publish an event to all interested agents.

        Args:
            event_type: Event type (e.g., 'task_completed')
            payload: Event data
        """
        self.transport.publish_event(
            type_suffix=event_type,
            payload=payload
        )

    def resolve_path(self, relative_path: str) -> Path:
        """Helper to resolve paths relative to project root."""
        return self.project_root / relative_path


class LLMAgent(BaseAgent):
    """
    Base class for agents that use LLM inference.

    Provides abstraction for different LLM providers (QwenCLI, Grok4, etc.)
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        llm_provider: str = "qwen",
        rabbitmq_host: str = "rabbitmq",
        capabilities: list[AgentCapability] | None = None
    ):
        """
        Initialize LLM agent.

        Args:
            agent_id: Unique agent identifier
            agent_type: Agent type
            llm_provider: LLM provider to use ('qwen', 'grok4', 'openai')
            rabbitmq_host: RabbitMQ host
            capabilities: Agent capabilities
        """
        super().__init__(agent_id, agent_type, rabbitmq_host, capabilities)
        self.llm_provider = llm_provider
        self._llm_client = None

    def _get_llm_client(self):
        """Get or initialize LLM client."""
        if self._llm_client is None:
            if self.llm_provider == "qwen":
                from ocr.core.infrastructure.agents.llm.qwen_client import QwenClient
                self._llm_client = QwenClient()
            elif self.llm_provider == "grok4":
                from ocr.core.infrastructure.agents.llm.grok_client import Grok4Client
                self._llm_client = Grok4Client()
            elif self.llm_provider == "openai":
                from ocr.core.infrastructure.agents.llm.openai_client import OpenAIClient
                self._llm_client = OpenAIClient()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

        return self._llm_client

    def generate_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate a response using the configured LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        client = self._get_llm_client()
        return client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
