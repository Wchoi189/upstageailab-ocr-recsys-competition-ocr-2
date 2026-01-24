from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class IACPEnvelope(BaseModel):
    """
    Standard envelope for Inter-Agent Communication Protocol.
    Validated by the RabbitMQTransport during send/receive.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    source_agent: str = Field(..., example="agent.coordinator")
    target_agent: str = Field(..., example="agent.ocr.inference")
    correlation_id: str = Field(..., description="Link back to the original workflow request")

    # Payload details
    type: str = Field(..., example="cmd.process_image")
    payload: Dict[str, Any]

    # Metadata for AgentQMS Monitoring
    priority: int = Field(default=5, ge=1, le=10)
    token_usage: Optional[int] = None
