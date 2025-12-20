"""Alibaba DashScope API Backend Implementation.

Supports Qwen VL models via Alibaba Cloud DashScope API using OpenAPI-compatible interface.
"""

import time
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from AgentQMS.vlm.backends.base import BaseVLMBackend
from AgentQMS.vlm.core.config import get_config, resolve_env_value
from AgentQMS.vlm.core.contracts import AnalysisMode, BackendConfig, ProcessedImage
from AgentQMS.vlm.core.interfaces import BackendError


class DashScopeBackend(BaseVLMBackend):
    """Alibaba DashScope API backend for Qwen VLM analysis using OpenAPI-compatible interface."""

    def __init__(self, config: BackendConfig):
        """Initialize DashScope backend.

        Args:
            config: Backend configuration with DashScope API key
        """
        super().__init__(config)
        if OpenAI is None:
            raise BackendError("openai package is required for DashScope backend. Install with: pip install openai")

        settings = get_config().backends.dashscope
        api_key = config.api_key or resolve_env_value(settings.api_key_env)
        if not api_key:
            raise BackendError("DashScope API key is required. Provide it via DASHSCOPE_API_KEY environment variable.")

        # Initialize OpenAI client for DashScope compatible-mode endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.endpoint or settings.endpoint,
        )
        self.model = config.model or settings.default_model

    @property
    def model_name(self) -> str:
        """Human-readable model identifier used for this backend."""
        return self.model

    def analyze_image(
        self,
        image_data: ProcessedImage,
        prompt: str,
        mode: AnalysisMode,
        **kwargs: Any,
    ) -> str:
        """Analyze an image using DashScope OpenAPI-compatible endpoint.

        Args:
            image_data: Preprocessed image data
            prompt: Analysis prompt
            mode: Analysis mode
            **kwargs: Additional parameters

        Returns:
            Analysis text from VLM

        Raises:
            BackendError: If analysis fails
        """
        if not image_data.base64_encoded:
            raise BackendError("Image must be base64 encoded for DashScope API")

        max_retries = self.config.max_retries
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Prepare messages using OpenAI format
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_data.base64_encoded}"},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )

                # Extract response text
                if response.choices and len(response.choices) > 0:
                    message_content = response.choices[0].message.content
                    if message_content:
                        return message_content
                    else:
                        raise BackendError("Empty response from DashScope API")
                else:
                    raise BackendError("No choices in DashScope API response")

            except BackendError:
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    raise BackendError(f"Failed to analyze image with DashScope: {e}") from e

        if last_error:
            raise BackendError(f"Failed after {max_retries + 1} attempts") from last_error

        raise BackendError("Analysis failed without specific error")

    def is_available(self) -> bool:
        """Check if DashScope backend is available.

        Returns:
            True if openai package is installed and API key is configured
        """
        if OpenAI is None:
            return False
        return bool(self.client.api_key)
