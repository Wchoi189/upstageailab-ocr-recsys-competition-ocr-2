"""OpenRouter API Backend Implementation."""

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


class OpenRouterBackend(BaseVLMBackend):
    """OpenRouter API backend for VLM analysis."""

    def __init__(self, config: BackendConfig):
        """Initialize OpenRouter backend.

        Args:
            config: Backend configuration with OpenRouter API key
        """
        super().__init__(config)
        if OpenAI is None:
            raise BackendError("openai package is required for OpenRouter backend. Install with: pip install openai")

        settings = get_config().backends.openrouter
        api_key = config.api_key or resolve_env_value(settings.api_key_env)
        if not api_key:
            raise BackendError("OpenRouter API key is required. Provide it via configuration or environment variables.")

        base_url = config.endpoint or settings.base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
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
        """Analyze an image using OpenRouter API.

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
            raise BackendError("Image must be base64 encoded for OpenRouter API")

        max_retries = self.config.max_retries
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{image_data.format.value.lower()};base64,{image_data.base64_encoded}"
                                    },
                                },
                            ],
                        }
                    ],
                    timeout=self.config.timeout_seconds,
                )

                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()

                raise BackendError("Empty response from OpenRouter API")

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(2**attempt)  # Exponential backoff
                    continue
                break

        raise BackendError(f"Failed to analyze image after {max_retries + 1} attempts: {last_error}") from last_error

    def supports_batch(self) -> bool:
        """OpenRouter supports batch processing via multiple API calls."""
        return True

    def is_available(self) -> bool:
        """Check if OpenRouter backend is available."""
        if OpenAI is None:
            return False

        settings = get_config().backends.openrouter
        api_key = self.config.api_key or resolve_env_value(settings.api_key_env)
        return api_key is not None
