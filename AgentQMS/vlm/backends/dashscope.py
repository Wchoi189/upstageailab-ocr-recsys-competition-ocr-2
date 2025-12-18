"""Alibaba DashScope API Backend Implementation.

Supports Qwen VL models via Alibaba Cloud DashScope API.
"""

import time
from typing import Any

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    dashscope = None
    MultiModalConversation = None

from AgentQMS.vlm.backends.base import BaseVLMBackend
from AgentQMS.vlm.core.config import get_config, resolve_env_value
from AgentQMS.vlm.core.contracts import AnalysisMode, BackendConfig, ProcessedImage
from AgentQMS.vlm.core.interfaces import BackendError


class DashScopeBackend(BaseVLMBackend):
    """Alibaba DashScope API backend for Qwen VLM analysis."""

    def __init__(self, config: BackendConfig):
        """Initialize DashScope backend.

        Args:
            config: Backend configuration with DashScope API key
        """
        super().__init__(config)
        if dashscope is None:
            raise BackendError(
                "dashscope package is required for DashScope backend. "
                "Install with: pip install dashscope"
            )

        settings = get_config().backends.dashscope
        api_key = config.api_key or resolve_env_value(settings.api_key_env)
        if not api_key:
            raise BackendError(
                "DashScope API key is required. "
                "Provide it via DASHSCOPE_API_KEY environment variable."
            )

        # Set API key globally for dashscope SDK
        dashscope.api_key = api_key
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
        """Analyze an image using DashScope API.

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
                # Prepare messages for DashScope API
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:image/jpeg;base64,{image_data.base64_encoded}"},
                            {"text": prompt},
                        ],
                    }
                ]

                # Call DashScope API
                response = MultiModalConversation.call(
                    model=self.model,
                    messages=messages,
                )

                # Check response status
                if response.status_code == 200:
                    # Extract text from response
                    output = response.output
                    if output and "choices" in output:
                        choices = output["choices"]
                        if choices and len(choices) > 0:
                            message = choices[0].get("message", {})
                            content = message.get("content", "")
                            if isinstance(content, list):
                                # Content is a list of parts, extract text parts
                                text_parts = [
                                    part.get("text", "")
                                    for part in content
                                    if isinstance(part, dict) and "text" in part
                                ]
                                return "\n".join(text_parts)
                            elif isinstance(content, str):
                                return content
                            else:
                                raise BackendError(f"Unexpected content format: {type(content)}")
                        else:
                            raise BackendError("No choices in response")
                    else:
                        raise BackendError("No output in response")
                else:
                    error_msg = response.message or "Unknown error"
                    raise BackendError(f"DashScope API error: {error_msg} (status: {response.status_code})")

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
