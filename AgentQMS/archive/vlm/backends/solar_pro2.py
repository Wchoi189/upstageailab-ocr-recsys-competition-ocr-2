"""Solar Pro 2 API Backend Implementation."""

import time
from typing import TYPE_CHECKING, Any

try:
    import httpx

    _httpx = httpx
except ImportError:
    _httpx = None  # type: ignore[assignment]

if TYPE_CHECKING:
    pass

from AgentQMS.vlm.backends.base import BaseVLMBackend
from AgentQMS.vlm.core.config import get_config, resolve_env_value
from AgentQMS.vlm.core.contracts import AnalysisMode, BackendConfig, ProcessedImage
from AgentQMS.vlm.core.interfaces import BackendError


class SolarPro2Backend(BaseVLMBackend):
    """Solar Pro 2 API backend for VLM analysis."""

    def __init__(self, config: BackendConfig):
        """Initialize Solar Pro 2 backend.

        Args:
            config: Backend configuration with Solar Pro 2 API key
        """
        super().__init__(config)
        if _httpx is None:
            raise BackendError("httpx package is required for Solar Pro 2 backend. Install with: pip install httpx")

        settings = get_config().backends.solar_pro2
        api_key = config.api_key or resolve_env_value(settings.api_key_env)
        if not api_key:
            raise BackendError("Solar Pro 2 API key is required. Provide it via configuration or environment variables.")

        self.api_key = api_key
        self.endpoint = config.endpoint or settings.endpoint
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
        """Analyze an image using Solar Pro 2 API.

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
            raise BackendError("Image must be base64 encoded for Solar Pro 2 API")

        max_retries = self.config.max_retries
        last_error = None

        payload = {
            "model": self.model,
            "prompt": prompt,
            "image": {
                "format": image_data.format.value.lower(),
                "data": image_data.base64_encoded,
            },
            "mode": mode.value,
        }

        for attempt in range(max_retries + 1):
            try:
                with _httpx.Client(timeout=self.config.timeout_seconds) as client:
                    response = client.post(
                        self.endpoint,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    response.raise_for_status()
                    result = response.json()

                    if "analysis" in result:
                        return result["analysis"].strip()
                    elif "text" in result:
                        return result["text"].strip()
                    else:
                        raise BackendError(f"Unexpected response format: {result}")

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(2**attempt)  # Exponential backoff
                    continue
                break

        raise BackendError(f"Failed to analyze image after {max_retries + 1} attempts: {last_error}") from last_error

    def supports_batch(self) -> bool:
        """Solar Pro 2 supports batch processing via multiple API calls."""
        return True

    def is_available(self) -> bool:
        """Check if Solar Pro 2 backend is available."""
        if _httpx is None:
            return False

        settings = get_config().backends.solar_pro2
        api_key = self.config.api_key or resolve_env_value(settings.api_key_env)
        return api_key is not None
