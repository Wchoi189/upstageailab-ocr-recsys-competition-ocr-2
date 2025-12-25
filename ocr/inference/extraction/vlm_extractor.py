"""VLM-based receipt extraction using Qwen2.5-VL via vLLM.

This module provides vision-language model based extraction for complex
receipts that fall below confidence thresholds in rule-based extraction.
"""

from __future__ import annotations

import logging
import httpx
from dataclasses import dataclass
from PIL import Image
import base64
import io

from .receipt_schema import ReceiptData

LOGGER = logging.getLogger(__name__)


@dataclass
class VLMExtractorConfig:
    """Configuration for VLM extraction.

    Attributes:
        server_url: vLLM server URL
        model: Model name/path on vLLM server
        timeout: Request timeout in seconds
        max_tokens: Maximum tokens in response
    """

    server_url: str = "http://localhost:8001"
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    timeout: float = 5.0
    max_tokens: int = 2048


class VLMExtractor:
    """Extract receipt data using Qwen2.5-VL vision-language model.

    This extractor sends receipt images to a vLLM server running Qwen2.5-VL
    and parses the JSON response into structured ReceiptData.

    Example:
        >>> extractor = VLMExtractor()
        >>> if extractor.is_server_healthy():
        ...     receipt = extractor.extract(pil_image)
    """

    EXTRACTION_PROMPT = '''Extract the following fields from this receipt image as JSON:
{
    "store_name": "...",
    "store_address": "...",
    "transaction_date": "YYYY-MM-DD",
    "transaction_time": "HH:MM",
    "items": [{"name": "...", "quantity": 1, "total_price": 0.00}],
    "subtotal": 0.00,
    "tax": 0.00,
    "total": 0.00,
    "payment_method": "card|cash",
    "card_last_four": "1234"
}
Return ONLY valid JSON, no explanation.'''

    def __init__(self, config: VLMExtractorConfig | None = None):
        """Initialize VLM extractor.

        Args:
            config: VLM configuration (uses defaults if None)
        """
        self.config = config or VLMExtractorConfig()
        self._client = httpx.Client(timeout=self.config.timeout)
        LOGGER.info("VLMExtractor initialized | server=%s", self.config.server_url)

    def extract(self, image: Image.Image, ocr_context: str = "") -> ReceiptData:
        """Extract receipt data from image using VLM.

        Args:
            image: PIL Image of receipt
            ocr_context: Optional OCR text for context

        Returns:
            ReceiptData with extracted fields

        Raises:
            httpx.HTTPError: If server request fails
            json.JSONDecodeError: If response is not valid JSON
        """
        # Encode image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Build prompt with OCR context
        prompt = self.EXTRACTION_PROMPT
        if ocr_context:
            prompt = f"OCR Text:\n{ocr_context}\n\n{prompt}"

        # Call vLLM server
        response = self._client.post(
            f"{self.config.server_url}/v1/chat/completions",
            json={
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": self.config.max_tokens,
            },
        )
        response.raise_for_status()

        # Parse response
        result = response.json()
        json_str = result["choices"][0]["message"]["content"]

        # Parse JSON string into ReceiptData
        return ReceiptData.model_validate_json(json_str)

    def is_server_healthy(self) -> bool:
        """Check if vLLM server is responding.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self._client.get(f"{self.config.server_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def __del__(self):
        """Clean up HTTP client."""
        try:
            self._client.close()
        except Exception:
            pass


__all__ = [
    "VLMExtractor",
    "VLMExtractorConfig",
]
