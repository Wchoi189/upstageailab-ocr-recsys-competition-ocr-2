"""Image preprocessing and validation service."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Service for image decoding and validation."""

    @staticmethod
    def decode_base64_image(image_base64: str) -> np.ndarray:
        """Decode base64-encoded image to numpy array.

        Args:
            image_base64: Base64-encoded image string (with or without data URL prefix)

        Returns:
            Decoded image as numpy array (BGR format)

        Raises:
            ValueError: If image decoding fails
        """
        import cv2
        import numpy as np

        if not image_base64:
            raise ValueError("image_base64 cannot be empty")

        try:
            # Handle data URL prefix (e.g., "data:image/jpeg;base64,...")
            if "base64," in image_base64:
                image_base64 = image_base64.split("base64,", 1)[1]

            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_base64)

            # Decode bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("cv2.imdecode returned None - invalid image data")

            return image

        except base64.binascii.Error as e:
            raise ValueError(f"Invalid base64 encoding: {str(e)}")
        except Exception as e:
            raise ValueError(f"Image decoding failed: {str(e)}")
