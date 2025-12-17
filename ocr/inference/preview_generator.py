from __future__ import annotations

"""Preview image generation utilities for OCR inference.

Handles encoding preview images and attaching them to prediction payloads with
proper coordinate transformation and metadata.
"""

import base64
import logging
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class PreviewGenerator:
    """Generates preview images for OCR inference results.

    Encodes preview images as base64-encoded JPEG and attaches them to
    prediction payloads along with metadata for coordinate transformation.
    """

    def __init__(self, jpeg_quality: int = 85):
        """Initialize preview generator.

        Args:
            jpeg_quality: JPEG encoding quality (0-100). Default 85 provides
                good quality while reducing file size by ~10x compared to PNG.
        """
        if not 0 <= jpeg_quality <= 100:
            raise ValueError(f"JPEG quality must be between 0 and 100, got {jpeg_quality}")

        self.jpeg_quality = jpeg_quality
        self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    def encode_preview_image(
        self,
        preview_image: np.ndarray,
        format: str = "jpg",
    ) -> str | None:
        """Encode preview image to base64 string.

        Args:
            preview_image: Preview image as numpy array (BGR format for cv2)
            format: Image format ('jpg' or 'png'). Default 'jpg'.

        Returns:
            Base64-encoded image string, or None if encoding fails

        Example:
            >>> generator = PreviewGenerator(jpeg_quality=85)
            >>> image = np.zeros((640, 640, 3), dtype=np.uint8)
            >>> base64_str = generator.encode_preview_image(image)
            >>> isinstance(base64_str, str)
            True
        """
        if preview_image is None or preview_image.size == 0:
            LOGGER.warning("Cannot encode empty or None preview image")
            return None

        # Validate format first (let ValueError propagate)
        if format.lower() == "jpg" or format.lower() == "jpeg":
            encode_params = self._encode_params
            extension = ".jpg"
        elif format.lower() == "png":
            encode_params = []
            extension = ".png"
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'jpg' or 'png'.")

        try:
            success, buffer = cv2.imencode(extension, preview_image, encode_params)

            if not success:
                LOGGER.warning(f"cv2.imencode failed for format {format}")
                return None

            base64_str = base64.b64encode(buffer).decode("ascii")
            return base64_str

        except Exception as e:
            LOGGER.exception(f"Failed to encode preview image: {e}")
            return None

    def attach_preview_to_payload(
        self,
        payload: dict[str, Any],
        preview_image: np.ndarray,
        metadata: dict[str, Any] | None = None,
        transform_polygons: bool = True,
        original_shape: tuple[int, int] | None = None,
        target_size: int = 640,
    ) -> dict[str, Any]:
        """Attach encoded preview image and metadata to prediction payload.

        Args:
            payload: Prediction payload dictionary
            preview_image: Preview image to encode and attach
            metadata: Preprocessing metadata to attach (optional)
            transform_polygons: Whether to transform polygons to preview space
            original_shape: Original image shape (height, width) for polygon transformation
            target_size: Target size for coordinate transformation

        Returns:
            Updated payload with preview image and metadata

        Note:
            BUG-001: Using JPEG instead of PNG reduces file size by ~10x while
            maintaining acceptable quality for visualization.
        """
        if not isinstance(payload, dict):
            LOGGER.warning("Payload is not a dict, cannot attach preview")
            return payload

        # Create a copy to avoid modifying the input
        result = dict(payload)

        try:
            # Transform polygons to preview space if requested
            if transform_polygons and original_shape is not None:
                result = self._transform_polygons_to_preview_space(
                    result,
                    original_shape=original_shape,
                    target_size=target_size,
                )

            # Encode and attach preview image
            base64_str = self.encode_preview_image(preview_image, format="jpg")
            if base64_str is not None:
                result["preview_image_base64"] = base64_str

            # Attach metadata if available
            if metadata is not None:
                result["meta"] = metadata
                LOGGER.debug(
                    "Attached meta to preview response: original_size=%s, processed_size=%s, "
                    "coordinate_system=%s",
                    metadata.get("original_size"),
                    metadata.get("processed_size"),
                    metadata.get("coordinate_system"),
                )
            else:
                LOGGER.warning(
                    "Metadata is None when attaching preview - coordinate system contract may be incomplete. "
                    "This may cause frontend to fall back to heuristic normalization."
                )

        except Exception as e:
            LOGGER.exception(f"Failed to attach preview to payload: {e}")
            # Even if encoding fails, try to attach metadata if available
            if metadata is not None:
                result["meta"] = metadata

        return result

    def _transform_polygons_to_preview_space(
        self,
        payload: dict[str, Any],
        original_shape: tuple[int, int],
        target_size: int = 640,
    ) -> dict[str, Any]:
        """Transform polygon coordinates from original space to preview space.

        Args:
            payload: Payload containing polygons in original image space
            original_shape: Original image shape (height, width)
            target_size: Target size for processed image

        Returns:
            Updated payload with transformed polygons
        """
        if not isinstance(payload, dict) or not payload.get("polygons"):
            return payload

        polygons_str = payload["polygons"]
        if not polygons_str:
            return payload

        try:
            from .coordinate_manager import transform_polygons_string_to_processed_space

            transformed_polygons_str = transform_polygons_string_to_processed_space(
                polygons_str,
                original_shape=original_shape,
                target_size=target_size,
                tolerance=2.0,
            )

            result = dict(payload)
            result["polygons"] = transformed_polygons_str
            return result

        except Exception as e:
            LOGGER.exception(f"Failed to transform polygons to preview space: {e}")
            return payload


def create_preview_with_metadata(
    payload: dict[str, Any],
    preview_image: np.ndarray,
    metadata: dict[str, Any] | None = None,
    original_shape: tuple[int, int] | None = None,
    target_size: int = 640,
    jpeg_quality: int = 85,
) -> dict[str, Any]:
    """Convenience function to create preview with metadata.

    Args:
        payload: Prediction payload
        preview_image: Preview image to encode
        metadata: Preprocessing metadata
        original_shape: Original image shape (height, width)
        target_size: Target size for coordinate transformation
        jpeg_quality: JPEG encoding quality (0-100)

    Returns:
        Updated payload with preview and metadata

    Example:
        >>> import numpy as np
        >>> payload = {"polygons": "0 0 100 0 100 100 0 100", "texts": ["Text"]}
        >>> preview = np.zeros((640, 640, 3), dtype=np.uint8)
        >>> meta = {"original_size": (640, 640), "processed_size": (640, 640)}
        >>> result = create_preview_with_metadata(payload, preview, meta, (640, 640))
        >>> "preview_image_base64" in result
        True
    """
    generator = PreviewGenerator(jpeg_quality=jpeg_quality)
    return generator.attach_preview_to_payload(
        payload=payload,
        preview_image=preview_image,
        metadata=metadata,
        transform_polygons=True,
        original_shape=original_shape,
        target_size=target_size,
    )
