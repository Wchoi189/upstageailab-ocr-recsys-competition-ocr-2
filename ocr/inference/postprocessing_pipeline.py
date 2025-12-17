"""Postprocessing pipeline for OCR inference.

This module consolidates postprocessing logic into a focused pipeline that:
1. Decodes predictions from model outputs
2. Transforms coordinates to original image space
3. Validates and filters detections
4. Formats output in competition format

The pipeline ensures consistent coordinate transformations and provides
a clean API for testing and extension.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config_loader import PostprocessSettings
from .postprocess import decode_polygons_with_head, fallback_postprocess

LOGGER = logging.getLogger(__name__)


@dataclass
class PostprocessingResult:
    """Result of postprocessing pipeline.

    Attributes:
        polygons: Pipe-separated list of space-separated polygon coordinates
        texts: List of text labels for each detection
        confidences: List of confidence scores for each detection
        method: Method used for postprocessing ("head" or "fallback")
    """

    polygons: str
    texts: list[str]
    confidences: list[float]
    method: str


class PostprocessingPipeline:
    """Postprocessing pipeline for OCR inference.

    This pipeline encapsulates all postprocessing steps:
    1. Decode predictions using model head (primary method)
    2. Fallback to contour-based detection if head decoding fails
    3. Transform coordinates to original image space
    4. Filter detections by size and confidence
    5. Format output in competition format

    The pipeline maintains coordinate consistency with preprocessing
    and provides structured, testable output.
    """

    def __init__(
        self,
        settings: PostprocessSettings | None = None,
    ):
        """Initialize postprocessing pipeline.

        Args:
            settings: Postprocessing settings (thresholds, limits, etc.)
        """
        self._settings = settings

    def process(
        self,
        model: Any,
        processed_tensor: Any,
        predictions: dict[str, Any],
        original_shape: tuple[int, int, int],
    ) -> PostprocessingResult | None:
        """Run postprocessing pipeline on model predictions.

        Attempts to decode using model head first, then falls back to
        contour-based detection if head decoding fails.

        Args:
            model: Model instance with optional head
            processed_tensor: Processed tensor input to model (1, C, H, W)
            predictions: Model predictions dict
            original_shape: Original image shape before preprocessing (H, W, C)

        Returns:
            PostprocessingResult with formatted detections,
            or None if postprocessing fails

        Example:
            >>> pipeline = PostprocessingPipeline(settings=my_settings)
            >>> result = pipeline.process(model, batch, predictions, original_shape)
            >>> if result:
            ...     print(f"Found {len(result.texts)} detections")
            ...     print(f"Method: {result.method}")
        """
        # Stage 1: Try head-based decoding (primary method)
        decoded = decode_polygons_with_head(
            model,
            processed_tensor,
            predictions,
            original_shape,
        )

        if decoded is not None:
            LOGGER.debug("Primary head-based decoding successful")
            return PostprocessingResult(
                polygons=decoded.get("polygons", ""),
                texts=decoded.get("texts", []),
                confidences=decoded.get("confidences", []),
                method="head",
            )

        # Stage 2: Fallback to contour-based postprocessing
        if self._settings is None:
            LOGGER.error("Settings not configured for fallback postprocessing")
            return None

        try:
            LOGGER.debug("Head decoding failed, trying fallback postprocessing")
            result = fallback_postprocess(
                predictions,
                original_shape,
                self._settings,
            )

            return PostprocessingResult(
                polygons=result.get("polygons", ""),
                texts=result.get("texts", []),
                confidences=result.get("confidences", []),
                method="fallback",
            )

        except Exception:
            LOGGER.exception("Fallback postprocessing failed")
            return None

    def set_settings(self, settings: PostprocessSettings) -> None:
        """Set postprocessing settings.

        Args:
            settings: Postprocessing settings
        """
        self._settings = settings


__all__ = [
    "PostprocessingPipeline",
    "PostprocessingResult",
]
