"""Crop extraction module for OCR recognition pipeline.

This module provides the CropExtractor class for extracting and rectifying
text crops from detected polygon regions, preparing them for text recognition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)


@dataclass
class CropConfig:
    """Configuration for crop extraction.

    Attributes:
        target_height: Target height for normalized crops (standard: 32)
        min_width: Minimum width for valid crops
        max_aspect_ratio: Maximum width/height aspect ratio
        padding_ratio: Padding ratio around detected region (0.0-0.5)
        enable_perspective_correction: Whether to apply perspective correction
    """

    target_height: int = 32
    min_width: int = 8
    max_aspect_ratio: float = 25.0
    padding_ratio: float = 0.1
    enable_perspective_correction: bool = True


@dataclass
class CropResult:
    """Result from crop extraction.

    Attributes:
        crop: Extracted and rectified crop as numpy array (H, W, C)
        original_polygon: Original polygon coordinates
        success: Whether extraction was successful
        error_message: Error message if extraction failed
    """

    crop: np.ndarray | None
    original_polygon: np.ndarray
    success: bool
    error_message: str | None = None


class CropExtractor:
    """Extract rectified text crops from detected polygon regions.

    This class provides methods to extract text regions from images,
    applying perspective correction to produce normalized horizontal
    text crops suitable for recognition.

    Example:
        >>> extractor = CropExtractor()
        >>> crops = extractor.extract_crops(image, polygons)
    """

    def __init__(self, config: CropConfig | None = None) -> None:
        """Initialize crop extractor.

        Args:
            config: Crop extraction configuration
        """
        self.config = config or CropConfig()
        LOGGER.info(
            "Initialized CropExtractor | target_height=%d | perspective=%s",
            self.config.target_height,
            self.config.enable_perspective_correction,
        )

    def extract_crops(
        self,
        image: np.ndarray,
        polygons: list[np.ndarray],
        target_height: int | None = None,
    ) -> list[CropResult]:
        """Extract perspective-corrected text crops from detected regions.

        Args:
            image: Source image as numpy array (H, W, C) in BGR format
            polygons: List of Nx2 polygon coordinate arrays
            target_height: Override target height (uses config default if None)

        Returns:
            List of CropResult objects with extracted crops
        """
        if image.ndim != 3:
            raise ValueError(f"Image must be 3D (H, W, C), got shape {image.shape}")

        height = target_height or self.config.target_height
        results: list[CropResult] = []

        for polygon in polygons:
            try:
                result = self._extract_single_crop(image, polygon, height)
                results.append(result)
            except Exception as e:
                LOGGER.warning("Failed to extract crop: %s", e)
                results.append(
                    CropResult(
                        crop=None,
                        original_polygon=polygon,
                        success=False,
                        error_message=str(e),
                    )
                )

        successful = sum(1 for r in results if r.success)
        LOGGER.debug(
            "Extracted %d/%d crops successfully",
            successful,
            len(polygons),
        )

        return results

    def _extract_single_crop(
        self,
        image: np.ndarray,
        polygon: np.ndarray,
        target_height: int,
    ) -> CropResult:
        """Extract a single crop from the image.

        Args:
            image: Source image
            polygon: Polygon coordinates (Nx2 array)
            target_height: Target height for the crop

        Returns:
            CropResult with extracted crop
        """
        polygon = np.array(polygon, dtype=np.float32)

        if polygon.ndim == 1:
            # Flatten format: [x1, y1, x2, y2, ...]
            polygon = polygon.reshape(-1, 2)

        if len(polygon) < 3:
            return CropResult(
                crop=None,
                original_polygon=polygon,
                success=False,
                error_message=f"Polygon has too few points: {len(polygon)}",
            )

        if self.config.enable_perspective_correction and len(polygon) == 4:
            crop = self._perspective_crop(image, polygon, target_height)
        else:
            crop = self._bounding_box_crop(image, polygon, target_height)

        if crop is None or crop.size == 0:
            return CropResult(
                crop=None,
                original_polygon=polygon,
                success=False,
                error_message="Extracted crop is empty",
            )

        # Validate aspect ratio
        h, w = crop.shape[:2]
        if w < self.config.min_width:
            return CropResult(
                crop=None,
                original_polygon=polygon,
                success=False,
                error_message=f"Crop too narrow: {w}px",
            )

        aspect_ratio = w / h if h > 0 else float("inf")
        if aspect_ratio > self.config.max_aspect_ratio:
            return CropResult(
                crop=None,
                original_polygon=polygon,
                success=False,
                error_message=f"Aspect ratio too large: {aspect_ratio:.1f}",
            )

        return CropResult(
            crop=crop,
            original_polygon=polygon,
            success=True,
            error_message=None,
        )

    def _perspective_crop(
        self,
        image: np.ndarray,
        polygon: np.ndarray,
        target_height: int,
    ) -> np.ndarray | None:
        """Extract crop using perspective transformation.

        Orders the 4 polygon points and applies perspective warp
        to produce a horizontal text crop.

        Args:
            image: Source image
            polygon: 4-point polygon coordinates
            target_height: Target height for output

        Returns:
            Rectified crop or None if transformation fails
        """
        try:
            # Order points: top-left, top-right, bottom-right, bottom-left
            ordered_pts = self._order_points(polygon)

            # Calculate output dimensions
            width_top = np.linalg.norm(ordered_pts[1] - ordered_pts[0])
            width_bottom = np.linalg.norm(ordered_pts[2] - ordered_pts[3])
            width = int(max(width_top, width_bottom))

            height_left = np.linalg.norm(ordered_pts[3] - ordered_pts[0])
            height_right = np.linalg.norm(ordered_pts[2] - ordered_pts[1])
            height = int(max(height_left, height_right))

            if width <= 0 or height <= 0:
                return None

            # Calculate target width maintaining aspect ratio
            aspect_ratio = width / height
            target_width = int(target_height * aspect_ratio)

            if target_width <= 0:
                return None

            # Define destination points
            dst_pts = np.array(
                [
                    [0, 0],
                    [target_width - 1, 0],
                    [target_width - 1, target_height - 1],
                    [0, target_height - 1],
                ],
                dtype=np.float32,
            )

            # Compute and apply perspective transform
            matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
            crop = cv2.warpPerspective(
                image,
                matrix,
                (target_width, target_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

            return crop

        except Exception as e:
            LOGGER.debug("Perspective crop failed: %s", e)
            return None

    def _bounding_box_crop(
        self,
        image: np.ndarray,
        polygon: np.ndarray,
        target_height: int,
    ) -> np.ndarray | None:
        """Extract crop using axis-aligned bounding box.

        Fallback method when perspective correction is not possible.

        Args:
            image: Source image
            polygon: Polygon coordinates
            target_height: Target height for output

        Returns:
            Resized crop or None if extraction fails
        """
        try:
            img_h, img_w = image.shape[:2]

            # Get bounding box with padding
            x_min = max(0, int(polygon[:, 0].min()))
            y_min = max(0, int(polygon[:, 1].min()))
            x_max = min(img_w, int(polygon[:, 0].max()))
            y_max = min(img_h, int(polygon[:, 1].max()))

            if x_max <= x_min or y_max <= y_min:
                return None

            # Add padding
            pad_x = int((x_max - x_min) * self.config.padding_ratio)
            pad_y = int((y_max - y_min) * self.config.padding_ratio)

            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(img_w, x_max + pad_x)
            y_max = min(img_h, y_max + pad_y)

            # Extract and resize
            crop = image[y_min:y_max, x_min:x_max]

            if crop.size == 0:
                return None

            h, w = crop.shape[:2]
            aspect_ratio = w / h
            target_width = int(target_height * aspect_ratio)

            if target_width <= 0:
                return None

            crop = cv2.resize(
                crop,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )

            return crop

        except Exception as e:
            LOGGER.debug("Bounding box crop failed: %s", e)
            return None

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left.

        Args:
            pts: 4x2 array of points

        Returns:
            Ordered 4x2 array of points
        """
        pts = pts.astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum and diff to find corners
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()

        rect[0] = pts[np.argmin(s)]  # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has largest sum
        rect[1] = pts[np.argmin(d)]  # Top-right has smallest difference
        rect[3] = pts[np.argmax(d)]  # Bottom-left has largest difference

        return rect


__all__ = [
    "CropConfig",
    "CropResult",
    "CropExtractor",
]
