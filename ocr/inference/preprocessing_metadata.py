from __future__ import annotations

"""Preprocessing metadata utilities for OCR inference.

Provides convenience functions for creating preprocessing metadata dictionaries
used throughout the inference pipeline. Builds on coordinate_manager for
consistent transformation calculations.
"""

import logging
from collections.abc import Sequence
from typing import Any

from .coordinate_manager import calculate_transform_metadata

LOGGER = logging.getLogger(__name__)


def create_preprocessing_metadata(
    original_shape: Sequence[int],
    target_size: int = 640,
) -> dict[str, Any]:
    """Create preprocessing metadata dictionary for inference pipeline.

    Provides metadata about image transformation including original size,
    processed size, padding, content area, and scale. This metadata is used
    for coordinate transformations and preview image generation.

    Args:
        original_shape: Original image shape (height, width) or (height, width, channels)
        target_size: Target size for square processed image (default: 640)

    Returns:
        Dictionary with preprocessing metadata:
        - original_size: (width, height) tuple
        - processed_size: (width, height) tuple
        - padding: dict with top, bottom, left, right padding pixels
        - padding_position: "top_left" (content starts at 0,0)
        - content_area: (width, height) of resized content before padding
        - scale: resize scale factor
        - coordinate_system: "pixel" (absolute pixel coordinates)

    Example:
        >>> meta = create_preprocessing_metadata((800, 600), target_size=640)
        >>> meta['original_size']
        (600, 800)
        >>> meta['content_area']
        (480, 640)
        >>> meta['padding']
        {'top': 0, 'bottom': 0, 'left': 0, 'right': 160}
    """
    try:
        transform_meta = calculate_transform_metadata(original_shape, target_size)
    except ValueError as e:
        LOGGER.error(f"Failed to calculate transform metadata: {e}")
        raise

    # Build metadata dictionary following inference pipeline contract
    metadata = {
        "original_size": (transform_meta.original_w, transform_meta.original_h),
        "processed_size": (target_size, target_size),
        "padding": {
            "top": 0,  # Top-left padding: no top padding
            "bottom": transform_meta.pad_h,
            "left": 0,  # Top-left padding: no left padding
            "right": transform_meta.pad_w,
        },
        "padding_position": "top_left",  # Content starts at (0,0), padding at bottom/right
        "content_area": (transform_meta.resized_w, transform_meta.resized_h),
        "scale": float(transform_meta.scale),
        "coordinate_system": "pixel",  # Absolute pixels in processed frame [0-target_size]
    }

    LOGGER.debug(
        "Created preprocessing metadata: original_size=%s, processed_size=%s, padding=%s, content_area=%s, scale=%.4f",
        metadata["original_size"],
        metadata["processed_size"],
        metadata["padding"],
        metadata["content_area"],
        metadata["scale"],
    )

    return metadata


def calculate_resize_dimensions(
    original_shape: Sequence[int],
    target_size: int = 640,
) -> tuple[int, int, float]:
    """Calculate resized dimensions after LongestMaxSize transformation.

    Args:
        original_shape: Original image shape (height, width) or (height, width, channels)
        target_size: Target size for longest side (default: 640)

    Returns:
        Tuple of (resized_height, resized_width, scale)

    Example:
        >>> calculate_resize_dimensions((800, 600), target_size=640)
        (640, 480, 0.8)
    """
    try:
        transform_meta = calculate_transform_metadata(original_shape, target_size)
    except ValueError as e:
        LOGGER.error(f"Failed to calculate resize dimensions: {e}")
        raise

    return (transform_meta.resized_h, transform_meta.resized_w, transform_meta.scale)


def calculate_padding(
    original_shape: Sequence[int],
    target_size: int = 640,
) -> tuple[int, int]:
    """Calculate padding needed to reach target square size.

    With top_left padding position, padding is added to bottom and right only.

    Args:
        original_shape: Original image shape (height, width) or (height, width, channels)
        target_size: Target size for square processed image (default: 640)

    Returns:
        Tuple of (pad_height, pad_width) for bottom and right padding

    Example:
        >>> calculate_padding((800, 600), target_size=640)
        (0, 160)  # No height padding, 160 pixels width padding on right
    """
    try:
        transform_meta = calculate_transform_metadata(original_shape, target_size)
    except ValueError as e:
        LOGGER.error(f"Failed to calculate padding: {e}")
        raise

    return (transform_meta.pad_h, transform_meta.pad_w)


def get_content_area(
    original_shape: Sequence[int],
    target_size: int = 640,
) -> tuple[int, int]:
    """Get content area dimensions (resized size before padding).

    Args:
        original_shape: Original image shape (height, width) or (height, width, channels)
        target_size: Target size for processed image (default: 640)

    Returns:
        Tuple of (content_width, content_height)

    Example:
        >>> get_content_area((800, 600), target_size=640)
        (480, 640)  # Content is 480x640, with 160px padding on right
    """
    try:
        transform_meta = calculate_transform_metadata(original_shape, target_size)
    except ValueError as e:
        LOGGER.error(f"Failed to get content area: {e}")
        raise

    return (transform_meta.resized_w, transform_meta.resized_h)
