"""Perspective correction utilities for OCR images.

This module provides mask-based rectangle fitting and perspective transformation.
"""

from .core import (
    calculate_target_dimensions,
    correct_perspective_from_mask,
    four_point_transform,
    remove_background_and_mask,
    transform_polygons_inverse,
)
from .fitting import fit_mask_rectangle
from .types import LineQualityReport, MaskRectangleResult

__all__ = [
    # Types
    "LineQualityReport",
    "MaskRectangleResult",
    # Core functions
    "calculate_target_dimensions",
    "four_point_transform",
    "correct_perspective_from_mask",
    "remove_background_and_mask",
    "transform_polygons_inverse",
    # Fitting
    "fit_mask_rectangle",
]
