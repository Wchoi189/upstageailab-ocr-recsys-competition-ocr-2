"""
Legacy import path compatibility shim.

This module provides backward compatibility for code importing from the old path:
    from ocr.core.utils.geometry_utils import *

The actual implementation has been moved to:
    ocr.domains.detection.utils.geometry

This shim will be deprecated in a future release.
"""

from ocr.domains.detection.utils.geometry import (
    apply_padding_offset_to_polygons,
    calculate_cropbox,
    calculate_inverse_transform,
    compute_padding_offsets,
)

__all__ = [
    "apply_padding_offset_to_polygons",
    "calculate_cropbox",
    "calculate_inverse_transform",
    "compute_padding_offsets",
]
