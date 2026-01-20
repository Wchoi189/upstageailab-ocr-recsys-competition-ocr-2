"""
Legacy import path compatibility shim.

This module provides backward compatibility for code importing from the old path:
    from ocr.core.utils.polygon_utils import *

The actual implementation has been moved to:
    ocr.domains.detection.utils.polygons

This shim will be deprecated in a future release.
"""

from ocr.domains.detection.utils.polygons import (
    ensure_polygon_array,
    filter_degenerate_polygons,
    has_duplicate_consecutive_points,
    is_valid_polygon,
    validate_map_shapes,
    validate_polygon_area,
    validate_polygon_finite,
)

__all__ = [
    "ensure_polygon_array",
    "filter_degenerate_polygons",
    "has_duplicate_consecutive_points",
    "is_valid_polygon",
    "validate_map_shapes",
    "validate_polygon_area",
    "validate_polygon_finite",
]
