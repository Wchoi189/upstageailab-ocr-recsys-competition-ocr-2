"""\nDEPRECATED: 
Legacy import path compatibility shim.

This module provides backward compatibility for code importing from the old path:
    from ocr.core.utils.polygon_utils import *

The actual implementation has been moved to:
⚠️  WARNING: This compatibility shim will be REMOVED in v0.4.0
⚠️  Update your imports to use the new path

"""

import warnings

warnings.warn(
    "Importing from 'ocr.core.utils.polygon_utils' is deprecated. "
    "Use 'ocr.domains.detection.utils.polygons' instead. "
    "This compatibility shim will be removed in v0.4.0.",
    DeprecationWarning,
    stacklevel=2
)


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
