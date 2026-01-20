"""
DEPRECATED: Legacy import path compatibility shim.

This module provides backward compatibility for code importing from the old path:
    from ocr.core.utils.geometry_utils import *

The actual implementation has been moved to:
    ocr.domains.detection.utils.geometry

⚠️  WARNING: This compatibility shim will be REMOVED in v0.4.0
⚠️  Update your imports to use the new path

New import (use this):
    from ocr.domains.detection.utils.geometry import calculate_cropbox, ...
"""

import warnings

warnings.warn(
    "Importing from 'ocr.core.utils.geometry_utils' is deprecated. "
    "Use 'ocr.domains.detection.utils.geometry' instead. "
    "This compatibility shim will be removed in v0.4.0.",
    DeprecationWarning,
    stacklevel=2
)

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
