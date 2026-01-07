"""Backward compatibility shim for ocr.data.datasets.schemas.

DEPRECATED: All schemas moved to ocr.core.validation.
This module will remain indefinitely for compatibility.
"""

import warnings

warnings.warn("ocr.data.datasets.schemas is deprecated. Import from ocr.core.validation instead.", DeprecationWarning, stacklevel=2)

# Re-export all classes
from ocr.core.validation import (
    CacheConfig,
    DataItem,
    DatasetConfig,
    ImageData,
    ImageLoadingConfig,
    ImageMetadata,
    MapData,
    PolygonData,
    TransformConfig,
    TransformInput,
    TransformOutput,
    ValidatedPolygonData,
)

__all__ = [
    "CacheConfig",
    "DataItem",
    "DatasetConfig",
    "ImageData",
    "ImageLoadingConfig",
    "ImageMetadata",
    "MapData",
    "PolygonData",
    "TransformConfig",
    "TransformInput",
    "TransformOutput",
    "ValidatedPolygonData",
]
