"""Backward compatibility shim for ocr.validation.models.

DEPRECATED: All models moved to ocr.core.validation.
This module will remain indefinitely for compatibility.
"""

import warnings

warnings.warn(
    "ocr.validation.models is deprecated. Import from ocr.core.validation instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all classes
from ocr.core.validation import (
    VALID_EXIF_ORIENTATIONS,
    BatchSample,
    CollateOutput,
    DatasetSample,
    LightningStepPrediction,
    MetricConfig,
    ModelOutput,
    PolygonArray,
    TransformOutput,
    ValidatedTensorData,
    validate_predictions,
    validator,
)

__all__ = [
    "VALID_EXIF_ORIENTATIONS",
    "BatchSample",
    "CollateOutput",
    "DatasetSample",
    "LightningStepPrediction",
    "MetricConfig",
    "ModelOutput",
    "PolygonArray",
    "TransformOutput",
    "ValidatedTensorData",
    "validate_predictions",
    "validator",
]
