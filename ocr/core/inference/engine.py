"""\nDEPRECATED: 
Legacy import path compatibility shim.

This module provides backward compatibility for code importing from the old path:
    from ocr.core.inference.engine import InferenceEngine

The actual implementation has been moved to:
⚠️  WARNING: This compatibility shim will be REMOVED in v0.4.0
⚠️  Update your imports to use the new path

"""

import warnings

warnings.warn(
    "Importing from 'ocr.core.inference.engine' is deprecated. "
    "Use 'ocr.pipelines.engine' instead. "
    "This compatibility shim will be removed in v0.4.0.",
    DeprecationWarning,
    stacklevel=2
)


from ocr.pipelines.engine import (
    InferenceEngine,
    get_available_checkpoints,
    run_inference_on_image,
)

__all__ = [
    "InferenceEngine",
    "get_available_checkpoints",
    "run_inference_on_image",
]
