"""
Legacy import path compatibility shim.

This module provides backward compatibility for code importing from the old path:
    from ocr.core.inference.engine import InferenceEngine

The actual implementation has been moved to:
    ocr.pipelines.engine

This shim will be deprecated in a future release.
"""

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
