"""Inference module - re-exports InferenceEngine from ocr.core.inference.

This module provides a stable import path for the proven InferenceEngine
implementation from the ocr package.
"""

from ocr.core.inference.engine import InferenceEngine

__all__ = [
    "InferenceEngine",
]
