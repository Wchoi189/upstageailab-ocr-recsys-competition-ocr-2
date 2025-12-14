"""Pydantic models for shared backend API contracts.

This module exports type-safe request/response models used by domain-specific
backends for inference operations.
"""

from .inference import (
    InferenceMetadata,
    InferenceRequest,
    InferenceResponse,
    Padding,
    TextRegion,
)

__all__ = [
    "InferenceMetadata",
    "InferenceRequest",
    "InferenceResponse",
    "Padding",
    "TextRegion",
]
