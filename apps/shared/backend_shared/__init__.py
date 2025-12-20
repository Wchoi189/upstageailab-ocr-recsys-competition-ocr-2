"""Shared backend components for domain-specific FastAPI applications.

This package provides common inference functionality used by OCR Console and
Playground Console backends.

Primary exports:
    - InferenceEngine: OCR inference engine for real-time predictions
    - Pydantic models: Type-safe request/response models

Usage:
    from apps.shared.backend_shared.inference import InferenceEngine
    from apps.shared.backend_shared.models.inference import (
        InferenceRequest,
        InferenceResponse,
    )

See: docs/artifacts/specs/shared-backend-contract.md
"""

# from .inference import InferenceEngine
#
# __all__ = [
#     "InferenceEngine",
# ]

__version__ = "1.0.0"
