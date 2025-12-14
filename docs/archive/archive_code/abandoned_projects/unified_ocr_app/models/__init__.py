"""Data models for unified OCR app.

All models use Pydantic for validation and type safety.
"""

from .app_state import UnifiedAppState
from .preprocessing_config import PreprocessingConfig

__all__ = [
    "UnifiedAppState",
    "PreprocessingConfig",
]
