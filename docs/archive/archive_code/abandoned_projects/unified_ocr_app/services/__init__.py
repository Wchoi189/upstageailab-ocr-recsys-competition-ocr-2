"""Service layer for unified OCR app.

Services handle business logic and are independent of UI framework.
This module also provides cached service factories for performance optimization.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from .config_loader import ConfigLoader, load_unified_config
from .inference_service import InferenceService
from .preprocessing_service import PreprocessingService

# ============================================================================
# Cached Service Factories
# ============================================================================


@st.cache_resource(show_spinner=False)
def get_preprocessing_service(mode_config: dict[str, Any]) -> PreprocessingService:
    """Get cached PreprocessingService instance.

    Args:
        mode_config: Mode configuration dictionary

    Returns:
        Cached PreprocessingService instance
    """
    return PreprocessingService(mode_config)


@st.cache_resource(show_spinner=False)
def get_inference_service(mode_config: dict[str, Any]) -> InferenceService:
    """Get cached InferenceService instance.

    Args:
        mode_config: Mode configuration dictionary

    Returns:
        Cached InferenceService instance
    """
    return InferenceService(mode_config)


__all__ = [
    "ConfigLoader",
    "load_unified_config",
    "get_preprocessing_service",
    "get_inference_service",
    "PreprocessingService",
    "InferenceService",
]
