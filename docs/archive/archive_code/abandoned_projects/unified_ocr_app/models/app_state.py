"""Unified application state management.

Centralized state for all modes with type-safe access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import streamlit as st


@dataclass
class UnifiedAppState:
    """Unified application state across all modes.

    This class provides a type-safe interface to Streamlit's session_state.
    All state access should go through this class to ensure consistency.
    """

    # Current mode
    current_mode: str = "preprocessing"

    # Image state
    uploaded_images: list[np.ndarray] = field(default_factory=list)
    current_image_index: int = 0

    # Preprocessing state
    preprocessing_config: dict[str, Any] = field(default_factory=dict)
    preprocessing_results: dict[str, np.ndarray] = field(default_factory=dict)
    preprocessing_metadata: dict[str, Any] = field(default_factory=dict)

    # Inference state
    selected_checkpoint: str | None = None
    inference_results: list[dict[str, Any]] = field(default_factory=list)

    # Comparison state
    comparison_configs: list[dict[str, Any]] = field(default_factory=list)
    comparison_results: list[dict[str, Any]] = field(default_factory=list)

    # Cache
    cache: dict[str, Any] = field(default_factory=dict)

    # Preferences
    preferences: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_session(cls) -> UnifiedAppState:
        """Load state from Streamlit session_state.

        Returns:
            UnifiedAppState instance with current session state
        """
        if "unified_app_state" not in st.session_state:
            st.session_state.unified_app_state = cls()

        return st.session_state.unified_app_state

    def to_session(self) -> None:
        """Save state to Streamlit session_state."""
        st.session_state.unified_app_state = self

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.preprocessing_results.clear()
        self.inference_results.clear()
        self.comparison_results.clear()

    def set_mode(self, mode: str) -> None:
        """Change current mode and optionally clear cache.

        Args:
            mode: New mode ID ('preprocessing', 'inference', 'comparison')
        """
        if mode != self.current_mode:
            self.current_mode = mode
            # Optionally clear cache on mode switch
            # (controlled by config.shared.session.clear_cache_on_mode_switch)

    def get_current_image(self) -> np.ndarray | None:
        """Get currently selected image.

        Returns:
            Current image as numpy array, or None if no images uploaded
        """
        if not self.uploaded_images:
            return None

        if 0 <= self.current_image_index < len(self.uploaded_images):
            return self.uploaded_images[self.current_image_index]

        return None

    def add_image(self, image: np.ndarray) -> None:
        """Add image to uploaded images list.

        Args:
            image: Image as numpy array
        """
        self.uploaded_images.append(image)
        self.current_image_index = len(self.uploaded_images) - 1

    def clear_images(self) -> None:
        """Clear all uploaded images and reset index."""
        self.uploaded_images.clear()
        self.current_image_index = 0
        self.clear_cache()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference value.

        Args:
            key: Preference key
            default: Default value if key not found

        Returns:
            Preference value or default
        """
        return self.preferences.get(key, default)

    def set_preference(self, key: str, value: Any) -> None:
        """Set user preference value.

        Args:
            key: Preference key
            value: Preference value
        """
        self.preferences[key] = value
        self.to_session()  # Persist immediately

    def get_image_count(self) -> int:
        """Get the number of uploaded images.

        Returns:
            Number of images currently in the session
        """
        return len(self.uploaded_images)
