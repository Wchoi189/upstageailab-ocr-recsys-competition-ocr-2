"""Shared image upload component.

Provides image upload widget with configuration-driven behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from ...models.app_state import UnifiedAppState

logger = logging.getLogger(__name__)


def render_image_upload(state: UnifiedAppState, config: dict[str, Any]) -> None:
    """Render image upload widget with configuration-driven settings.

    Args:
        state: Application state
        config: Upload configuration from YAML (config.shared.upload)
    """
    upload_config = config.get("upload", {})

    # Extract settings
    enabled_types = upload_config.get("enabled_file_types", ["jpg", "jpeg", "png"])
    max_size_mb = upload_config.get("max_file_size_mb", 10)
    multi_file = upload_config.get("multi_file_selection", True)
    show_instructions = upload_config.get("show_upload_instructions", True)
    help_text = upload_config.get("upload_help_text", "Upload images for processing")

    # Show instructions if enabled
    if show_instructions:
        st.markdown(f"**{help_text}**")
        st.caption(f"Supported formats: {', '.join(enabled_types).upper()} | " f"Max size: {max_size_mb}MB per file")

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose image(s)",
        type=enabled_types,
        accept_multiple_files=multi_file,
        key="image_uploader",
        label_visibility="collapsed",
    )

    # Process uploaded files
    if uploaded_files:
        # Handle single vs multiple files
        files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]

        # Check if new files were uploaded
        current_file_names = [f.name for f in files]
        cached_file_names = state.get_preference("uploaded_file_names", [])

        if current_file_names != cached_file_names:
            logger.info(f"Processing {len(files)} uploaded file(s)")

            # Clear existing images
            state.clear_images()

            # Load and convert images
            for uploaded_file in files:
                try:
                    # Check file size
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    if file_size_mb > max_size_mb:
                        st.error(f"File {uploaded_file.name} is too large " f"({file_size_mb:.1f}MB > {max_size_mb}MB)")
                        continue

                    # Load image
                    image = Image.open(uploaded_file)

                    # Apply EXIF orientation (fix rotated images from cameras)
                    try:
                        from PIL import ImageOps

                        image = ImageOps.exif_transpose(image)
                    except Exception:
                        # If EXIF processing fails, continue with original image
                        pass

                    # Convert to RGB if needed
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # Convert to numpy array (OpenCV format: BGR)
                    image_np = np.array(image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    # Add to state
                    state.add_image(image_bgr)

                    logger.info(f"Loaded {uploaded_file.name}: " f"{image_bgr.shape[1]}x{image_bgr.shape[0]} " f"({file_size_mb:.1f}MB)")

                except Exception as e:
                    logger.error(f"Failed to load {uploaded_file.name}: {e}")
                    st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
                    continue

            # Cache file names
            state.set_preference("uploaded_file_names", current_file_names)

            # Clear processing cache since new images were uploaded
            state.clear_cache()

            # Save state
            state.to_session()

            st.success(f"Loaded {len(state.uploaded_images)} image(s)")
            st.rerun()

    # Display current state
    if state.uploaded_images:
        st.divider()

        # Image selector if multiple images
        if len(state.uploaded_images) > 1:
            col1, col2 = st.columns([3, 1])

            with col1:
                selected_idx = st.selectbox(
                    "Select image",
                    range(len(state.uploaded_images)),
                    index=state.current_image_index,
                    format_func=lambda x: f"Image {x + 1}",
                    key="image_selector",
                )

                if selected_idx != state.current_image_index:
                    state.current_image_index = selected_idx
                    state.to_session()
                    st.rerun()

            with col2:
                if st.button("🗑️ Clear All", key="clear_images"):
                    state.clear_images()
                    state.set_preference("uploaded_file_names", [])
                    state.to_session()
                    st.rerun()
        else:
            if st.button("🗑️ Clear", key="clear_single_image"):
                state.clear_images()
                state.set_preference("uploaded_file_names", [])
                state.to_session()
                st.rerun()

        # Show current image info
        current_image = state.get_current_image()
        if current_image is not None:
            h, w = current_image.shape[:2]
            size_mb = current_image.nbytes / (1024 * 1024)
            st.caption(f"📐 Dimensions: {w}×{h} | 💾 Size: {size_mb:.2f}MB")


def load_image_from_path(image_path: str | Path) -> np.ndarray | None:
    """Load image from file path with EXIF orientation handling.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (BGR format) or None if failed
    """
    try:
        image_path = Path(image_path)

        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        # Load with PIL to handle EXIF orientation
        image = Image.open(image_path)

        # Apply EXIF orientation (fix rotated images from cameras)
        try:
            from PIL import ImageOps

            image = ImageOps.exif_transpose(image)
        except Exception:
            # If EXIF processing fails, continue with original image
            pass

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array (OpenCV format: BGR)
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        logger.info(f"Loaded image from {image_path}: {image_bgr.shape[1]}x{image_bgr.shape[0]}")
        return image_bgr

    except Exception as e:
        logger.error(f"Error loading image from {image_path}: {e}")
        return None
