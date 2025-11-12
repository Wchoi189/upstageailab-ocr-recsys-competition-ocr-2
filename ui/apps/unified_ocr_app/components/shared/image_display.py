"""Shared image display utilities.

Provides flexible image display with grid and side-by-side layouts.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)


def display_image_grid(
    images: list[np.ndarray],
    labels: list[str] | None = None,
    config: dict[str, Any] | None = None,
    max_width: int | None = None,
) -> None:
    """Display images in grid layout.

    Args:
        images: List of images to display (BGR format)
        labels: Optional labels for each image
        config: Display configuration from YAML (config.shared.image_display)
        max_width: Override max width from config
    """
    if not images:
        st.warning("No images to display")
        return

    # Extract config settings
    if config is None:
        config = {}

    display_config = config.get("image_display", {})
    grid_columns = display_config.get("grid_columns", 3)
    default_max_width = display_config.get("max_width", 800)
    show_dimensions = display_config.get("show_dimensions", True)

    # Use override or default
    max_width = max_width or default_max_width

    # Create labels if not provided
    if labels is None:
        labels = [f"Image {i + 1}" for i in range(len(images))]

    # Ensure labels match images
    if len(labels) < len(images):
        labels.extend([f"Image {i + 1}" for i in range(len(labels), len(images))])

    # Display in grid
    num_images = len(images)
    num_rows = (num_images + grid_columns - 1) // grid_columns

    for row in range(num_rows):
        cols = st.columns(grid_columns)

        for col_idx in range(grid_columns):
            img_idx = row * grid_columns + col_idx

            if img_idx < num_images:
                with cols[col_idx]:
                    image = images[img_idx]
                    label = labels[img_idx]

                    # Display image
                    display_single_image(
                        image,
                        caption=label,
                        max_width=max_width,
                        show_dimensions=show_dimensions,
                    )


def display_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    labels: tuple[str, str] | None = None,
    max_width: int = 400,
    show_dimensions: bool = True,
) -> None:
    """Display two images side-by-side for comparison.

    Args:
        left: Left image (BGR format)
        right: Right image (BGR format)
        labels: (left_label, right_label) or None for defaults
        max_width: Maximum width per image
        show_dimensions: Show image dimensions
    """
    if labels is None:
        labels = ("Original", "Processed")

    col1, col2 = st.columns(2)

    with col1:
        display_single_image(
            left,
            caption=labels[0],
            max_width=max_width,
            show_dimensions=show_dimensions,
        )

    with col2:
        display_single_image(
            right,
            caption=labels[1],
            max_width=max_width,
            show_dimensions=show_dimensions,
        )


def display_single_image(
    image: np.ndarray,
    caption: str | None = None,
    max_width: int | None = None,
    show_dimensions: bool = True,
    use_container_width: bool = False,
) -> None:
    """Display single image with caption and metadata.

    Args:
        image: Image to display (BGR format)
        caption: Optional caption
        max_width: Maximum display width (None for auto)
        show_dimensions: Show image dimensions below
        use_container_width: Use full container width
    """
    if image is None or image.size == 0:
        st.warning(f"{caption or 'Image'}: No image data")
        return

    # Convert BGR to RGB for display
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    # Display image
    st.image(
        image_rgb,
        caption=caption,
        width=max_width if not use_container_width else None,
        use_container_width=use_container_width,
    )

    # Show metadata
    if show_dimensions:
        h, w = image.shape[:2]
        size_mb = image.nbytes / (1024 * 1024)
        st.caption(f"📐 {w}×{h} | 💾 {size_mb:.2f}MB")


def display_stage_comparison(
    stages: dict[str, np.ndarray],
    stage_order: list[str] | None = None,
    max_width: int = 300,
) -> None:
    """Display multiple processing stages in sequence.

    Args:
        stages: Dict mapping stage names to images
        stage_order: Order to display stages (None for dict order)
        max_width: Maximum width per image
    """
    if not stages:
        st.warning("No stages to display")
        return

    # Determine order
    if stage_order is None:
        stage_order = list(stages.keys())

    # Filter to existing stages
    stage_order = [s for s in stage_order if s in stages]

    if not stage_order:
        st.warning("No valid stages found")
        return

    # Display in columns
    num_stages = len(stage_order)
    cols = st.columns(num_stages)

    for idx, stage_name in enumerate(stage_order):
        with cols[idx]:
            image = stages[stage_name]

            # Format stage name for display
            display_name = stage_name.replace("_", " ").title()

            display_single_image(
                image,
                caption=display_name,
                max_width=max_width,
                show_dimensions=True,
            )


def create_image_comparison_slider(
    image_before: np.ndarray,
    image_after: np.ndarray,
    label_before: str = "Before",
    label_after: str = "After",
) -> None:
    """Create interactive slider to compare before/after images.

    Note: This is a simple version using columns with a slider.
    For true overlay comparison, consider using streamlit-image-comparison.

    Args:
        image_before: Before image (BGR format)
        image_after: After image (BGR format)
        label_before: Label for before image
        label_after: Label for after image
    """
    st.subheader("Before/After Comparison")

    # Slider to control view
    comparison_mode = st.radio(
        "View mode",
        ["Side by Side", "Before Only", "After Only"],
        horizontal=True,
        key="comparison_mode",
    )

    if comparison_mode == "Side by Side":
        display_side_by_side(
            image_before,
            image_after,
            labels=(label_before, label_after),
        )
    elif comparison_mode == "Before Only":
        display_single_image(
            image_before,
            caption=label_before,
            use_container_width=True,
        )
    else:  # After Only
        display_single_image(
            image_after,
            caption=label_after,
            use_container_width=True,
        )


def display_image_with_overlay(
    image: np.ndarray,
    overlay_data: dict[str, Any] | None = None,
    caption: str | None = None,
) -> None:
    """Display image with optional overlay data (bboxes, text, etc).

    Args:
        image: Base image (BGR format)
        overlay_data: Optional overlay information (bboxes, polygons, text)
        caption: Optional caption
    """
    # Convert to RGB for display
    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # If overlay data provided, draw on image
    if overlay_data:
        display_image = draw_overlay(display_image.copy(), overlay_data)

    # Display
    st.image(display_image, caption=caption, use_container_width=True)


def draw_overlay(image: np.ndarray, overlay_data: dict[str, Any]) -> np.ndarray:
    """Draw overlay annotations on image.

    Args:
        image: Image to draw on (RGB format)
        overlay_data: Overlay information (bboxes, polygons, labels)

    Returns:
        Image with overlays drawn
    """
    # Convert back to BGR for OpenCV drawing
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes if provided
    if "bboxes" in overlay_data:
        for bbox in overlay_data["bboxes"]:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw polygons if provided
    if "polygons" in overlay_data:
        for polygon in overlay_data["polygons"]:
            pts = np.array(polygon, dtype=np.int32)
            cv2.polylines(image_bgr, [pts], True, (0, 0, 255), 2)

    # Draw text labels if provided
    if "labels" in overlay_data:
        for label_info in overlay_data["labels"]:
            text = label_info.get("text", "")
            position = label_info.get("position", (10, 30))
            cv2.putText(
                image_bgr,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

    # Convert back to RGB
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
