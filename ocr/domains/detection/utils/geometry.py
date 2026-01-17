"""Geometry helper functions extracted from dataset transforms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def calculate_inverse_transform(
    original_size: tuple[int, int],
    transformed_size: tuple[int, int],
    *,
    crop_box: tuple[int, int, int, int] | None = None,
    padding_position: str | None = None,
) -> np.ndarray:
    """Compute inverse transform matrix mapping transformed coordinates back to original space.

    🚨 CRITICAL FUNCTION - DO NOT MODIFY WITHOUT TESTS

    BUG-20251116-001: This function MUST match the padding position used in transforms.
    Incorrect padding_position causes coordinate transformation errors (negative coords,
    out-of-bounds values, incorrect metrics).

    Requirements:
    - padding_position MUST match PadIfNeeded transform position ("top_left" or "center")
    - For "top_left": translation is (0, 0) - no offset
    - For "center": translation includes padding offset to undo centering

    See: docs/bug_reports/BUG-20251116-001_DEBUGGING_HANDOVER.md

    Args:
        original_size: (width, height) of the original image
        transformed_size: (width, height) of the transformed image (after padding)
        crop_box: Optional (x, y, w, h) crop box. If None, computed from sizes.
        padding_position: Padding position used in transforms - "top_left" or "center".
                         If None, uses crop_box if provided, otherwise assumes "center".

    Returns:
        3x3 inverse transformation matrix (numpy array)
    """
    ox, oy = original_size
    tx, ty = transformed_size
    cx, cy = 0, 0

    if crop_box:
        cx, cy, tx, ty = crop_box
    elif padding_position == "top_left":
        # For top_left padding, no translation needed (cx=0, cy=0)
        # Scale is computed from original to transformed (which is same as scaled size)
        # The transformed_size is already the padded size (640x640), but we need to
        # compute scale from the scaled (pre-padded) size
        scale = transformed_size[0] / max(ox, oy)  # Assuming square output
        scaled_w = int(round(ox * scale))
        scaled_h = int(round(oy * scale))
        tx, ty = scaled_w, scaled_h
        cx, cy = 0, 0
    # else: use default cx=0, cy=0 (no translation)

    # Scale back to the original size
    scale_x = ox / tx
    scale_y = oy / ty
    scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)

    # Padding back to the original size
    translation_matrix = np.eye(3, dtype=np.float32)
    translation_matrix[0, 2] = -cx
    translation_matrix[1, 2] = -cy

    inverse_matrix = scale_matrix @ translation_matrix
    return inverse_matrix.astype(np.float32)


def compute_padding_offsets(
    original_size: tuple[int, int],
    target_size: int = 640,
    *,
    position: str = "top_left",
) -> tuple[int, int]:
    """Compute padding offsets for LongestMaxSize + PadIfNeeded transform.

    This function computes the padding offsets that would be applied by:
    1. LongestMaxSize(max_size=target_size) - scales longest side to target_size
    2. PadIfNeeded(min_width=target_size, min_height=target_size, position=position)

    Args:
        original_size: (width, height) of the original image
        target_size: Target size for LongestMaxSize and PadIfNeeded (default: 640)
        position: Padding position - "top_left" (default) or "center"

    Returns:
        (pad_x, pad_y): Padding offsets in (x, y) format
        - For "top_left": (0, 0) - no offset, padding is at bottom/right
        - For "center": (pad_left, pad_top) - offset to center the image

    Example:
        >>> compute_padding_offsets((960, 1280), 640, position="top_left")
        (0, 0)  # No offset, padding at bottom/right
        >>> compute_padding_offsets((960, 1280), 640, position="center")
        (0, 0)  # For portrait: scale to 480x640, pad to 640x640, center at (80, 0)
    """
    orig_w, orig_h = original_size

    # Step 1: LongestMaxSize scales longest side to target_size
    scale = target_size / max(orig_w, orig_h)
    scaled_w = int(round(orig_w * scale))
    scaled_h = int(round(orig_h * scale))

    # Step 2: PadIfNeeded pads to target_size x target_size
    pad_w = target_size - scaled_w
    pad_h = target_size - scaled_h

    if position == "top_left":
        # Padding is at bottom/right, no offset
        return (0, 0)
    elif position == "center":
        # Padding is centered
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        return (pad_left, pad_top)
    else:
        raise ValueError(f"Unsupported padding position: {position}. Use 'top_left' or 'center'.")


def apply_padding_offset_to_polygons(
    polygons: Sequence[np.ndarray],
    pad_x: int,
    pad_y: int,
) -> list[np.ndarray]:
    """Apply padding offset to polygon coordinates.

    Args:
        polygons: List of polygon arrays, each of shape (N, 2) with (x, y) coordinates
        pad_x: X-axis padding offset (typically 0 for top_left, or pad_left for center)
        pad_y: Y-axis padding offset (typically 0 for top_left, or pad_top for center)

    Returns:
        List of polygons with padding offset applied
    """
    if pad_x == 0 and pad_y == 0:
        return list(polygons)

    offset_polygons = []
    for polygon in polygons:
        if polygon.size == 0:
            offset_polygons.append(polygon)
            continue
        # Apply offset: (x, y) -> (x + pad_x, y + pad_y)
        offset_polygon = polygon.copy()
        offset_polygon[:, 0] += pad_x  # x coordinates
        offset_polygon[:, 1] += pad_y  # y coordinates
        offset_polygons.append(offset_polygon)

    return offset_polygons


def calculate_cropbox(
    original_size: tuple[int, int],
    target_size: int = 640,
    *,
    position: str = "center",
) -> tuple[int, int, int, int]:
    """Determine crop box applied during resize letterboxing.

    🚨 CRITICAL FUNCTION - DO NOT MODIFY WITHOUT TESTS

    BUG-20251116-001: This function MUST match the padding position used in transforms.
    Incorrect position causes inverse_matrix to have wrong translation components.

    Requirements:
    - position MUST match PadIfNeeded transform position ("top_left" or "center")
    - For "top_left": returns (0, 0, w, h) - no offset
    - For "center": returns (delta_w//2, delta_h//2, w, h) - centered offset

    See: docs/bug_reports/BUG-20251116-001_DEBUGGING_HANDOVER.md

    Args:
        original_size: (width, height) of the original image
        target_size: Target size for LongestMaxSize and PadIfNeeded (default: 640)
        position: Padding position - "top_left" or "center" (default: "center" for backward compatibility)

    Returns:
        (x, y, w, h): Crop box coordinates where (x, y) is the top-left corner of the scaled image
        in the padded space, and (w, h) is the scaled image dimensions.
        - For "top_left": (0, 0, new_width, new_height) - no offset
        - For "center": (delta_w // 2, delta_h // 2, new_width, new_height) - centered
    """
    ox, oy = original_size
    scale = target_size / max(ox, oy)
    new_width, new_height = int(ox * scale), int(oy * scale)
    delta_w = target_size - new_width
    delta_h = target_size - new_height

    if position == "top_left":
        x, y = 0, 0
    elif position == "center":
        x, y = delta_w // 2, delta_h // 2
    else:
        raise ValueError(f"Unsupported padding position: {position}. Use 'top_left' or 'center'.")

    w, h = new_width, new_height
    return x, y, w, h
