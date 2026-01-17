from __future__ import annotations

"""Validation functions for perspective correction."""

import math

import cv2
import numpy as np

from .geometry import _compute_edge_vectors


def _validate_edge_angles(corners: np.ndarray, angle_tolerance_deg: float = 15.0) -> bool:
    """
    Validate that edges form approximately right angles (rectangular shape).

    Args:
        corners: Ordered quadrilateral corners (4, 2)
        angle_tolerance_deg: Maximum deviation from 90 degrees allowed (default: 15 degrees)

    Returns:
        True if all angles are approximately 90 degrees, False otherwise
    """
    if corners is None or len(corners) != 4:
        return False

    edges, _ = _compute_edge_vectors(corners)

    # Compute angles at each corner
    for i in range(4):
        v1 = edges[i]
        v2 = -edges[(i - 1) % 4]  # Previous edge reversed

        # Normalize
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return False

        # Dot product gives cosine of angle
        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))

        # Check deviation from 90 degrees
        deviation = abs(angle_deg - 90.0)
        if deviation > angle_tolerance_deg:
            return False

    return True


def _validate_edge_lengths(corners: np.ndarray, min_aspect_ratio: float = 0.1, max_aspect_ratio: float = 10.0) -> bool:
    """
    Validate edge lengths form reasonable document proportions.

    Args:
        corners: Ordered quadrilateral corners (4, 2)
        min_aspect_ratio: Minimum width/height or height/width ratio (default: 0.1)
        max_aspect_ratio: Maximum width/height or height/width ratio (default: 10.0)

    Returns:
        True if aspect ratio is within bounds, False otherwise
    """
    if corners is None or len(corners) != 4:
        return False

    _, lengths = _compute_edge_vectors(corners)

    # Opposite edges should be approximately equal (top-bottom, left-right)
    top_bottom_avg = (lengths[0] + lengths[2]) / 2.0
    left_right_avg = (lengths[1] + lengths[3]) / 2.0

    if top_bottom_avg < 1e-6 or left_right_avg < 1e-6:
        return False

    # Check aspect ratio
    aspect1 = top_bottom_avg / left_right_avg
    aspect2 = left_right_avg / top_bottom_avg

    if aspect1 < min_aspect_ratio or aspect1 > max_aspect_ratio:
        if aspect2 < min_aspect_ratio or aspect2 > max_aspect_ratio:
            return False

    return True


def _validate_contour_alignment(
    corners: np.ndarray,
    contour: np.ndarray,
    mask_bbox: tuple[int, int, int, int],
    alignment_tolerance: float = 0.15,
) -> bool:
    """
    Cross-validate fitted rectangle against mask bounding box and contour structure.

    Args:
        corners: Fitted rectangle corners (4, 2)
        contour: Original mask contour
        mask_bbox: (x, y, w, h) bounding box of mask foreground
        alignment_tolerance: Maximum relative deviation from bbox alignment (default: 15%)

    Returns:
        True if rectangle aligns reasonably with mask bbox, False otherwise
    """
    if corners is None or contour is None or len(contour) == 0:
        return False

    x, y, w, h = mask_bbox

    # Compute fitted rectangle bbox
    min_x = float(np.min(corners[:, 0]))
    max_x = float(np.max(corners[:, 0]))
    min_y = float(np.min(corners[:, 1]))
    max_y = float(np.max(corners[:, 1]))

    fitted_w = max_x - min_x
    fitted_h = max_y - min_y

    if fitted_w < 1e-6 or fitted_h < 1e-6 or w < 1e-6 or h < 1e-6:
        return False

    # Check width/height alignment
    width_ratio = fitted_w / w
    height_ratio = fitted_h / h

    if abs(width_ratio - 1.0) > alignment_tolerance or abs(height_ratio - 1.0) > alignment_tolerance:
        return False

    # Check position alignment (centroid should be close)
    mask_center_x = x + w / 2.0
    mask_center_y = y + h / 2.0
    fitted_center_x = (min_x + max_x) / 2.0
    fitted_center_y = (min_y + max_y) / 2.0

    center_offset_x = abs(fitted_center_x - mask_center_x) / w
    center_offset_y = abs(fitted_center_y - mask_center_y) / h

    if center_offset_x > alignment_tolerance or center_offset_y > alignment_tolerance:
        return False

    return True


def _validate_contour_segments(contour: np.ndarray, min_segment_length: float = 5.0) -> bool:
    """
    Validate contour has sufficient structure (not too fragmented).

    Args:
        contour: Contour points
        min_segment_length: Minimum average segment length (default: 10 pixels)

    Returns:
        True if contour has reasonable structure, False otherwise
    """
    if contour is None or len(contour) < 4:
        return False

    # Compute arc length
    arc_length = cv2.arcLength(contour, closed=True)

    if arc_length < 1e-6:
        return False

    # Average segment length
    avg_segment = arc_length / len(contour)

    return avg_segment >= min_segment_length


__all__ = [
    "_validate_edge_angles",
    "_validate_edge_lengths",
    "_validate_contour_alignment",
    "_validate_contour_segments",
]
