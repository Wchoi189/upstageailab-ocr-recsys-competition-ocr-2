from __future__ import annotations

"""
Perspective correction utilities for OCR images.

This module generalizes the experimental perspective-correction pipeline into
reusable, dependency-free helpers that operate purely on NumPy arrays:

- Mask-based rectangle fitting (`fit_mask_rectangle`) operating on binary masks.
- Max-Edge target sizing and high-quality warp (`four_point_transform`).
- A high-level helper (`correct_perspective_from_mask`) that combines both.

Rembg-based background removal is provided separately so that callers can decide
when to incur that cost.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:  # Optional dependency
    from rembg import remove as _rembg_remove

    _REMBG_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _rembg_remove = None
    _REMBG_AVAILABLE = False


@dataclass
class MaskRectangleResult:
    """Result of fitting a rectangle to a foreground mask."""

    corners: Optional[np.ndarray]
    raw_corners: Optional[np.ndarray]
    contour_area: float
    hull_area: float
    mask_area: float
    contour: Optional[np.ndarray]
    hull: Optional[np.ndarray]
    reason: Optional[str] = None
    used_epsilon: Optional[float] = None


def calculate_target_dimensions(pts: np.ndarray) -> Tuple[int, int]:
    """
    Calculate the target width and height using the 'Max-Edge' aspect ratio rule.

    Args:
        pts: The 4 detected corners in order [TL, TR, BR, BL]

    Returns:
        (maxWidth, maxHeight) for the destination image.
    """

    (tl, tr, br, bl) = pts

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = int(max(width_a, width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = int(max(height_a, height_b))

    return max_width, max_height


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective transform to the image using the provided corners.

    Args:
        image: Source image (BGR or RGB)
        pts: The 4 detected corners in order [TL, TR, BR, BL]

    Returns:
        Warped image with size computed via `calculate_target_dimensions`.
    """

    if pts.dtype != np.float32:
        pts = pts.astype(np.float32)

    max_width, max_height = calculate_target_dimensions(pts)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    m = cv2.getPerspectiveTransform(pts, dst)

    # Use INTER_LANCZOS4 for better quality on text documents
    warped = cv2.warpPerspective(image, m, (max_width, max_height), flags=cv2.INTER_LANCZOS4)
    return warped


def _prepare_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is binary {0,255} uint8."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    return binary


def _extract_largest_component(mask: np.ndarray) -> np.ndarray:
    """Return a binary mask containing only the largest connected component."""
    if mask is None or mask.size == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask

    largest_idx = 1 + int(np.argmax(areas))
    largest_mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)
    return largest_mask


def fit_mask_rectangle(mask: np.ndarray) -> MaskRectangleResult:
    """
    Fit rectangle corners directly from a binary foreground mask.

    This is a streamlined variant of the experimental implementation:
    - Operates on a single binary mask (0 background, 255 foreground).
    - Finds the largest connected component.
    - Approximates its convex hull with a quadrilateral when possible.
    - Falls back to the bounding box if necessary.

    Args:
        mask: Binary mask (0 background, >0 foreground).

    Returns:
        MaskRectangleResult with ordered corners (TL, TR, BR, BL) or None on failure.
    """

    binary = _prepare_mask(mask)
    img_h, img_w = binary.shape[:2]
    total_pixels = float(img_h * img_w)

    largest_component = _extract_largest_component(binary)
    if largest_component is None or np.count_nonzero(largest_component) == 0:
        return MaskRectangleResult(
            corners=None,
            raw_corners=None,
            contour_area=0.0,
            hull_area=0.0,
            mask_area=0.0,
            contour=None,
            hull=None,
            reason="no_connected_components",
        )

    contours, _ = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return MaskRectangleResult(
            corners=None,
            raw_corners=None,
            contour_area=0.0,
            hull_area=0.0,
            mask_area=float(np.count_nonzero(largest_component)),
            contour=None,
            hull=None,
            reason="no_contours",
        )

    largest = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(largest))
    mask_area = float(np.count_nonzero(largest_component))

    # Require a minimal area relative to the image to avoid noise
    min_area_ratio = 0.005
    if contour_area < total_pixels * min_area_ratio:
        return MaskRectangleResult(
            corners=None,
            raw_corners=None,
            contour_area=contour_area,
            hull_area=0.0,
            mask_area=mask_area,
            contour=largest,
            hull=None,
            reason="insufficient_mask_support",
        )

    # Fallback corners: axis-aligned bounding box of the largest component
    x, y, w, h = cv2.boundingRect(largest_component)
    bbox_corners = np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )

    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))

    peri = cv2.arcLength(hull, True)
    if peri < 1e-3:
        return MaskRectangleResult(
            corners=bbox_corners,
            raw_corners=None,
            contour_area=contour_area,
            hull_area=hull_area,
            mask_area=mask_area,
            contour=largest,
            hull=hull,
            reason="degenerate_hull_using_bbox",
        )

    # Simple approxPolyDP-based quadrilateral fitting
    eps = max(peri * 0.01, 1.0)
    max_eps = peri * 0.08
    used_eps = eps
    quad: Optional[np.ndarray] = None

    while eps <= max_eps:
        approx = cv2.approxPolyDP(hull, eps, True)
        if len(approx) == 4:
            quad = approx.reshape(-1, 2).astype(np.float32)
            used_eps = eps
            break
        eps *= 1.5

    if quad is None:
        # Fallback: try to coerce hull bbox into 4 points
        quad = bbox_corners.astype(np.float32)
        reason = "quad_fit_failed_using_bbox"
    else:
        reason = None

    # Order points as TL, TR, BR, BL for downstream compatibility
    ordered = _order_points(quad)

    return MaskRectangleResult(
        corners=ordered,
        raw_corners=quad,
        contour_area=contour_area,
        hull_area=hull_area,
        mask_area=mask_area,
        contour=largest,
        hull=hull,
        reason=reason,
        used_epsilon=used_eps,
    )


def _order_points(points: np.ndarray) -> np.ndarray:
    """Order quadrilateral corners as TL, TR, BR, BL."""
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def correct_perspective_from_mask(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, MaskRectangleResult]:
    """
    High-level helper: correct perspective of an image given a foreground mask.

    Args:
        image: Input image as numpy array (BGR or RGB).
        mask: Binary mask (0 background, >0 foreground).

    Returns:
        Tuple of (warped_image, fit_result).
        If fitting fails, the original image is returned.
    """

    fit_result = fit_mask_rectangle(mask)
    if fit_result.corners is None:
        # No usable rectangle; return original image for safety.
        return image, fit_result

    warped = four_point_transform(image, fit_result.corners)
    return warped, fit_result


def remove_background_and_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove background using rembg and return a foreground mask for perspective fitting.

    This helper is a generalized, UI-friendly version of the experimental pipeline:
    - Converts BGR → RGB for rembg.
    - Runs rembg to obtain RGBA/RGB output.
    - Produces:
        * An RGB image with background composited on white.
        * A binary mask (0 background, 255 foreground) derived from the alpha channel
          when available, or from non-white pixels otherwise.

    Args:
        image_bgr: Input image in BGR format (OpenCV convention).

    Returns:
        Tuple of (image_no_bg_bgr, mask_binary).

    Raises:
        RuntimeError: If rembg is not available.
    """

    if not _REMBG_AVAILABLE or _rembg_remove is None:
        raise RuntimeError("rembg is not available. Install it to use background removal.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Run rembg with a reasonable default model; caller can override globally if needed
    output = _rembg_remove(image_rgb)
    output_array = np.asarray(output)

    if output_array.ndim != 3:
        # Fallback: no structure; treat everything as foreground
        mask = np.ones(image_rgb.shape[:2], dtype=np.uint8) * 255
        image_no_bg_bgr = image_bgr.copy()
        return image_no_bg_bgr, mask

    if output_array.shape[2] == 4:
        # RGBA → composite on white and build mask from alpha
        rgb = output_array[:, :, :3]
        alpha = output_array[:, :, 3].astype(np.float32) / 255.0
        alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)

        white_bg = np.ones_like(rgb, dtype=np.float32) * 255.0
        result_rgb = (rgb.astype(np.float32) * alpha_3 + white_bg * (1.0 - alpha_3)).astype(np.uint8)

        mask = (alpha * 255.0).astype(np.uint8)
    else:
        # No alpha: treat non-white pixels as foreground
        rgb = output_array
        # Simple luminance-based foreground: anything significantly darker than white
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Threshold at high value so near-white is background
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        result_rgb = rgb

    image_no_bg_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    mask_binary = _prepare_mask(mask)

    return image_no_bg_bgr, mask_binary


__all__ = [
    "MaskRectangleResult",
    "calculate_target_dimensions",
    "four_point_transform",
    "fit_mask_rectangle",
    "correct_perspective_from_mask",
    "remove_background_and_mask",
]

