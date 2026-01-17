from __future__ import annotations

"""Core perspective correction functions."""

import logging

import cv2
import numpy as np

from .fitting import fit_mask_rectangle
from .types import MaskRectangleResult

# Optional rembg dependency
try:
    from rembg import remove as _rembg_remove

    _REMBG_AVAILABLE = True
except Exception:
    _rembg_remove = None
    _REMBG_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


def calculate_target_dimensions(pts: np.ndarray) -> tuple[int, int]:
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


def correct_perspective_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    return_matrix: bool = False,
) -> tuple[np.ndarray, MaskRectangleResult] | tuple[np.ndarray, MaskRectangleResult, np.ndarray]:
    """
    High-level helper: correct perspective of an image given a foreground mask.

    Args:
        image: Input image as numpy array (BGR or RGB).
        mask: Binary mask (0 background, >0 foreground).
        return_matrix: If True, also return the perspective transform matrix.

    Returns:
        Tuple of (warped_image, fit_result) or (warped_image, fit_result, transform_matrix)
        if return_matrix is True.
        If fitting fails, the original image is returned.
    """

    fit_result = fit_mask_rectangle(mask)
    if fit_result.corners is None:
        # No usable rectangle; return original image for safety.
        if return_matrix:
            return image, fit_result, np.eye(3, dtype=np.float32)
        return image, fit_result

    # Calculate transform matrix before warping
    if fit_result.corners is not None and fit_result.corners.dtype != np.float32:
        pts = fit_result.corners.astype(np.float32)
    else:
        pts = fit_result.corners

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
    transform_matrix = cv2.getPerspectiveTransform(pts, dst)

    warped = four_point_transform(image, fit_result.corners)

    if return_matrix:
        return warped, fit_result, transform_matrix
    return warped, fit_result


def remove_background_and_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove background using rembg and return a foreground mask for perspective fitting.

    This helper is a generalized, UI-friendly version of the experimental pipeline:
    - Converts BGR -> RGB for rembg.
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
        # RGBA -> composite on white and build mask from alpha
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

    # Prepare mask to ensure it's binary
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask_binary = np.where(mask > 0, 255, 0).astype(np.uint8)

    return image_no_bg_bgr, mask_binary


def transform_polygons_inverse(
    polygons_str: str,
    transform_matrix: np.ndarray,
) -> str:
    """
    Transform polygon coordinates from corrected image space back to original image space.

    Args:
        polygons_str: Polygons in string format "x1 y1 x2 y2 ... | x1 y1 x2 y2 ..."
        transform_matrix: Forward perspective transform matrix (original -> corrected)

    Returns:
        Transformed polygons in the same string format, mapped back to original space
    """
    if not polygons_str or polygons_str.strip() == "":
        return polygons_str

    # Compute inverse transform matrix
    try:
        inverse_matrix = cv2.invert(transform_matrix)[1]
    except Exception:
        # If inversion fails, return original polygons unchanged
        return polygons_str

    # Parse polygons
    polygon_groups = polygons_str.split("|")
    transformed_groups = []

    for polygon_str in polygon_groups:
        coords = polygon_str.strip().split()
        if len(coords) < 2:
            transformed_groups.append(polygon_str)
            continue

        try:
            # Convert to float array
            coord_floats = [float(c) for c in coords]
            points = np.array(
                [[coord_floats[i], coord_floats[i + 1]] for i in range(0, len(coord_floats), 2)],
                dtype=np.float32,
            )

            # Transform points using inverse matrix
            # cv2.perspectiveTransform expects shape (1, N, 2)
            points_reshaped = points.reshape(1, -1, 2)
            transformed_points = cv2.perspectiveTransform(points_reshaped, inverse_matrix)
            transformed_points = transformed_points.reshape(-1, 2)

            # Convert back to string format
            transformed_str = " ".join(f"{float(pt[0]):.6f} {float(pt[1]):.6f}" for pt in transformed_points)
            transformed_groups.append(transformed_str)

        except (ValueError, IndexError):
            # If parsing fails, keep original
            transformed_groups.append(polygon_str)

    return " | ".join(transformed_groups)


__all__ = [
    "calculate_target_dimensions",
    "four_point_transform",
    "correct_perspective_from_mask",
    "remove_background_and_mask",
    "transform_polygons_inverse",
]
