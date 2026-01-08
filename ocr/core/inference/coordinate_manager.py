from __future__ import annotations

"""Coordinate transformation utilities for OCR inference.

Unified coordinate transformation logic for mapping between:
- Original image space
- Processed/padded image space (e.g., 640x640)

Consolidates duplicate logic from engine.py and postprocess.py.

BugRef: BUG-20251116-001 — inference inverse-mapping must match padding position
Report: docs/bug_reports/BUG-20251116-001_DEBUGGING_HANDOVER.md
"""

import logging
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

LOGGER = logging.getLogger(__name__)


class TransformMetadata(NamedTuple):
    """Metadata for coordinate transformations between image spaces.

    Attributes:
        original_h: Original image height
        original_w: Original image width
        resized_h: Content height after resize (before padding)
        resized_w: Content width after resize (before padding)
        target_size: Target size for processed image (e.g., 640)
        scale: Scale factor applied during resize
        pad_h: Bottom padding pixels
        pad_w: Right padding pixels
    """

    original_h: int
    original_w: int
    resized_h: int
    resized_w: int
    target_size: int
    scale: float
    pad_h: int
    pad_w: int


def calculate_transform_metadata(
    original_shape: Sequence[int],
    target_size: int = 640,
) -> TransformMetadata:
    """Calculate transformation metadata for resize and padding.

    Implements LongestMaxSize(max_size=target_size) + PadIfNeeded(position="top_left").
    With top_left padding, content starts at (0,0) and padding is at bottom/right only.

    Args:
        original_shape: Original image shape (height, width) or (height, width, channels)
        target_size: Target size for square processed image (default: 640)

    Returns:
        TransformMetadata with all transformation parameters

    Raises:
        ValueError: If original_shape dimensions are invalid
    """
    if len(original_shape) < 2:
        raise ValueError(f"original_shape must have at least 2 dimensions, got {len(original_shape)}")

    original_h, original_w = original_shape[:2]

    if original_h <= 0 or original_w <= 0:
        raise ValueError(f"Invalid original dimensions: {original_w}x{original_h}")

    # LongestMaxSize: scale to fit longest side within target_size
    max_side = float(max(original_h, original_w))
    if max_side == 0:
        raise ValueError("Invalid image size: max dimension is 0")

    scale = target_size / max_side
    resized_h = int(round(original_h * scale))
    resized_w = int(round(original_w * scale))

    # PadIfNeeded with position="top_left": pad bottom and right only
    pad_h = target_size - resized_h
    pad_w = target_size - resized_w

    return TransformMetadata(
        original_h=original_h,
        original_w=original_w,
        resized_h=resized_h,
        resized_w=resized_w,
        target_size=target_size,
        scale=scale,
        pad_h=pad_h,
        pad_w=pad_w,
    )


def compute_inverse_matrix(
    original_shape: Sequence[int],
    target_size: int = 640,
) -> np.ndarray:
    """Compute inverse transformation matrix: processed space → original space.

    🚨 CRITICAL FUNCTION - DO NOT MODIFY WITHOUT TESTS

    BUG-20251116-001: This function MUST match the padding position used in transforms.
    The transforms use PadIfNeeded with position="top_left", so padding is at bottom/right only.
    Therefore, there is NO translation needed (translation = 0, 0).

    Args:
        original_shape: Original image shape (height, width) or (height, width, channels)
        target_size: Target size for square processed image (default: 640)

    Returns:
        3x3 inverse transformation matrix as numpy array

    See: docs/bug_reports/BUG-20251116-001_DEBUGGING_HANDOVER.md
    """
    try:
        metadata = calculate_transform_metadata(original_shape, target_size)
    except ValueError:
        # Return identity matrix for invalid inputs
        return np.eye(3, dtype=np.float32)

    # Inverse scale: processed → original
    inv_scale = 1.0 / metadata.scale

    # BUG-20251116-001 FIX: For top_left padding, there is NO translation
    # Padding is at bottom/right only, so content starts at (0, 0) in padded space
    # Translation components must be (0, 0) - no offset needed
    matrix = np.array(
        [[inv_scale, 0.0, 0.0], [0.0, inv_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return matrix


def compute_forward_scales(
    original_shape: Sequence[int],
    target_size: int = 640,
) -> tuple[float, float]:
    """Compute forward transformation scales: original space → processed space.

    Returns separate x and y scales to match the exact calculation used in postprocessing.
    With top_left padding, no translation offset is needed.

    Args:
        original_shape: Original image shape (height, width) or (height, width, channels)
        target_size: Target size for square processed image (default: 640)

    Returns:
        Tuple of (scale_x, scale_y) for forward transformation
    """
    try:
        metadata = calculate_transform_metadata(original_shape, target_size)
    except ValueError:
        # Return identity scales for invalid inputs
        return (1.0, 1.0)

    # Forward scales: original → resized content (before padding)
    # This matches the inverse of postprocessing scales for perfect coordinate alignment
    scale_x = metadata.resized_w / float(metadata.original_w) if metadata.original_w > 0 else metadata.scale
    scale_y = metadata.resized_h / float(metadata.original_h) if metadata.original_h > 0 else metadata.scale

    return (scale_x, scale_y)


def transform_polygon_to_processed_space(
    polygon: np.ndarray,
    original_shape: Sequence[int],
    target_size: int = 640,
) -> np.ndarray:
    """Transform polygon coordinates from original space to processed space.

    Args:
        polygon: Nx2 array of polygon coordinates in original image space
        original_shape: Original image shape (height, width) or (height, width, channels)
        target_size: Target size for square processed image (default: 640)

    Returns:
        Nx2 array of transformed polygon coordinates in processed space
    """
    scale_x, scale_y = compute_forward_scales(original_shape, target_size)

    # Apply forward scaling
    coords_2d = polygon.reshape(-1, 2)
    transformed_coords = coords_2d * np.array([scale_x, scale_y], dtype=np.float32)

    return transformed_coords


def transform_polygons_string_to_processed_space(
    polygons_str: str,
    original_shape: Sequence[int],
    target_size: int = 640,
    tolerance: float = 2.0,
) -> str:
    """Transform polygon string from original space to processed space.

    Parses pipe-separated polygon strings, transforms coordinates, and returns
    the transformed string in the same format.

    Args:
        polygons_str: Pipe-separated polygon strings (format: "x1 y1 x2 y2 ... | x1 y1 ...")
        original_shape: Original image shape (height, width) or (height, width, channels)
        target_size: Target size for square processed image (default: 640)
        tolerance: Tolerance for coordinate bounds checking (pixels)

    Returns:
        Transformed polygon string in same format
    """
    if not polygons_str:
        return ""

    try:
        metadata = calculate_transform_metadata(original_shape, target_size)
    except ValueError:
        LOGGER.warning("Invalid original_shape, returning empty polygon string")
        return ""

    scale_x, scale_y = compute_forward_scales(original_shape, target_size)

    # Log coordinate mapping details for debugging
    LOGGER.debug(
        "Coordinate mapping - original: %dx%d, resized content: %dx%d, "
        "forward_scales: x=%.6f, y=%.6f, processed_size=%dx%d, padding: top=%d bottom=%d left=%d right=%d",
        metadata.original_w,
        metadata.original_h,
        metadata.resized_w,
        metadata.resized_h,
        scale_x,
        scale_y,
        target_size,
        target_size,
        0,
        metadata.pad_h,
        0,
        metadata.pad_w,
    )

    # Parse and transform polygons
    polygon_groups = polygons_str.split("|")
    transformed_polygons = []

    for polygon_str in polygon_groups:
        coords = polygon_str.strip().split()
        if len(coords) < 6:  # Need at least 3 points (6 coordinates)
            continue

        try:
            coord_floats = [float(c) for c in coords]
            polygon = np.array([[coord_floats[i], coord_floats[i + 1]] for i in range(0, len(coord_floats), 2)], dtype=np.float32)

            # Apply forward transform to map coordinates from original space to processed space
            coords_2d = polygon.reshape(-1, 2)
            transformed_coords = coords_2d * np.array([scale_x, scale_y], dtype=np.float32)

            # Verify coordinates are within processed bounds with tolerance
            if len(transformed_coords) > 0:
                min_x = min(c[0] for c in transformed_coords)
                min_y = min(c[1] for c in transformed_coords)
                max_x = max(c[0] for c in transformed_coords)
                max_y = max(c[1] for c in transformed_coords)

                if max_x > target_size + tolerance or max_y > target_size + tolerance or min_x < -tolerance or min_y < -tolerance:
                    LOGGER.warning(
                        "Transformed polygon coordinates out of processed bounds: "
                        "min=(%.1f, %.1f), max=(%.1f, %.1f), processed_size=%dx%d, "
                        "content_area=[0-%d, 0-%d] (original: %dx%d)",
                        min_x,
                        min_y,
                        max_x,
                        max_y,
                        target_size,
                        target_size,
                        metadata.resized_w,
                        metadata.resized_h,
                        metadata.original_w,
                        metadata.original_h,
                    )

            # Convert back to space-separated string (round to nearest integer)
            transformed_polygons.append(" ".join(str(int(round(c))) for row in transformed_coords for c in row))
        except (ValueError, IndexError):
            LOGGER.warning("Failed to parse polygon: %s", polygon_str)
            continue

    return "|".join(transformed_polygons) if transformed_polygons else ""


class CoordinateTransformationManager:
    """Manages coordinate transformations between original and processed image spaces.

    Provides a stateful interface for coordinate transformations, caching metadata
    to avoid redundant calculations.
    """

    def __init__(self, target_size: int = 640):
        """Initialize coordinate transformation manager.

        Args:
            target_size: Target size for square processed image (default: 640)
        """
        self.target_size = target_size
        self._cached_metadata: TransformMetadata | None = None
        self._cached_original_shape: tuple[int, int] | None = None

    def set_original_shape(self, original_shape: Sequence[int]) -> None:
        """Set original image shape and compute transformation metadata.

        Args:
            original_shape: Original image shape (height, width) or (height, width, channels)
        """
        shape_key = (original_shape[0], original_shape[1])

        # Cache metadata to avoid redundant calculations
        if self._cached_original_shape != shape_key:
            self._cached_metadata = calculate_transform_metadata(original_shape, self.target_size)
            self._cached_original_shape = shape_key

    @property
    def metadata(self) -> TransformMetadata | None:
        """Get cached transformation metadata."""
        return self._cached_metadata

    def get_inverse_matrix(self, original_shape: Sequence[int] | None = None) -> np.ndarray:
        """Get inverse transformation matrix (processed → original).

        Args:
            original_shape: Optional original shape (uses cached if not provided)

        Returns:
            3x3 inverse transformation matrix
        """
        if original_shape is not None:
            return compute_inverse_matrix(original_shape, self.target_size)

        if self._cached_metadata is None:
            raise ValueError("No original shape set. Call set_original_shape() first or provide original_shape.")

        return compute_inverse_matrix((self._cached_metadata.original_h, self._cached_metadata.original_w), self.target_size)

    def get_forward_scales(self, original_shape: Sequence[int] | None = None) -> tuple[float, float]:
        """Get forward transformation scales (original → processed).

        Args:
            original_shape: Optional original shape (uses cached if not provided)

        Returns:
            Tuple of (scale_x, scale_y)
        """
        if original_shape is not None:
            return compute_forward_scales(original_shape, self.target_size)

        if self._cached_metadata is None:
            raise ValueError("No original shape set. Call set_original_shape() first or provide original_shape.")

        return compute_forward_scales((self._cached_metadata.original_h, self._cached_metadata.original_w), self.target_size)

    def transform_polygon_forward(self, polygon: np.ndarray, original_shape: Sequence[int] | None = None) -> np.ndarray:
        """Transform polygon from original to processed space.

        Args:
            polygon: Nx2 array of polygon coordinates in original space
            original_shape: Optional original shape (uses cached if not provided)

        Returns:
            Nx2 array of transformed coordinates in processed space
        """
        if original_shape is not None:
            return transform_polygon_to_processed_space(polygon, original_shape, self.target_size)

        if self._cached_metadata is None:
            raise ValueError("No original shape set. Call set_original_shape() first or provide original_shape.")

        return transform_polygon_to_processed_space(
            polygon, (self._cached_metadata.original_h, self._cached_metadata.original_w), self.target_size
        )

    def transform_polygons_string_forward(
        self, polygons_str: str, original_shape: Sequence[int] | None = None, tolerance: float = 2.0
    ) -> str:
        """Transform polygon string from original to processed space.

        Args:
            polygons_str: Pipe-separated polygon strings
            original_shape: Optional original shape (uses cached if not provided)
            tolerance: Tolerance for bounds checking (pixels)

        Returns:
            Transformed polygon string
        """
        if original_shape is not None:
            return transform_polygons_string_to_processed_space(polygons_str, original_shape, self.target_size, tolerance)

        if self._cached_metadata is None:
            raise ValueError("No original shape set. Call set_original_shape() first or provide original_shape.")

        return transform_polygons_string_to_processed_space(
            polygons_str, (self._cached_metadata.original_h, self._cached_metadata.original_w), self.target_size, tolerance
        )
