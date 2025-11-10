"""
AI_DOCS: Polygon Utils - Polygon Processing & Validation Utilities

This module provides specialized polygon processing utilities for OCR:
- Polygon coordinate validation and normalization
- Degenerate polygon detection and filtering
- Shape validation for probability/threshold maps
- Coordinate system transformations

ARCHITECTURE OVERVIEW:
- Utilities extracted from dataset transformation logic
- Focus on geometric validation and processing
- NumPy-based coordinate manipulation
- Integration with Albumentations transforms

DATA CONTRACTS:
- Input: numpy arrays with shape (N, 2) or (1, N, 2)
- Output: validated numpy arrays or None (filtered)
- Coordinate System: (x, y) pixel coordinates
- Data Types: float32 for consistency

CORE CONSTRAINTS:
- ALWAYS validate polygon shapes before processing
- FILTER degenerate polygons (< 3 points)
- PRESERVE coordinate precision (float32)
- USE consistent coordinate ordering (x, y)
- VALIDATE map shapes against image dimensions

PERFORMANCE FEATURES:
- Vectorized NumPy operations for speed
- Early filtering prevents downstream errors
- Memory-efficient coordinate processing
- Batch processing support

VALIDATION REQUIREMENTS:
- Check polygon dimensionality (2D/3D arrays)
- Validate coordinate ranges (non-negative)
- Ensure minimum point counts (≥ 3)
- Verify map-image shape compatibility

RELATED DOCUMENTATION:
- Base Dataset: ocr/datasets/base.py
- Transforms: ocr/datasets/transforms.py
- Data Schemas: ocr/datasets/schemas.py
- Geometric Utils: ocr/utils/geometry_utils.py

MIGRATION NOTES:
- Utilities extracted from ValidatedOCRDataset.__getitem__
- Pydantic integration for data validation
- Improved error handling and filtering
"""

import numpy as np


def ensure_polygon_array(polygon: np.ndarray) -> np.ndarray | None:
    """
    AI_DOCS: Polygon Array Normalization

    Normalizes polygon input to standard numpy array format.

    CRITICAL CONSTRAINTS:
    - ACCEPT both lists and numpy arrays
    - CONVERT to float32 dtype
    - RESHAPE to (N, 2) format if needed
    - VALIDATE coordinate dimensions
    - RETURN None for invalid inputs

    Output: float32 array with shape (N, 2) or None
    """
    if polygon is None:
        return None

    polygon_array = polygon if isinstance(polygon, np.ndarray) else np.asarray(polygon, dtype=np.float32)
    polygon_array = np.asarray(polygon_array, dtype=np.float32)

    if polygon_array.size == 0:
        return polygon_array.reshape(0, 2)

    if polygon_array.ndim == 1:
        if polygon_array.size % 2 != 0:
            raise ValueError("Polygon coordinate list must contain an even number of values")
        return polygon_array.reshape(-1, 2)

    if polygon_array.ndim == 2:
        if polygon_array.shape[1] == 2:
            return polygon_array
        return polygon_array.reshape(-1, 2)

    if polygon_array.ndim == 3 and polygon_array.shape[0] == 1:
        reshaped = polygon_array[0]
        return reshaped if reshaped.ndim == 2 else reshaped.reshape(-1, 2)

    return polygon_array.reshape(-1, 2)


def is_polygon_out_of_bounds(
    polygon: np.ndarray,
    image_width: float,
    image_height: float,
    tolerance: float = 0.5,
) -> bool:
    """
    Check if a polygon has coordinates outside image bounds.

    BUG-20251110-001: Centralized out-of-bounds coordinate checking function.
    This prevents CUDA errors and numerical instability from invalid polygon coordinates.
    See: docs/bug_reports/BUG-20251110-001_out-of-bounds-polygon-coordinates-in-training-dataset.md

    Args:
        polygon: Polygon array with shape (N, 2) or compatible shape
        image_width: Image width in pixels
        image_height: Image height in pixels
        tolerance: Tolerance for floating point errors (default: 0.5)

    Returns:
        True if polygon has out-of-bounds coordinates, False otherwise
    """
    if polygon is None or polygon.size == 0:
        return False

    reshaped = polygon.reshape(-1, 2)
    if reshaped.shape[0] < 1:
        return False

    x_coords = reshaped[:, 0]
    y_coords = reshaped[:, 1]

    # Check if any coordinates are out of bounds
    return (
        x_coords.min() < -tolerance
        or x_coords.max() > image_width + tolerance
        or y_coords.min() < -tolerance
        or y_coords.max() > image_height + tolerance
    )


def clamp_polygon_to_bounds(
    polygon: np.ndarray,
    image_width: float,
    image_height: float,
) -> np.ndarray:
    """
    Clamp polygon coordinates to image bounds.

    BUG-20251110-001: Centralized coordinate clamping function.
    This prevents CUDA errors and numerical instability from invalid polygon coordinates.
    See: docs/bug_reports/BUG-20251110-001_out-of-bounds-polygon-coordinates-in-training-dataset.md

    Args:
        polygon: Polygon array with shape (N, 2) or compatible shape
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Clamped polygon array with coordinates within [0, width] x [0, height]
    """
    if polygon is None or polygon.size == 0:
        return polygon

    reshaped = polygon.reshape(-1, 2).copy()

    # Clamp coordinates to image bounds
    reshaped[:, 0] = np.clip(reshaped[:, 0], 0.0, float(image_width))
    reshaped[:, 1] = np.clip(reshaped[:, 1], 0.0, float(image_height))

    return reshaped


def fix_polygon_with_shapely(polygon: np.ndarray, min_buffer: float = 1e-3) -> np.ndarray | None:
    """
    Attempt to fix a degenerate polygon using Shapely's buffer(0) method.

    This is the "robust" approach: repair invalid polygons before filtering them.
    The buffer(0) operation tidies polygons by:
    - Collapsing zero-span lines
    - Fixing self-intersections
    - Cleaning up messy edges

    For polygons that are "too small", a tiny positive buffer can give them
    a guaranteed minimum area.

    Args:
        polygon: Polygon array with shape (N, 2)
        min_buffer: Minimum buffer size for too-small polygons (default: 1e-3)

    Returns:
        Fixed polygon array with shape (N, 2), or None if fixing failed
    """
    try:
        from shapely.geometry import Polygon
    except ImportError:
        # Shapely not available, return None to fall back to filtering
        return None

    if polygon is None or polygon.size == 0:
        return None

    reshaped = polygon.reshape(-1, 2)
    if reshaped.shape[0] < 3:
        return None

    try:
        # Convert numpy array to Shapely Polygon
        shapely_poly = Polygon(reshaped)

        # Try to fix with buffer(0) first (handles self-intersections, zero-span lines)
        fixed_poly = shapely_poly.buffer(0)

        # If buffer(0) resulted in empty or invalid geometry, try with small positive buffer
        if fixed_poly.is_empty or not fixed_poly.is_valid:
            fixed_poly = shapely_poly.buffer(min_buffer)

        # If still empty or invalid after fixing, return None
        if fixed_poly.is_empty or not fixed_poly.is_valid:
            return None

        # Extract coordinates from fixed polygon
        if hasattr(fixed_poly, 'exterior'):
            coords = np.array(fixed_poly.exterior.coords[:-1], dtype=np.float32)  # Remove duplicate last point
        else:
            # MultiPolygon or other geometry - take the largest part
            if hasattr(fixed_poly, 'geoms'):
                largest = max(fixed_poly.geoms, key=lambda p: p.area)
                coords = np.array(largest.exterior.coords[:-1], dtype=np.float32)
            else:
                return None

        # Ensure we have at least 3 points
        if coords.shape[0] < 3:
            return None

        return coords

    except Exception:
        # Any error during fixing means we should fall back to filtering
        return None


def filter_degenerate_polygons(
    polygons: list[np.ndarray],
    min_side: float = 1.0,
    image_width: float | None = None,
    image_height: float | None = None,
    attempt_fix: bool = True,
) -> list[np.ndarray]:
    """
    AI_DOCS: Degenerate Polygon Filtering with Fixing

    Attempts to fix degenerate polygons using Shapely's buffer(0) method before filtering.
    This is the "robust" approach: repair invalid polygons before discarding them.

    BUG-20251110-001: Added out-of-bounds coordinate checking to filter polygons
    with coordinates exceeding image dimensions. This prevents CUDA errors and
    numerical instability from invalid polygon coordinates.
    See: docs/bug_reports/BUG-20251110-001_out-of-bounds-polygon-coordinates-in-training-dataset.md

    CRITICAL CONSTRAINTS:
    - ATTEMPT to fix polygons using Shapely buffer(0) before filtering
    - REQUIRE minimum 3 points per polygon
    - PRESERVE valid polygons unchanged
    - LOG fixing and filtering decisions for debugging
    - RETURN filtered list (may be shorter)
    - BUG-20251110-001: Filter out-of-bounds polygons if image dimensions provided

    Geometric Requirement: Polygons need ≥ 3 points for area calculation

    Args:
        polygons: List of polygon arrays to process
        min_side: Minimum side length for valid polygons
        image_width: Image width for bounds checking (optional)
        image_height: Image height for bounds checking (optional)
        attempt_fix: Whether to attempt fixing polygons with Shapely before filtering (default: True)
    """
    from collections import Counter

    removed_counts = Counter(
        {
            "too_few_points": 0,
            "too_small": 0,
            "zero_span": 0,
            "empty": 0,
            "none": 0,
            "out_of_bounds": 0,
            "fixed": 0,
        }
    )
    filtered = []
    for polygon in polygons:
        if polygon is None:
            removed_counts["none"] += 1
            continue
        if polygon.size == 0:
            removed_counts["empty"] += 1
            continue

        reshaped = polygon.reshape(-1, 2)
        if reshaped.shape[0] < 3:
            removed_counts["too_few_points"] += 1
            continue

        # Step 1: Always clamp coordinates to image bounds if dimensions provided
        # This ensures all coordinates are valid before any other processing
        # This fixes the root cause: 867 images with out-of-bounds coordinates (26% of dataset)
        was_clamped = False
        if image_width is not None and image_height is not None:
            if is_polygon_out_of_bounds(reshaped, image_width, image_height):
                # Clamp coordinates to image bounds instead of filtering
                # This preserves data while fixing coordinate errors
                reshaped = clamp_polygon_to_bounds(reshaped, image_width, image_height)
                removed_counts["out_of_bounds"] += 1  # Track that we fixed out-of-bounds
                was_clamped = True
                # Continue processing the clamped polygon (may now be degenerate)

        width_span = float(reshaped[:, 0].max() - reshaped[:, 0].min())
        height_span = float(reshaped[:, 1].max() - reshaped[:, 1].min())

        rounded = np.rint(reshaped).astype(np.int32, copy=False)
        width_span_int = int(rounded[:, 0].ptp())
        height_span_int = int(rounded[:, 1].ptp())

        # Step 2: Check if polygon is degenerate after clamping
        # Clipping may have created degenerate shapes (e.g., all points on a single line)
        is_degenerate = (width_span < min_side or height_span < min_side or
                        width_span_int == 0 or height_span_int == 0)

        # Step 3: Use Shapely buffer(0) to fix degenerate shapes created by clipping
        # This repairs polygons that became degenerate after coordinate clamping
        if is_degenerate and attempt_fix:
            fixed_polygon = fix_polygon_with_shapely(reshaped, min_buffer=min_side)
            if fixed_polygon is not None and fixed_polygon.shape[0] >= 3:
                # Re-check the fixed polygon
                fixed_width_span = float(fixed_polygon[:, 0].max() - fixed_polygon[:, 0].min())
                fixed_height_span = float(fixed_polygon[:, 1].max() - fixed_polygon[:, 1].min())
                fixed_rounded = np.rint(fixed_polygon).astype(np.int32, copy=False)
                fixed_width_span_int = int(fixed_rounded[:, 0].ptp())
                fixed_height_span_int = int(fixed_rounded[:, 1].ptp())

                # If fixed polygon is now valid, use it
                if (fixed_width_span >= min_side and fixed_height_span >= min_side and
                    fixed_width_span_int > 0 and fixed_height_span_int > 0):
                    filtered.append(fixed_polygon)
                    removed_counts["fixed"] += 1
                    continue

        # If fixing failed or wasn't attempted, filter based on original checks
        if is_degenerate:
            # Categorize the specific reason for filtering
            if width_span < min_side or height_span < min_side:
                removed_counts["too_small"] += 1
            elif width_span_int == 0 or height_span_int == 0:
                removed_counts["zero_span"] += 1
            continue

        # Polygon is valid (use clamped version if it was clamped)
        filtered.append(reshaped if was_clamped else polygon)

    # Calculate total removed (excluding fixed and clamped polygons, which are kept)
    total_removed = sum(removed_counts.values()) - removed_counts["fixed"] - removed_counts["out_of_bounds"]
    if total_removed > 0 or removed_counts["fixed"] > 0 or removed_counts["out_of_bounds"] > 0:
        import logging

        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.INFO):
            # Include fixed count and clamped_out_of_bounds count in log message
            # Note: out_of_bounds polygons are clamped, not filtered
            logger.info(
                "Processed %d degenerate polygons (fixed=%d, clamped_out_of_bounds=%d, too_few_points=%d, too_small=%d, zero_span=%d, empty=%d, none=%d)",
                total_removed + removed_counts["fixed"] + removed_counts["out_of_bounds"],
                removed_counts["fixed"],
                removed_counts["out_of_bounds"],
                removed_counts["too_few_points"],
                removed_counts["too_small"],
                removed_counts["zero_span"],
                removed_counts["empty"],
                removed_counts["none"],
            )

    return filtered


def validate_map_shapes(
    prob_map: np.ndarray,
    thresh_map: np.ndarray,
    image_height: int | None,
    image_width: int | None,
    filename: str,
) -> bool:
    """
    AI_DOCS: Map Shape Validation

    Validates probability/threshold map dimensions against image.

    CRITICAL CONSTRAINTS:
    - CHECK both maps have identical shapes
    - VALIDATE against image dimensions if provided
    - LOG validation failures with context
    - RETURN boolean validation result

    Map Requirements: (H, W) float arrays matching image dimensions
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Check that both maps exist and are arrays
        if prob_map is None or thresh_map is None:
            logger.warning(f"Map validation failed for {filename}: prob_map or thresh_map is None")
            return False

        # Check shapes are compatible
        if prob_map.shape != thresh_map.shape:
            logger.warning(f"Map validation failed for {filename}: prob_map shape {prob_map.shape} != thresh_map shape {thresh_map.shape}")
            return False

        # Check expected shape format (should be CHW with C=1)
        if len(prob_map.shape) != 3 or prob_map.shape[0] != 1:
            logger.warning(f"Map validation failed for {filename}: prob_map shape {prob_map.shape} should be (1, H, W)")
            return False

        # If image dimensions provided, check they match
        if image_height is not None and image_width is not None:
            expected_shape = (1, image_height, image_width)
            if prob_map.shape != expected_shape:
                logger.warning(
                    f"Map validation failed for {filename}: prob_map shape {prob_map.shape} doesn't match image dimensions {expected_shape}"
                )
                return False

        return True

    except Exception as e:
        logger.warning(f"Map validation failed for {filename}: {e}")
        return False


__all__ = [
    "ensure_polygon_array",
    "is_polygon_out_of_bounds",
    "clamp_polygon_to_bounds",
    "fix_polygon_with_shapely",
    "filter_degenerate_polygons",
    "validate_map_shapes",
]
