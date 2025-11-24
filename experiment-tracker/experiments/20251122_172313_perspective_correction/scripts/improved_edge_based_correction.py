#!/usr/bin/env python3
"""
Improved perspective correction using multi-point edge detection and line fitting.

This approach:
1. Extracts edge points from rembg mask
2. Groups points by edge (top, right, bottom, left)
3. Fits lines to each edge using multiple points
4. Finds corner intersections from line intersections
"""

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# BUG-20251124-002: Validation / Utility Functions
# ============================================================================

def calculate_background_ratio(mask: np.ndarray) -> float:
    """
    Calculate background to total area ratio.

    BUG-20251124-002: Background threshold check for passthrough condition.

    Args:
        mask: Binary mask (0=background, 255=foreground)

    Returns:
        Background ratio (0.0 to 1.0)
    """
    total_pixels = mask.size
    background_pixels = np.sum(mask == 0)
    return background_pixels / total_pixels if total_pixels > 0 else 0.0


def mask_bounding_box_corners(mask: np.ndarray) -> np.ndarray:
    """
    Return corners of the tight bounding box around the foreground mask.
    """
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        h, w = mask.shape[:2]
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    ys = coords[:, 0]
    xs = coords[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=np.float32,
    )


def check_corner_proximity(corners: np.ndarray, image_shape: tuple[int, int], threshold: int = 20) -> bool:
    """
    Check if corners are within threshold pixels of image boundaries.

    BUG-20251124-002: Passthrough condition - skip correction if already aligned.

    Args:
        corners: Array of 4 corner points (4, 2)
        image_shape: (height, width)
        threshold: Pixel distance threshold (default: 20)

    Returns:
        True if all corners are near boundaries (image already corrected)
    """
    h, w = image_shape[:2]

    for corner in corners:
        x, y = corner[0], corner[1]
        # Check if corner is near any boundary
        near_left = x <= threshold
        near_right = x >= (w - threshold)
        near_top = y <= threshold
        near_bottom = y >= (h - threshold)

        # If corner is not near any boundary, return False
        if not (near_left or near_right or near_top or near_bottom):
            return False

    return True


def check_collinearity(corners: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Check if corners are collinear (would cause singular homography matrix).

    BUG-20251124-002: Prevent homography collapse from collinear corners.

    Args:
        corners: Array of 4 corner points (4, 2)
        threshold: Angle threshold in radians (default: 0.1 ≈ 5.7 degrees)

    Returns:
        True if corners are collinear (should reject)
    """
    if len(corners) < 3:
        return False

    # Check if any three corners are collinear
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            for k in range(j + 1, len(corners)):
                p1, p2, p3 = corners[i], corners[j], corners[k]

                # Calculate vectors
                v1 = p2 - p1
                v2 = p3 - p1

                # Check if vectors are parallel (cross product near zero)
                cross_product = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 > 1e-6 and norm2 > 1e-6:
                    # Normalized cross product (sine of angle)
                    sin_angle = cross_product / (norm1 * norm2)
                    if sin_angle < threshold:
                        return True  # Collinear

    return False


def validate_homography_matrix(matrix: np.ndarray, condition_threshold: float = 1e10) -> tuple[bool, float]:
    """
    Validate homography matrix condition number.

    BUG-20251124-002: Prevent homography collapse from ill-conditioned matrices.

    Args:
        matrix: 3x3 homography matrix
        condition_threshold: Maximum condition number (default: 1e10)

    Returns:
        Tuple (is_valid, condition_number)
    """
    try:
        condition_number = np.linalg.cond(matrix)
        is_valid = condition_number < condition_threshold
        return (is_valid, condition_number)
    except:
        return (False, np.inf)


def ensure_point_array(points: np.ndarray) -> np.ndarray | None:
    """
    Ensure point array has shape (N, 2).

    BUG-20251124-002: Prevent broadcasting errors by rejecting invalid shapes.
    """
    if points is None:
        return None

    pts = np.asarray(points, dtype=np.float32)

    if pts.ndim == 1:
        if pts.size % 2 != 0:
            logger.warning(
                "BUG-20251124-002: Invalid point array length (%s) for reshaping to (N, 2)",
                pts.size,
            )
            return None
        pts = pts.reshape(-1, 2)

    if pts.ndim != 2 or pts.shape[1] != 2:
        logger.warning(
            "BUG-20251124-002: Invalid point array shape %s, expected (N, 2)",
            pts.shape,
        )
        return None

    return pts


def reinforce_mask(mask: np.ndarray, collar: int = 3) -> np.ndarray:
    """
    Enforce a thin background collar around the document object.

    BUG-20251124-002: Ensures continuous background to aid edge detection even
    when rembg output lacks visible margins on certain sides.
    """
    collar = max(1, collar)
    kernel = np.ones((collar, collar), np.uint8)

    # Treat background as white for morphology operations
    background = ((mask == 0).astype(np.uint8) * 255)
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel)
    expanded_bg = cv2.dilate(background, kernel, iterations=1)

    # Convert back to standard mask format (document=255, background=0)
    reinforced = np.where(expanded_bg == 255, 0, 255).astype(np.uint8)
    return reinforced


def _select_outer_band(points: np.ndarray, axis: int, keep_high: bool, band_size: float, min_points: int) -> np.ndarray:
    """
    Keep the outermost band of points along the requested axis.

    Args:
        points: Input points (N, 2)
        axis: 0 for x, 1 for y
        keep_high: True keeps the highest values, False keeps lowest
        band_size: Absolute band size in pixels
        min_points: Minimum number of points required to accept the filtered band
    """
    if len(points) == 0:
        return points

    values = points[:, axis]
    if keep_high:
        boundary = np.max(values)
        mask = values >= (boundary - band_size)
    else:
        boundary = np.min(values)
        mask = values <= (boundary + band_size)

    filtered = points[mask]
    if len(filtered) >= max(min_points, int(len(points) * 0.2)):
        return filtered
    return points


def enforce_single_edge_per_side(edge_groups: dict[str, np.ndarray], image_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """
    Ensure each side retains a single dominant band of points (one edge per side).
    """
    h, w = image_shape[:2]
    band_y = max(10, h * 0.12)
    band_x = max(10, w * 0.12)
    refined = {}

    for edge_name, points in edge_groups.items():
        pts = points
        if edge_name == 'top':
            pts = _select_outer_band(points, axis=1, keep_high=False, band_size=band_y, min_points=25)
        elif edge_name == 'bottom':
            pts = _select_outer_band(points, axis=1, keep_high=True, band_size=band_y, min_points=25)
        elif edge_name == 'left':
            pts = _select_outer_band(points, axis=0, keep_high=False, band_size=band_x, min_points=25)
        elif edge_name == 'right':
            pts = _select_outer_band(points, axis=0, keep_high=True, band_size=band_x, min_points=25)

        refined[edge_name] = pts

    return refined


def general_line_from_points(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float] | None:
    """
    Return normalized line coefficients (a, b, c) for the line passing through p1 and p2.
    Equation: a*x + b*y + c = 0 with sqrt(a^2 + b^2) = 1.
    """
    if p1 is None or p2 is None:
        return None

    if np.linalg.norm(p2 - p1) < 1e-6:
        return None

    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0] * p2[1] - p2[0] * p1[1]

    norm = math.hypot(a, b)
    if norm < 1e-6:
        return None

    return (a / norm, b / norm, c / norm)


def intersection_with_bounds(line: tuple[float, float, float], image_shape: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Compute intersections between a line (ax + by + c = 0) and image bounds.
    Returns list of points within the image.
    """
    if line is None:
        return []

    a, b, c = line
    h, w = image_shape[:2]
    points: list[tuple[int, int]] = []

    # x = 0
    if abs(b) > 1e-6:
        y = (-c - a * 0) / b
        if 0 <= y < h:
            points.append((0, int(round(y))))

    # x = w-1
    if abs(b) > 1e-6:
        x = w - 1
        y = (-c - a * x) / b
        if 0 <= y < h:
            points.append((x, int(round(y))))

    # y = 0
    if abs(a) > 1e-6:
        x = (-c - b * 0) / a
        if 0 <= x < w:
            points.append((int(round(x)), 0))

    # y = h-1
    if abs(a) > 1e-6:
        y = h - 1
        x = (-c - b * y) / a
        if 0 <= x < w:
            points.append((int(round(x)), y))

    # Deduplicate while preserving order
    dedup = []
    for pt in points:
        if pt not in dedup:
            dedup.append(pt)

    return dedup[:2]


def fit_line_least_squares(points: np.ndarray) -> tuple[float, float, float] | None:
    """
    Fit a line in general form using PCA on the point cloud.
    """
    points = ensure_point_array(points)
    if points is None or len(points) < 2:
        return None

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    if centered.shape[0] < 2:
        return None

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    normal = np.array([-direction[1], direction[0]])

    a, b = normal
    norm = math.hypot(a, b)
    if norm < 1e-6:
        return None
    c = -(a * centroid[0] + b * centroid[1])
    return (a / norm, b / norm, c / norm)


def extract_edge_points_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Extract edge points from binary mask.

    Uses Canny edge detection on the mask to get clean edge points.

    Args:
        mask: Binary mask (0=background, 255=foreground)

    Returns:
        Array of edge points (N, 2)
    """
    h, w = mask.shape[:2]

    # BUG-20251124-002: Reinforce mask to guarantee thin background collar
    mask = reinforce_mask(mask)

    # Pad with background to ensure edges exist even when document touches borders
    pad = 4
    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    padded = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Use Canny to detect edges in the padded mask
    edges = cv2.Canny(padded, 50, 150)

    # Find all edge pixels
    edge_pixels = np.column_stack(np.where(edges > 0))

    # Convert from (row, col) to (x, y)
    if len(edge_pixels) > 0:
        # Subtract padding to map back to original coordinates
        cols = edge_pixels[:, 1] - pad
        rows = edge_pixels[:, 0] - pad

        valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
        cols = cols[valid]
        rows = rows[valid]

        if len(cols) == 0:
            return np.array([])

        edge_points = np.column_stack([cols, rows]).astype(np.float32)
    else:
        # Fallback: use contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            edge_points = largest_contour.reshape(-1, 2).astype(np.float32)
        else:
            return np.array([])

    return edge_points


def group_edge_points(points: np.ndarray, image_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """
    Group edge points into 4 edges: top, right, bottom, left.

    Uses a combination of position and angle to robustly group points.

    Args:
        points: Array of edge points (N, 2)
        image_shape: (height, width) of image

    Returns:
        Dictionary with keys: 'top', 'right', 'bottom', 'left'
        Each value is array of points for that edge
    """
    points = ensure_point_array(points)
    if points is None or len(points) == 0:
        return {'top': np.array([]), 'right': np.array([]), 'bottom': np.array([]), 'left': np.array([])}

    h, w = image_shape[:2]

    # Calculate center
    center_x = w / 2
    center_y = h / 2

    # Use position-based grouping (more robust for rectangular documents)
    # Divide image into 4 quadrants with some overlap margin
    margin = min(h, w) * 0.15  # 15% margin for overlap

    # Top edge: points in upper portion of image
    top_y_threshold = center_y - margin
    top_mask = points[:, 1] < top_y_threshold

    # Bottom edge: points in lower portion of image
    bottom_y_threshold = center_y + margin
    bottom_mask = points[:, 1] > bottom_y_threshold

    # Left edge: points in left portion of image
    left_x_threshold = center_x - margin
    left_mask = points[:, 0] < left_x_threshold

    # Right edge: points in right portion of image
    right_x_threshold = center_x + margin
    right_mask = points[:, 0] > right_x_threshold

    # For points in the middle region, use angle from center
    middle_mask = ~(top_mask | bottom_mask | left_mask | right_mask)
    if np.any(middle_mask):
        middle_points = points[middle_mask]
        angles = []
        for point in middle_points:
            dx = point[0] - center_x
            dy = point[1] - center_y
            angle = np.arctan2(dy, dx) * 180 / np.pi
            angles.append(angle)
        angles = np.array(angles)

        # Assign middle points based on angle
        top_angle_mask = ((angles >= -135) & (angles < -45)) | ((angles >= 225) & (angles < 315))
        right_angle_mask = (angles >= -45) & (angles < 45)
        bottom_angle_mask = (angles >= 45) & (angles < 135)
        left_angle_mask = (angles >= 135) & (angles < 225)

        # Add middle points to respective groups
        top_mask = top_mask | (middle_mask & top_angle_mask)
        right_mask = right_mask | (middle_mask & right_angle_mask)
        bottom_mask = bottom_mask | (middle_mask & bottom_angle_mask)
        left_mask = left_mask | (middle_mask & left_angle_mask)

    # BUG-20251124-002: Fix broadcasting errors - ensure all arrays are properly shaped
    # Convert boolean masks to indices and ensure arrays are 2D
    result = {}
    for edge_name, mask_array in [
        ('top', top_mask),
        ('right', right_mask),
        ('bottom', bottom_mask),
        ('left', left_mask),
    ]:
        if np.any(mask_array):
            edge_points = ensure_point_array(points[mask_array])
            if edge_points is None:
                result[edge_name] = np.array([], dtype=np.float32).reshape(0, 2)
            else:
                result[edge_name] = edge_points
        else:
            result[edge_name] = np.array([], dtype=np.float32).reshape(0, 2)

    return enforce_single_edge_per_side(result, image_shape)


def fit_line_to_points(points: np.ndarray) -> tuple[float, float, float] | None:
    """
    Fit a line to a set of points using least squares (general form ax + by + c = 0).
    """
    return fit_line_least_squares(points)


def fit_line_ransac(points: np.ndarray, max_iterations: int = 100, threshold: float = 5.0) -> tuple[float, float, float] | None:
    """
    Fit a line using RANSAC for robustness against outliers.

    BUG-20251124-002: Fixed broadcasting errors with proper array shape validation.

    Args:
        points: Array of points (N, 2)
        max_iterations: Maximum RANSAC iterations
        threshold: Distance threshold for inliers

    Returns:
        Tuple (a, b, c) representing ax + by + c = 0, or None
    """
    # BUG-20251124-002: Validate input array shape
    points = ensure_point_array(points)
    if points is None or len(points) < 2:
        return None

    if len(points) == 2:
        return general_line_from_points(points[0], points[1])

    best_inliers = 0
    best_line = None

    for _ in range(max_iterations):
        idx = np.random.choice(len(points), 2, replace=False)
        candidate = general_line_from_points(points[idx[0]], points[idx[1]])
        if candidate is None:
            continue

        a, b, c = candidate
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        inliers = np.sum(distances < threshold)

        if inliers > best_inliers:
            best_inliers = inliers
            best_line = candidate

    # If no good line found, use simple least squares
    if best_line is None or best_inliers < len(points) * 0.3:
        return fit_line_to_points(points)

    return best_line


def intersect_lines(line1: tuple[float, float, float] | None, line2: tuple[float, float, float] | None) -> np.ndarray | None:
    """
    Find intersection of two lines represented as ax + by + c = 0.
    """
    if line1 is None or line2 is None:
        return None

    a1, b1, c1 = line1
    a2, b2, c2 = line2

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None

    x = (b1 * (-c2) - b2 * (-c1)) / det
    y = (a2 * (-c1) - a1 * (-c2)) / det
    return np.array([x, y], dtype=np.float32)


def fit_quadrilateral_from_edges(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    use_ransac: bool = True,
    background_threshold: float = 0.05,
    corner_proximity_threshold: int = 20,
) -> np.ndarray | None:
    """
    Fit quadrilateral by detecting edges and fitting lines.

    BUG-20251124-002: Added passthrough conditions and validation checks.

    Process:
    1. Check background threshold (passthrough condition)
    2. Extract edge points from mask
    3. Group points by edge (top, right, bottom, left)
    4. Fit line to each edge using multiple points
    5. Find corner intersections from line intersections
    6. Validate corners (collinearity, proximity)

    Args:
        mask: Binary mask (0=background, 255=foreground)
        image_shape: (height, width) of image
        use_ransac: Whether to use RANSAC for line fitting (more robust)
        background_threshold: Skip correction if background < threshold (default: 0.05 = 5%)
        corner_proximity_threshold: Skip if corners within N pixels of boundaries (default: 20)

    Returns:
        Array of 4 corner points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] or None if passthrough
    """
    # BUG-20251124-002: Passthrough condition - check background threshold
    background_ratio = calculate_background_ratio(mask)
    if background_ratio < background_threshold:
        logger.info(f"BUG-20251124-002: Skipping correction - background ratio {background_ratio:.2%} < {background_threshold:.0%}")
        return None  # Signal to skip correction

    # Step 1: Extract edge points
    edge_points = extract_edge_points_from_mask(mask)
    edge_points = ensure_point_array(edge_points)

    if edge_points is None or len(edge_points) < 4:
        # Fallback: use bounding box
        h, w = image_shape[:2]
        corners = np.array([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1],
        ], dtype=np.float32)

        # BUG-20251124-002: Bounding box fallback must bypass proximity check so
        # we still return a quadrilateral when edges are missing.
        return corners

    # Step 2: Group points by edge
    edge_groups = group_edge_points(edge_points, image_shape)

    # Step 3: Fit lines to each edge
    lines = {}
    for edge_name, points in edge_groups.items():
        if len(points) >= 2:
            if use_ransac:
                line = fit_line_ransac(points)
            else:
                line = fit_line_to_points(points)
            lines[edge_name] = line
        else:
            lines[edge_name] = None

    # Step 4: Find corner intersections
    corners = []

    # Top-left: top and left lines
    corner_tl = intersect_lines(lines.get('top'), lines.get('left'))
    if corner_tl is not None:
        corners.append(corner_tl)
    else:
        # Fallback: use extreme point
        if len(edge_groups['top']) > 0 and len(edge_groups['left']) > 0:
            top_min_x = edge_groups['top'][np.argmin(edge_groups['top'][:, 0])]
            left_min_y = edge_groups['left'][np.argmin(edge_groups['left'][:, 1])]
            corners.append(np.array([top_min_x[0], left_min_y[1]], dtype=np.float32))
        else:
            corners.append(np.array([0, 0], dtype=np.float32))

    # Top-right: top and right lines
    corner_tr = intersect_lines(lines.get('top'), lines.get('right'))
    if corner_tr is not None:
        corners.append(corner_tr)
    else:
        if len(edge_groups['top']) > 0 and len(edge_groups['right']) > 0:
            top_max_x = edge_groups['top'][np.argmax(edge_groups['top'][:, 0])]
            right_min_y = edge_groups['right'][np.argmin(edge_groups['right'][:, 1])]
            corners.append(np.array([top_max_x[0], right_min_y[1]], dtype=np.float32))
        else:
            h, w = image_shape[:2]
            corners.append(np.array([w-1, 0], dtype=np.float32))

    # Bottom-right: bottom and right lines
    corner_br = intersect_lines(lines.get('bottom'), lines.get('right'))
    if corner_br is not None:
        corners.append(corner_br)
    else:
        if len(edge_groups['bottom']) > 0 and len(edge_groups['right']) > 0:
            bottom_max_x = edge_groups['bottom'][np.argmax(edge_groups['bottom'][:, 0])]
            right_max_y = edge_groups['right'][np.argmax(edge_groups['right'][:, 1])]
            corners.append(np.array([bottom_max_x[0], right_max_y[1]], dtype=np.float32))
        else:
            h, w = image_shape[:2]
            corners.append(np.array([w-1, h-1], dtype=np.float32))

    # Bottom-left: bottom and left lines
    corner_bl = intersect_lines(lines.get('bottom'), lines.get('left'))
    if corner_bl is not None:
        corners.append(corner_bl)
    else:
        if len(edge_groups['bottom']) > 0 and len(edge_groups['left']) > 0:
            bottom_min_x = edge_groups['bottom'][np.argmin(edge_groups['bottom'][:, 0])]
            left_max_y = edge_groups['left'][np.argmax(edge_groups['left'][:, 1])]
            corners.append(np.array([bottom_min_x[0], left_max_y[1]], dtype=np.float32))
        else:
            h, w = image_shape[:2]
            corners.append(np.array([0, h-1], dtype=np.float32))

    corners = np.array(corners, dtype=np.float32)

    # Clamp to image bounds
    h, w = image_shape[:2]
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    # BUG-20251124-002: Validate corners before returning
    # Check collinearity (would cause singular homography matrix)
    if check_collinearity(corners):
        logger.warning("BUG-20251124-002: Collinear corners detected - falling back to mask bounding box")
        corners = mask_bounding_box_corners(mask).astype(np.float32)

    # Check corner proximity for passthrough condition
    if check_corner_proximity(corners, image_shape, corner_proximity_threshold):
        logger.info("BUG-20251124-002: Skipping correction - corners near boundaries (already corrected)")
        return None

    return corners


def visualize_edges_and_lines(
    mask: np.ndarray,
    edge_groups: dict[str, np.ndarray],
    lines: dict[str, tuple[float, float] | None],
    corners: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Create visualization of edge points, fitted lines, and corners.

    Args:
        mask: Binary mask
        edge_groups: Dictionary of edge point groups
        lines: Dictionary of fitted lines
        corners: 4 corner points
        image_shape: (height, width)

    Returns:
        Visualization image (BGR)
    """
    # Create color image from mask
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Draw edge points with different colors
    colors = {
        'top': (0, 255, 0),      # Green
        'right': (255, 0, 0),   # Blue
        'bottom': (0, 0, 255),  # Red
        'left': (255, 255, 0),  # Cyan
    }

    for edge_name, points in edge_groups.items():
        if len(points) > 0:
            color = colors.get(edge_name, (255, 255, 255))
            for point in points:
                pt = tuple(point.astype(int))
                cv2.circle(vis, pt, 2, color, -1)

    # Draw fitted lines
    h, w = image_shape[:2]
    for edge_name, line in lines.items():
        if line is not None:
            color = colors.get(edge_name, (255, 255, 255))
            pts = intersection_with_bounds(line, image_shape)
            if len(pts) == 2:
                cv2.line(vis, pts[0], pts[1], color, 2)

    # Draw corners
    for i, corner in enumerate(corners):
        pt = tuple(corner.astype(int))
        cv2.circle(vis, pt, 10, (0, 255, 255), -1)
        cv2.putText(vis, str(i), (pt[0] + 15, pt[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Draw quadrilateral
    pts = corners.astype(int).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], True, (255, 255, 255), 2)

    return vis

