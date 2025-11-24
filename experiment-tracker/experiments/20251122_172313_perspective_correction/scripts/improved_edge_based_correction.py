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
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import stats

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

    return result


def fit_line_to_points(points: np.ndarray) -> tuple[float, float] | None:
    """
    Fit a line to a set of points using least squares.

    Returns line in form: y = mx + b
    Returns (m, b) or None if insufficient points

    Args:
        points: Array of points (N, 2) with columns [x, y]

    Returns:
        Tuple (slope, intercept) or None
    """
    points = ensure_point_array(points)
    if points is None or len(points) < 2:
        return None

    x = points[:, 0]
    y = points[:, 1]

    # Use scipy stats for robust line fitting
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return (slope, intercept)
    except:
        # Fallback: simple least squares
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return (m, b)


def fit_line_ransac(points: np.ndarray, max_iterations: int = 100, threshold: float = 5.0) -> tuple[float, float] | None:
    """
    Fit a line using RANSAC for robustness against outliers.

    BUG-20251124-002: Fixed broadcasting errors with proper array shape validation.

    Args:
        points: Array of points (N, 2)
        max_iterations: Maximum RANSAC iterations
        threshold: Distance threshold for inliers

    Returns:
        Tuple (slope, intercept) or None
    """
    # BUG-20251124-002: Validate input array shape
    points = ensure_point_array(points)
    if points is None or len(points) < 2:
        return None

    if len(points) == 2:
        # Two points define a line
        p1, p2 = points[0], points[1]
        if abs(p2[0] - p1[0]) < 1e-6:  # Vertical line
            return None  # Handle separately
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - m * p1[0]
        return (m, b)

    best_inliers = 0
    best_line = None

    for _ in range(max_iterations):
        # Randomly sample 2 points
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]

        # Skip if points are too close
        if np.linalg.norm(p2 - p1) < 1e-6:
            continue

        # Calculate line equation
        if abs(p2[0] - p1[0]) < 1e-6:  # Vertical line
            # Vertical line: x = constant
            x_const = p1[0]
            # BUG-20251124-002: Fix broadcasting - ensure points is 2D before indexing
            distances = np.abs(points[:, 0] - x_const)
            inliers = np.sum(distances < threshold)
            if inliers > best_inliers:
                best_inliers = inliers
                best_line = ('vertical', x_const)
        else:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            b = p1[1] - m * p1[0]

            # BUG-20251124-002: Fix broadcasting - ensure proper array operations
            # Calculate distances from all points to line
            # Distance from point (x0, y0) to line y = mx + b is |y0 - (mx0 + b)| / sqrt(m^2 + 1)
            distances = np.abs(points[:, 1] - (m * points[:, 0] + b)) / np.sqrt(m**2 + 1)
            inliers = np.sum(distances < threshold)

            if inliers > best_inliers:
                best_inliers = inliers
                best_line = (m, b)

    # If no good line found, use simple least squares
    if best_line is None or best_inliers < len(points) * 0.3:
        return fit_line_to_points(points)

    if isinstance(best_line, tuple) and best_line[0] == 'vertical':
        return None  # Vertical line, handle separately

    return best_line


def intersect_lines(line1: tuple[float, float] | None, line2: tuple[float, float] | None) -> np.ndarray | None:
    """
    Find intersection point of two lines.

    Lines are in form: y = mx + b
    Returns (x, y) or None if lines are parallel

    Args:
        line1: (slope1, intercept1) or None for vertical
        line2: (slope2, intercept2) or None for vertical

    Returns:
        Array [x, y] or None
    """
    if line1 is None or line2 is None:
        return None

    m1, b1 = line1
    m2, b2 = line2

    # Check if parallel
    if abs(m1 - m2) < 1e-6:
        return None

    # Intersection: x = (b2 - b1) / (m1 - m2)
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

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

        # BUG-20251124-002: Check corner proximity for passthrough
        if check_corner_proximity(corners, image_shape, corner_proximity_threshold):
            logger.info("BUG-20251124-002: Skipping correction - corners near boundaries (already corrected)")
            return None
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
        logger.warning("BUG-20251124-002: Rejecting corners - collinear points detected (would cause homography collapse)")
        return None

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
            m, b = line
            color = colors.get(edge_name, (255, 255, 255))
            # Draw line from x=0 to x=w
            x1, y1 = 0, int(b)
            x2, y2 = w-1, int(m * (w-1) + b)
            cv2.line(vis, (x1, y1), (x2, y2), color, 2)

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

