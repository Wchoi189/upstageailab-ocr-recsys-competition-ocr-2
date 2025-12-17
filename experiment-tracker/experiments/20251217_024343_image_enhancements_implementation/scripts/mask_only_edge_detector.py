#!/usr/bin/env python3
"""
Mask-only rectangle fitter for perspective correction.

Operates exclusively on rembg-produced binary masks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class MaskRectangleResult:
    corners: Optional[np.ndarray]
    raw_corners: Optional[np.ndarray]
    contour_area: float
    hull_area: float
    mask_area: float
    contour: Optional[np.ndarray]
    hull: Optional[np.ndarray]
    reason: Optional[str] = None
    line_quality: Optional["LineQualityReport"] = None
    used_epsilon: Optional[float] = None


@dataclass
class LineQualityReport:
    decision: str
    metrics: dict[str, Any]
    passes: dict[str, bool]
    fail_reasons: list[str]


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

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    # Skip background (index 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask

    largest_idx = 1 + int(np.argmax(areas))
    largest_mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)
    return largest_mask


def _geometric_synthesis(
    fitted_quad: np.ndarray,
    bbox_corners: np.ndarray,
    image_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Geometric Synthesis: Intersect fitted_quad with bbox_corners using bitwise operations.

    Creates a mask from fitted_quad, clips it to bbox_corners, and extracts the final contour.
    This ensures the output never exceeds the safe bounding box limits while preserving
    the regression-based shape.

    Args:
        fitted_quad: Fitted quadrilateral corners (4 points)
        bbox_corners: Bounding box corners (4 points, axis-aligned)
        image_shape: (height, width) of the image

    Returns:
        Final quadrilateral corners after intersection, or None if synthesis fails
    """
    if fitted_quad is None or bbox_corners is None:
        return None

    h, w = image_shape

    # Create blank canvas
    fitted_mask = np.zeros((h, w), dtype=np.uint8)
    bbox_mask = np.zeros((h, w), dtype=np.uint8)

    # Draw fitted_quad on canvas
    fitted_pts = fitted_quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(fitted_mask, [fitted_pts], 255)

    # Draw bbox_corners on canvas
    bbox_pts = bbox_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(bbox_mask, [bbox_pts], 255)

    # Compute intersection: Final_Mask = Fitted_Mask AND Bbox_Mask
    final_mask = cv2.bitwise_and(fitted_mask, bbox_mask)

    # Extract contours from final mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate as quadrilateral
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) < 4:
        # If approximation fails, use convex hull
        hull = cv2.convexHull(largest_contour)
        if len(hull) < 4:
            return None
        # Try to get 4 corners from hull
        epsilon = 0.05 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) < 4:
            return None

    # Ensure we have exactly 4 points
    if len(approx) > 4:
        # Take the 4 most extreme points
        approx = approx.reshape(-1, 2)
        # Find bounding box of approx
        x_min = np.min(approx[:, 0])
        x_max = np.max(approx[:, 0])
        y_min = np.min(approx[:, 1])
        y_max = np.max(approx[:, 1])
        # Create 4 corners
        approx = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ], dtype=np.float32)
    elif len(approx) == 4:
        approx = approx.reshape(-1, 2).astype(np.float32)
    else:
        return None

    return _order_points(approx)


def _order_points(points: np.ndarray) -> np.ndarray:
    """Order quadrilateral corners as TL, TR, BR, BL."""
    pts = np.asarray(points, dtype=np.float32)
    centers = np.mean(pts, axis=0)
    def angle(pt):
        return math.atan2(pt[1] - centers[1], pt[0] - centers[0])
    sorted_pts = sorted(pts, key=angle)
    # Convert to consistent order by rotating so smallest y+x first
    arr = np.array(sorted_pts, dtype=np.float32)
    sums = arr.sum(axis=1)
    start_idx = int(np.argmin(sums))
    ordered = np.roll(arr, -start_idx, axis=0)
    return ordered


def _compute_edge_vectors(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute edge vectors and lengths from ordered corners."""
    edges = np.diff(corners, axis=0, append=corners[0:1])
    lengths = np.linalg.norm(edges, axis=1)
    return edges, lengths


def _validate_edge_angles(corners: np.ndarray, angle_tolerance_deg: float = 15.0) -> bool:
    """
    Validate that edges form approximately right angles (rectangular shape).

    Args:
        corners: Ordered quadrilateral corners (4, 2)
        angle_tolerance_deg: Maximum deviation from 90° allowed (default: 15°)

    Returns:
        True if all angles are approximately 90°, False otherwise
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

        # Check deviation from 90°
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


def _fit_quadrilateral_from_hull(
    hull: np.ndarray,
    eps_start_ratio: float = 0.008,
    eps_growth: float = 1.5,
    eps_max_ratio: float = 0.08,
    max_epsilon_px: Optional[float] = None,
    strict_mode: bool = False,
) -> tuple[Optional[np.ndarray], float]:
    """
    Fit a quadrilateral to the convex hull using adaptive approxPolyDP.

    Returns (ordered points, used_epsilon) or (None, 0.0) if a 4-point fit cannot be found.
    """
    if hull is None or len(hull) < 4:
        return None, 0.0

    hull_closed = hull.copy()
    peri = cv2.arcLength(hull_closed, True)
    if peri < 1e-3:
        return None, 0.0

    eps = max(peri * eps_start_ratio, 1.0)
    eps_limit = peri * eps_max_ratio

    if max_epsilon_px is not None:
        eps_limit = min(eps_limit, max_epsilon_px)
        # Ensure start epsilon does not exceed the hard cap
        eps = min(eps, eps_limit)

    eps_max = max(eps_limit, eps)

    best_candidate: Optional[np.ndarray] = None
    best_eps: float = 0.0

    while eps <= eps_max:
        approx = cv2.approxPolyDP(hull_closed, eps, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype(np.float32), eps

        if len(approx) > 4:
            if best_candidate is None or len(approx) < len(best_candidate):
                best_candidate = approx
                best_eps = eps
        eps *= eps_growth

    # Strict mode: if we didn't find exactly 4 points, fail.
    if strict_mode:
        return None, 0.0

    # Could not reach exactly 4 points; attempt to downsample best candidate
    if best_candidate is not None and len(best_candidate) >= 4:
        points = best_candidate.reshape(-1, 2).astype(np.float32)
        idxs = np.linspace(0, len(points) - 1, num=4, dtype=int)
        sampled = points[idxs]
        return sampled, best_eps

    return None, 0.0


def _intersect_lines(
    line1: tuple[float, float, float, float],
    line2: tuple[float, float, float, float]
) -> Optional[np.ndarray]:
    """
    Find intersection of two lines given in (vx, vy, x0, y0) format.
    Returns (x, y) or None if parallel.
    """
    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2

    # Cross product of direction vectors to check parallelism
    det = vx1 * vy2 - vx2 * vy1
    if abs(det) < 1e-6:
        return None

    # Solve for t1:
    # x1 + t1*vx1 = x2 + t2*vx2  => t1*vx1 - t2*vx2 = x2 - x1
    # y1 + t1*vy1 = y2 + t2*vy2  => t1*vy1 - t2*vy2 = y2 - y1

    dx = x2 - x1
    dy = y2 - y1

    t1 = (dx * vy2 - dy * vx2) / det

    ix = x1 + t1 * vx1
    iy = y1 + t1 * vy1

    return np.array([ix, iy], dtype=np.float32)


def _fit_quadrilateral_regression(
    hull: np.ndarray,
    epsilon_px: float = 10.0,
) -> tuple[Optional[np.ndarray], float]:
    """
    Fit quadrilateral by approximating hull with low epsilon, classifying sides,
    regressing lines, and solving intersections.

    Returns (ordered points, used_epsilon)
    """
    if hull is None or len(hull) < 3:
        return None, 0.0

    # 1. Tight Approximation
    # Use fixed epsilon for regression preparation
    eps = epsilon_px
    approx = cv2.approxPolyDP(hull, eps, True)

    # If approxPolyDP reduces too much (e.g. < 4 points), fallback to hull directly
    if len(approx) < 4:
        approx = hull

    pts = approx.reshape(-1, 2).astype(np.float32)
    centroid = np.mean(pts, axis=0)
    cx, cy = centroid

    # 2. Side Classification
    top_points = []
    bottom_points = []
    left_points = []
    right_points = []

    num_pts = len(pts)
    # Using segments rather than just points helps capture the geometry better
    # But for regression, we regress on POINTS.
    # We will classify segments, and add both endpoints to the respective list.

    for i in range(num_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % num_pts]
        mid = (p1 + p2) / 2

        # Determine orientation
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx)) % 360

        # Horizontal: [315, 360], [0, 45], [135, 225]
        # Vertical: [45, 135], [225, 315]

        is_horz = (angle >= 315 or angle <= 45) or (angle >= 135 and angle <= 225)

        if is_horz:
            if mid[1] < cy:
                top_points.append(p1)
                top_points.append(p2)
            else:
                bottom_points.append(p1)
                bottom_points.append(p2)
        else: # vertical
            if mid[0] < cx:
                left_points.append(p1)
                left_points.append(p2)
            else:
                right_points.append(p1)
                right_points.append(p2)

    # 3. Line Regression
    # Need at least 2 points (1 segment) to fit a line
    if len(top_points) < 2 or len(bottom_points) < 2 or len(left_points) < 2 or len(right_points) < 2:
        return None, eps

    def fit_line(points):
        pts_array = np.array(points, dtype=np.float32)
        # fitLine returns (vx, vy, x0, y0)
        # DIST_L12 is robust
        [vx, vy, x, y] = cv2.fitLine(pts_array, cv2.DIST_L12, 0, 0.01, 0.01)
        return float(vx), float(vy), float(x), float(y)

    l_top = fit_line(top_points)
    l_bottom = fit_line(bottom_points)
    l_left = fit_line(left_points)
    l_right = fit_line(right_points)

    # 4. Intersection Solver
    tl = _intersect_lines(l_top, l_left)
    tr = _intersect_lines(l_top, l_right)
    br = _intersect_lines(l_bottom, l_right)
    bl = _intersect_lines(l_bottom, l_left)

    if tl is None or tr is None or br is None or bl is None:
        return None, eps

    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    return quad, eps


def _fit_quadrilateral_dominant_extension(
    hull: np.ndarray,
    epsilon_px: float = 10.0,
) -> tuple[Optional[np.ndarray], float]:
    """
    BUG-20251128-001: Stabilize dominant-edge fitting via angle-based bucketing.

    Fit a quadrilateral by:
    1. Running a tight approxPolyDP (keeps 8–12 hull segments).
    2. Binning *all* segments into Top/Bottom/Left/Right using horizontal/vertical classification.
    3. Regressing consensus lines per bin via cv2.fitLine and intersecting them.

    Uses `abs(dx) > abs(dy)` classification to avoid coordinate system confusion and
    adapt to different aspect ratios.

    Returns (ordered points, used_epsilon)
    """
    if hull is None or len(hull) < 3:
        return None, 0.0

    eps = epsilon_px
    approx = cv2.approxPolyDP(hull, eps, True)
    if len(approx) < 4:
        return None, eps

    pts = approx.reshape(-1, 2).astype(np.float32)
    num_pts = len(pts)
    centroid = np.mean(pts, axis=0)
    cx, cy = float(centroid[0]), float(centroid[1])

    segments: list[dict[str, np.ndarray | float]] = []
    for i in range(num_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % num_pts]
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        mid = (p1 + p2) / 2.0
        mid_x, mid_y = float(mid[0]), float(mid[1])
        segments.append({
            "p1": p1,
            "p2": p2,
            "mid": mid,
            "dx": dx,
            "dy": dy,
            "mid_x": mid_x,
            "mid_y": mid_y,
        })

    if not segments:
        return None, eps

    bins: dict[str, list[np.ndarray]] = {
        "top": [],
        "right": [],
        "bottom": [],
        "left": [],
    }

    def assign_bin(seg: dict[str, Any]) -> str:
        """
        Classify segment as Top/Bottom/Left/Right based on dominant direction.

        - If |dx| > |dy|: Horizontal segment → Top or Bottom (based on y < cy)
        - If |dy| > |dx|: Vertical segment → Left or Right (based on x < cx)

        This avoids coordinate system confusion and adapts to aspect ratios.
        """
        dx, dy = abs(seg["dx"]), abs(seg["dy"])
        mid_x, mid_y = seg["mid_x"], seg["mid_y"]

        if dx > dy:
            # Horizontal segment: Top or Bottom
            if mid_y < cy:
                return "top"
            else:
                return "bottom"
        else:
            # Vertical segment: Left or Right
            if mid_x < cx:
                return "left"
            else:
                return "right"

    for seg in segments:
        bin_name = assign_bin(seg)
        bins[bin_name].append(seg["p1"])
        bins[bin_name].append(seg["p2"])

    # Borrow segments if a bin is empty
    # Prefer borrowing from adjacent bins (top<->left/right, bottom<->left/right, etc.)
    adjacent_map = {
        "top": ["left", "right"],
        "bottom": ["left", "right"],
        "left": ["top", "bottom"],
        "right": ["top", "bottom"],
    }

    for bin_name, points in bins.items():
        if len(points) >= 2:
            continue

        # Try to borrow from adjacent bins first
        best_seg = None
        best_source_bin = None

        # First, try adjacent bins
        for adj_bin in adjacent_map.get(bin_name, []):
            if len(bins[adj_bin]) >= 4:  # Has at least 2 points (1 segment)
                # Find a segment from this adjacent bin
                for seg in segments:
                    if assign_bin(seg) == adj_bin:
                        best_seg = seg
                        best_source_bin = adj_bin
                        break
                if best_seg is not None:
                    break

        # If no adjacent bin available, borrow from any bin with points
        if best_seg is None:
            for other_bin, other_points in bins.items():
                if other_bin != bin_name and len(other_points) >= 2:
                    for seg in segments:
                        if assign_bin(seg) == other_bin:
                            best_seg = seg
                            best_source_bin = other_bin
                            break
                    if best_seg is not None:
                        break

        if best_seg is not None:
            bins[bin_name].append(best_seg["p1"])
            bins[bin_name].append(best_seg["p2"])

    def fit_line(points: list[np.ndarray]) -> Optional[tuple[float, float, float, float]]:
        if len(points) < 2:
            return None
        pts_array = np.array(points, dtype=np.float32)
        [vx, vy, x, y] = cv2.fitLine(pts_array, cv2.DIST_L12, 0, 0.01, 0.01)
        return float(vx), float(vy), float(x), float(y)

    l_top = fit_line(bins["top"])
    l_bottom = fit_line(bins["bottom"])
    l_left = fit_line(bins["left"])
    l_right = fit_line(bins["right"])

    if None in (l_top, l_bottom, l_left, l_right):
        min_x = float(np.min(pts[:, 0]))
        max_x = float(np.max(pts[:, 0]))
        min_y = float(np.min(pts[:, 1]))
        max_y = float(np.max(pts[:, 1]))
        bbox_quad = np.array(
            [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ],
            dtype=np.float32,
        )
        return bbox_quad, eps

    tl = _intersect_lines(l_top, l_left)
    tr = _intersect_lines(l_top, l_right)
    br = _intersect_lines(l_bottom, l_right)
    bl = _intersect_lines(l_bottom, l_left)

    if tl is None or tr is None or br is None or bl is None:
        min_x = float(np.min(pts[:, 0]))
        max_x = float(np.max(pts[:, 0]))
        min_y = float(np.min(pts[:, 1]))
        max_y = float(np.max(pts[:, 1]))
        bbox_quad = np.array(
            [
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y],
            ],
            dtype=np.float32,
        )
        return bbox_quad, eps

    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    return quad, eps


def _collect_edge_support_data(
    corners: np.ndarray,
    contour: np.ndarray,
    distance_threshold: float,
) -> list[dict[str, Any]]:
    """Collect per-edge point projections/distances for downstream heuristics."""
    if corners is None or contour is None or len(contour) == 0:
        return []

    contour_pts = np.squeeze(contour, axis=1).astype(np.float32)
    if contour_pts.ndim != 2 or contour_pts.shape[0] == 0:
        return []

    total_points = contour_pts.shape[0]
    edge_data: list[dict[str, Any]] = []

    for idx in range(4):
        p0 = corners[idx]
        p1 = corners[(idx + 1) % 4]
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1e-3:
            edge_data.append(
                {
                    "edge_len": edge_len,
                    "proj": np.array([], dtype=np.float32),
                    "dist": np.array([], dtype=np.float32),
                    "total_points": total_points,
                }
            )
            continue

        dir_vec = edge_vec / edge_len
        rel = contour_pts - p0
        proj = rel @ dir_vec
        perp = rel - np.outer(proj, dir_vec)
        dist = np.linalg.norm(perp, axis=1)

        mask = (
            (proj >= -distance_threshold)
            & (proj <= edge_len + distance_threshold)
            & (dist <= distance_threshold)
        )

        edge_data.append(
            {
                "edge_len": edge_len,
                "proj": proj[mask],
                "dist": dist[mask],
                "total_points": total_points,
            }
        )

    return edge_data


def _compute_edge_support_metrics(
    corners: np.ndarray,
    contour: np.ndarray,
    bins: int = 64,
    distance_threshold: float = 5.0,
) -> dict[str, Any]:
    """Compute edge coverage/support metrics based on contour proximity."""
    bins = max(4, bins)
    data = _collect_edge_support_data(corners, contour, distance_threshold)
    if not data:
        zero_list = [0.0, 0.0, 0.0, 0.0]
        return {
            "edge_support_coverage_per_edge": zero_list,
            "edge_support_points_ratio_per_edge": zero_list,
            "edge_support_coverage_min": 0.0,
            "edge_support_coverage_mean": 0.0,
        }

    coverage_values: list[float] = []
    ratio_values: list[float] = []

    for entry in data:
        edge_len = max(entry["edge_len"], 1e-6)
        proj = entry["proj"]
        total_points = max(entry["total_points"], 1)
        support_ratio = float(len(proj)) / float(total_points)
        ratio_values.append(support_ratio)

        if len(proj) == 0:
            coverage_values.append(0.0)
            continue

        normalized = np.clip(proj / edge_len, 0.0, 1.0)
        if normalized.size == 0:
            coverage_values.append(0.0)
            continue

        bin_indices = np.clip((normalized * bins).astype(int), 0, bins - 1)
        unique_bins = np.unique(bin_indices)
        coverage = float(len(unique_bins)) / float(bins)
        coverage_values.append(coverage)

    min_cov = float(min(coverage_values)) if coverage_values else 0.0
    mean_cov = float(np.mean(coverage_values)) if coverage_values else 0.0

    return {
        "edge_support_coverage_per_edge": [float(c) for c in coverage_values],
        "edge_support_points_ratio_per_edge": [float(r) for r in ratio_values],
        "edge_support_coverage_min": min_cov,
        "edge_support_coverage_mean": mean_cov,
    }


def _compute_linearity_rmse(
    corners: np.ndarray,
    contour: np.ndarray,
    distance_threshold: float = 5.0,
) -> dict[str, Any]:
    """Compute RMSE of contour distances to each fitted edge."""
    data = _collect_edge_support_data(corners, contour, distance_threshold)
    if not data:
        return {
            "linearity_rmse_per_edge": [None, None, None, None],
            "linearity_rmse_max": None,
            "linearity_rmse_mean": None,
        }

    rmses: list[float] = []
    for entry in data:
        dist = entry["dist"]
        if len(dist) == 0:
            rmses.append(None)
            continue
        rmses.append(float(np.sqrt(np.mean(np.square(dist)))))

    finite_rmses = [r for r in rmses if r is not None and np.isfinite(r)]
    max_rmse = float(max(finite_rmses)) if finite_rmses else None
    mean_rmse = float(np.mean(finite_rmses)) if finite_rmses else None

    return {
        "linearity_rmse_per_edge": [float(r) if r is not None else None for r in rmses],
        "linearity_rmse_max": max_rmse,
        "linearity_rmse_mean": mean_rmse,
    }


def _compute_solidity_metrics(
    contour_area: float,
    hull_area: float,
    rect_area: float,
) -> dict[str, float]:
    """Compute solidity/rectangularity style metrics."""
    solidity = 0.0
    if hull_area > 1e-6:
        solidity = contour_area / hull_area

    rectangularity = 0.0
    if rect_area > 1e-6:
        rectangularity = contour_area / rect_area

    return {
        "solidity": float(solidity),
        "rectangularity": float(rectangularity),
    }


def _compute_corner_sharpness_deviation(
    corners: np.ndarray,
    hull: np.ndarray,
    neighborhood: int = 6,
) -> Optional[dict[str, float]]:
    """Measure maximum and mean deviation from 90° using hull neighbors."""
    if corners is None or hull is None or len(hull) < 4:
        return None

    hull_pts = np.squeeze(hull, axis=1).astype(np.float32)
    if hull_pts.ndim != 2 or hull_pts.shape[0] < 4:
        return None

    deviations: list[float] = []
    hull_len = hull_pts.shape[0]

    for corner in corners:
        dists = np.linalg.norm(hull_pts - corner, axis=1)
        idx = int(np.argmin(dists))
        prev_idx = (idx - neighborhood) % hull_len
        next_idx = (idx + neighborhood) % hull_len
        prev_vec = hull_pts[prev_idx] - corner
        next_vec = hull_pts[next_idx] - corner

        if np.linalg.norm(prev_vec) < 1e-3 or np.linalg.norm(next_vec) < 1e-3:
            continue

        cos_val = np.clip(
            np.dot(prev_vec, next_vec) / (np.linalg.norm(prev_vec) * np.linalg.norm(next_vec)),
            -1.0,
            1.0,
        )
        angle = math.degrees(math.acos(cos_val))
        deviations.append(abs(angle - 90.0))

    if not deviations:
        return None

    return {
        "corner_sharpness_max_deviation": float(max(deviations)),
        "corner_sharpness_mean_deviation": float(np.mean(deviations)),
    }


def _compute_parallelism_misalignment(corners: np.ndarray) -> Optional[float]:
    """Return maximum angular deviation between opposite edges."""
    if corners is None or len(corners) != 4:
        return None

    edges, _ = _compute_edge_vectors(corners)
    if len(edges) != 4:
        return None

    def _parallel_angle(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(math.degrees(math.acos(abs(cos_val))))

    top_bottom = _parallel_angle(edges[0], edges[2])
    left_right = _parallel_angle(edges[1], edges[3])

    deviations = [val for val in (top_bottom, left_right) if val is not None]
    if not deviations:
        return None
    return float(max(deviations))


def _blend_corners(
    primary: np.ndarray,
    fallback: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Blend two ordered quadrilaterals."""
    weight = float(np.clip(weight, 0.0, 1.0))
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    return (weight * primary + (1.0 - weight) * fallback).astype(np.float32)


def fit_mask_rectangle(
    mask: np.ndarray,
    min_area_ratio: float = 0.005,
    min_support_points: int = 50,
    edge_support_threshold: float = 0.50,  # Loosened from 0.90
    edge_support_ratio_floor: float = 0.40, # Loosened from 0.80
    edge_support_bins: int = 64,
    edge_support_distance: float = 5.0,
    linearity_rmse_threshold: float = 5.0, # Loosened from 2.0
    solidity_threshold: float = 0.85, # Loosened from 0.95
    corner_sharpness_threshold: float = 45.0, # Loosened from 15.0 to allow perspective
    parallelism_threshold: float = 20.0, # Loosened from 5.0 to allow perspective
    partial_pass_threshold: int = 3,
    blend_weight: float = 0.65,
    max_epsilon_px: Optional[float] = None,
    strict_mode: bool = False,
    use_regression: bool = False,
    regression_epsilon_px: float = 10.0,
    use_dominant_extension: bool = False,
) -> MaskRectangleResult:
    """
    Fit rectangle corners directly from the binary mask.

    Args:
        mask: rembg binary mask (0 background, 255 foreground).
        min_area_ratio: minimum contour area relative to mask size.
        min_support_points: minimum contour points required.
        edge_support_threshold: minimum coverage ratio required per edge.
        edge_support_ratio_floor: minimum point-ratio per edge (coverage by pixels).
        edge_support_bins: number of bins for edge coverage estimation.
        edge_support_distance: maximum distance in pixels to consider support.
        linearity_rmse_threshold: maximum RMSE allowed for contour-line fit.
        solidity_threshold: minimum contour/hull solidity required.
        corner_sharpness_threshold: maximum deviation (degrees) allowed.
        parallelism_threshold: maximum deviation (degrees) for opposite edges.
        partial_pass_threshold: minimum heuristics passing to allow blending.
        blend_weight: weighting applied to fitted corners when blending.
        max_epsilon_px: maximum epsilon (pixels) for approxPolyDP.
        strict_mode: if True, fail gracefully if approxPolyDP > max_epsilon or < 4 points.
        use_regression: if True, use side-based line regression instead of pure approxPolyDP reduction.
        regression_epsilon_px: epsilon to use for initial hull approximation in regression mode.
        use_dominant_extension: if True, use dominant edge extension logic (ignores short segments/folds).
    """
    binary = _prepare_mask(mask)
    img_h, img_w = binary.shape[:2]  # Image dimensions (preserved to avoid shadowing)
    total_pixels = float(img_h * img_w)

    # Reinforce with closing to fill small gaps and ensure a collar
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    largest_component = _extract_largest_component(closed)

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

    if contour_area < total_pixels * min_area_ratio or len(largest) < min_support_points:
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

    # Get mask bbox for fallback
    x, y, w, h = cv2.boundingRect(largest_component)
    mask_bbox = (x, y, w, h)
    bbox_corners = np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )
    ordered_bbox = _order_points(bbox_corners)

    # Validate contour structure before fitting
    # If contour is too fragmented, fall back to mask bbox
    if not _validate_contour_segments(largest):
        return MaskRectangleResult(
            corners=ordered_bbox,
            raw_corners=None,
            contour_area=contour_area,
            hull_area=0.0,
            mask_area=mask_area,
            contour=largest,
            hull=None,
            reason="contour_too_fragmented_using_bbox",
        )

    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))

    if use_dominant_extension:
        fitted_quad, used_eps = _fit_quadrilateral_dominant_extension(
            hull,
            epsilon_px=regression_epsilon_px
        )
    elif use_regression:
        fitted_quad, used_eps = _fit_quadrilateral_regression(
            hull,
            epsilon_px=regression_epsilon_px
        )
    else:
        fitted_quad, used_eps = _fit_quadrilateral_from_hull(
            hull,
            max_epsilon_px=max_epsilon_px,
            strict_mode=strict_mode
        )

    if fitted_quad is None:
        ordered = None
    else:
        ordered = _order_points(fitted_quad)

    if ordered is None or len(ordered) != 4:
        reason = "quad_fit_failed_using_bbox"
        if use_dominant_extension:
            reason = "dominant_extension_failed_using_bbox"
        elif use_regression:
            reason = "regression_failed_using_bbox"

        return MaskRectangleResult(
            corners=ordered_bbox,
            raw_corners=None,
            contour_area=contour_area,
            hull_area=hull_area,
            mask_area=mask_area,
            contour=largest,
            hull=hull,
            reason=reason,
            used_epsilon=used_eps,
        )

    # Geometric Synthesis Approach for regression-based fits
    # Skip strict validation and use intersection logic instead
    if use_dominant_extension or use_regression:
        # Only validate basic geometry (angles, non-degenerate)
        if not _validate_edge_angles(ordered, angle_tolerance_deg=45.0):
            return MaskRectangleResult(
                corners=ordered_bbox,
                raw_corners=ordered,
                contour_area=contour_area,
                hull_area=hull_area,
                mask_area=mask_area,
                contour=largest,
                hull=hull,
                reason="validation_failed_invalid_edge_angles_using_bbox",
                used_epsilon=used_eps,
            )

        if not _validate_edge_lengths(ordered, min_aspect_ratio=0.1, max_aspect_ratio=10.0):
            return MaskRectangleResult(
                corners=ordered_bbox,
                raw_corners=ordered,
                contour_area=contour_area,
                hull_area=hull_area,
                mask_area=mask_area,
                contour=largest,
                hull=hull,
                reason="validation_failed_invalid_edge_proportions_using_bbox",
                used_epsilon=used_eps,
            )

        # Apply geometric synthesis: intersect fitted_quad with bbox_corners
        # Use image dimensions (img_h, img_w) not bounding box dimensions (h, w)
        synthesized_corners = _geometric_synthesis(
            ordered,
            ordered_bbox,
            (img_h, img_w)
        )

        if synthesized_corners is not None and len(synthesized_corners) == 4:
            # Trust regression results - accept synthesized corners
            return MaskRectangleResult(
                corners=synthesized_corners,
                raw_corners=ordered,
                contour_area=contour_area,
                hull_area=hull_area,
                mask_area=mask_area,
                contour=largest,
                hull=hull,
                reason=None,  # Success - no reason needed
                used_epsilon=used_eps,
            )
        else:
            # If synthesis fails, use fitted quad directly (still better than bbox)
            return MaskRectangleResult(
                corners=ordered,
                raw_corners=ordered,
                contour_area=contour_area,
                hull_area=hull_area,
                mask_area=mask_area,
                contour=largest,
                hull=hull,
                reason="geometric_synthesis_failed_using_fitted",
                used_epsilon=used_eps,
            )

    # Traditional validation path for non-regression fits
    # Validate fitted rectangle quality using heuristics
    validation_failed = False
    validation_reason = None

    if not _validate_edge_angles(ordered, angle_tolerance_deg=45.0): # Loosened from 15.0
        validation_failed = True
        validation_reason = "invalid_edge_angles"
    elif not _validate_edge_lengths(ordered, min_aspect_ratio=0.1, max_aspect_ratio=10.0):
        validation_failed = True
        validation_reason = "invalid_edge_proportions"
    elif not _validate_contour_alignment(ordered, largest, mask_bbox, alignment_tolerance=0.25): # Loosened from 0.15
        validation_failed = True
        validation_reason = "poor_bbox_alignment"

    # If validation fails, fall back to mask bounding box
    if validation_failed:
        return MaskRectangleResult(
            corners=ordered_bbox,
            raw_corners=ordered,
            contour_area=contour_area,
            hull_area=hull_area,
            mask_area=mask_area,
            contour=largest,
            hull=hull,
            reason=f"validation_failed_{validation_reason}_using_bbox",
            used_epsilon=used_eps,
        )

    # Advanced line-quality heuristics
    rect_area = float(abs(cv2.contourArea(ordered)))

    line_quality_metrics: dict[str, Any] = {}
    line_quality_passes: dict[str, bool] = {}

    # Adjust distance threshold based on fitting epsilon
    # The fitted quad is within `used_eps` of the hull.
    # Validation should allow for this deviation + user margin.
    adaptive_threshold = max(edge_support_distance, used_eps + 2.0)

    edge_support = _compute_edge_support_metrics(
        ordered,
        largest,
        bins=edge_support_bins,
        distance_threshold=adaptive_threshold,
    )
    line_quality_metrics.update(edge_support)
    edge_support_min = edge_support.get("edge_support_coverage_min", 0.0)
    line_quality_passes["edge_support"] = bool(edge_support_min >= edge_support_threshold)
    per_edge_ratios = edge_support.get("edge_support_points_ratio_per_edge", [])
    ratio_pass = all(r is not None and r >= edge_support_ratio_floor for r in per_edge_ratios)
    line_quality_passes["edge_support_ratio"] = bool(ratio_pass)

    linearity_metrics = _compute_linearity_rmse(
        ordered,
        largest,
        distance_threshold=adaptive_threshold,
    )
    line_quality_metrics.update(linearity_metrics)
    linearity_max = linearity_metrics.get("linearity_rmse_max")
    linearity_ok = linearity_max is not None and linearity_max <= linearity_rmse_threshold
    line_quality_passes["linearity_rmse"] = bool(linearity_ok)

    solidity_metrics = _compute_solidity_metrics(contour_area, hull_area, rect_area)
    line_quality_metrics.update(solidity_metrics)
    solidity_value = solidity_metrics.get("solidity", 0.0)
    line_quality_passes["solidity"] = bool(solidity_value >= solidity_threshold)

    corner_metrics = _compute_corner_sharpness_deviation(ordered, hull)
    if corner_metrics:
        line_quality_metrics.update(corner_metrics)
        corner_max = corner_metrics.get("corner_sharpness_max_deviation")
    else:
        corner_max = None
        line_quality_metrics["corner_sharpness_max_deviation"] = None
        line_quality_metrics["corner_sharpness_mean_deviation"] = None
    corner_ok = corner_max is not None and corner_max <= corner_sharpness_threshold
    line_quality_passes["corner_sharpness"] = bool(corner_ok)

    parallelism_misalignment = _compute_parallelism_misalignment(ordered)
    line_quality_metrics["parallelism_misalignment_deg"] = (
        float(parallelism_misalignment) if parallelism_misalignment is not None else None
    )
    parallel_ok = (
        parallelism_misalignment is not None
        and parallelism_misalignment <= parallelism_threshold
    )
    line_quality_passes["parallelism"] = bool(parallel_ok)

    total_checks = max(len(line_quality_passes), 1)
    pass_count = sum(1 for passed in line_quality_passes.values() if passed)
    adjusted_partial_threshold = min(partial_pass_threshold, total_checks - 1) if total_checks > 1 else 1
    fail_reasons = [name for name, passed in line_quality_passes.items() if not passed]

    if pass_count == total_checks:
        decision = "accept"
    elif pass_count >= adjusted_partial_threshold:
        decision = "blend"
    else:
        decision = "fallback"

    line_quality_report = LineQualityReport(
        decision=decision,
        metrics=line_quality_metrics,
        passes=line_quality_passes,
        fail_reasons=fail_reasons,
    )

    if decision == "accept":
        final_corners = ordered
        final_reason = None
    elif decision == "blend":
        final_corners = _blend_corners(ordered, ordered_bbox, blend_weight)
        final_reason = "line_quality_partial_blend"
    else:
        final_corners = ordered_bbox
        fail_suffix = "_".join(fail_reasons) if fail_reasons else "line_quality"
        final_reason = f"line_quality_fallback_{fail_suffix}"

    return MaskRectangleResult(
        corners=final_corners,
        raw_corners=ordered,
        contour_area=contour_area,
        hull_area=hull_area,
        mask_area=mask_area,
        contour=largest,
        hull=hull,
        reason=final_reason,
        line_quality=line_quality_report,
        used_epsilon=used_eps,
    )


def visualize_mask_fit(
    mask: np.ndarray,
    corners: np.ndarray,
    contour: np.ndarray,
    hull: np.ndarray,
) -> np.ndarray:
    """Render the mask, contour, hull, and fitted rectangle for inspection."""
    binary = _prepare_mask(mask)
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    if contour is not None and len(contour) > 0:
        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 1)

    if hull is not None and len(hull) > 0:
        cv2.drawContours(vis, [hull], -1, (255, 0, 0), 1)

    if corners is not None:
        pts = corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
        for idx, corner in enumerate(corners):
            cv2.circle(vis, tuple(corner.astype(int)), 6, (0, 255, 255), -1)
            cv2.putText(vis, str(idx), (int(corner[0]) + 8, int(corner[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return vis
