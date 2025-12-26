from __future__ import annotations

"""Mask-based rectangle fitting logic."""

import math
from typing import Any

import cv2
import numpy as np

from .geometry import _blend_corners, _geometric_synthesis, _intersect_lines, _order_points
from .quality_metrics import (
    _compute_corner_sharpness_deviation,
    _compute_edge_support_metrics,
    _compute_linearity_rmse,
    _compute_parallelism_misalignment,
    _compute_solidity_metrics,
)
from .types import LineQualityReport, MaskRectangleResult
from .validation import (
    _validate_contour_alignment,
    _validate_contour_segments,
    _validate_edge_angles,
    _validate_edge_lengths,
)


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


def _fit_quadrilateral_from_hull(
    hull: np.ndarray,
    eps_start_ratio: float = 0.008,
    eps_growth: float = 1.5,
    eps_max_ratio: float = 0.08,
    max_epsilon_px: float | None = None,
    strict_mode: bool = False,
) -> tuple[np.ndarray | None, float]:
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

    best_candidate: np.ndarray | None = None
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


def _fit_quadrilateral_regression(
    hull: np.ndarray,
    epsilon_px: float = 10.0,
) -> tuple[np.ndarray | None, float]:
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
        else:  # vertical
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
) -> tuple[np.ndarray | None, float]:
    """
    BUG-20251128-001: Stabilize dominant-edge fitting via angle-based bucketing.

    Fit a quadrilateral by:
    1. Running a tight approxPolyDP (keeps 812 hull segments).
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
        segments.append(
            {
                "p1": p1,
                "p2": p2,
                "mid": mid,
                "dx": dx,
                "dy": dy,
                "mid_x": mid_x,
                "mid_y": mid_y,
            }
        )

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

        - If |dx| > |dy|: Horizontal segment -> Top or Bottom (based on y < cy)
        - If |dy| > |dx|: Vertical segment -> Left or Right (based on x < cx)

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
        p1 = seg["p1"]
        p2 = seg["p2"]
        # p1 and p2 are always ndarrays from pts
        assert isinstance(p1, np.ndarray)
        assert isinstance(p2, np.ndarray)
        bins[bin_name].append(p1)
        bins[bin_name].append(p2)

    # Borrow segments if a bin is empty
    # Prefer borrowing from adjacent bins (top<->left/right, bottom<->left/right, etc.)
    adjacent_map = {
        "top": ["left", "right"],
        "bottom": ["left", "right"],
        "left": ["top", "bottom"],
        "right": ["top", "bottom"],
    }

    def find_segment_from_bins(target_bins: list[str]) -> dict | None:
        """Find a segment from the specified bins."""
        for target_bin in target_bins:
            if len(bins[target_bin]) >= 4:  # Has at least 2 points (1 segment)
                for seg in segments:
                    if assign_bin(seg) == target_bin:
                        return seg
        return None

    def borrow_segment_for_bin(bin_name: str) -> None:
        """Try to borrow a segment for the given bin if it's empty."""
        if len(bins[bin_name]) >= 2:
            return

        # Try to borrow from adjacent bins first
        adjacent_bins = adjacent_map.get(bin_name, [])
        best_seg = find_segment_from_bins(adjacent_bins)

        # If no adjacent bin available, borrow from any bin with points
        if best_seg is None:
            other_bins = [b for b in bins.keys() if b != bin_name and len(bins[b]) >= 2]
            best_seg = find_segment_from_bins(other_bins)

        if best_seg is not None:
            bins[bin_name].append(best_seg["p1"])
            bins[bin_name].append(best_seg["p2"])

    for bin_name in bins:
        borrow_segment_for_bin(bin_name)

    def fit_line(points: list[np.ndarray]) -> tuple[float, float, float, float] | None:
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

    # Type assertions for mypy - we've already checked for None above
    assert l_top is not None
    assert l_bottom is not None
    assert l_left is not None
    assert l_right is not None

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


def fit_mask_rectangle(
    mask: np.ndarray,
    min_area_ratio: float = 0.005,
    min_support_points: int = 50,
    edge_support_threshold: float = 0.50,
    edge_support_ratio_floor: float = 0.40,
    edge_support_bins: int = 64,
    edge_support_distance: float = 5.0,
    linearity_rmse_threshold: float = 5.0,
    solidity_threshold: float = 0.85,
    corner_sharpness_threshold: float = 45.0,
    parallelism_threshold: float = 20.0,
    partial_pass_threshold: int = 3,
    blend_weight: float = 0.65,
    max_epsilon_px: float | None = None,
    strict_mode: bool = False,
    use_regression: bool = False,
    regression_epsilon_px: float = 10.0,
    use_dominant_extension: bool = True,
) -> MaskRectangleResult:
    """
    Fit rectangle corners directly from the binary mask.

    This production implementation uses the validated experimental algorithm with
    advanced line quality metrics, dominant extension strategy, and geometric synthesis.
    Achieves d20 pixels data loss (vs 80+ pixels with previous simplified version).

    Args:
        mask: Binary mask (0 background, 255 foreground).
        min_area_ratio: Minimum contour area relative to mask size.
        min_support_points: Minimum contour points required.
        edge_support_threshold: Minimum coverage ratio required per edge.
        edge_support_ratio_floor: Minimum point-ratio per edge (coverage by pixels).
        edge_support_bins: Number of bins for edge coverage estimation.
        edge_support_distance: Maximum distance in pixels to consider support.
        linearity_rmse_threshold: Maximum RMSE allowed for contour-line fit.
        solidity_threshold: Minimum contour/hull solidity required.
        corner_sharpness_threshold: Maximum deviation (degrees) allowed.
        parallelism_threshold: Maximum deviation (degrees) for opposite edges.
        partial_pass_threshold: Minimum heuristics passing to allow blending.
        blend_weight: Weighting applied to fitted corners when blending.
        max_epsilon_px: Maximum epsilon (pixels) for approxPolyDP.
        strict_mode: If True, fail gracefully if approxPolyDP > max_epsilon or < 4 points.
        use_regression: If True, use side-based line regression instead of pure approxPolyDP reduction.
        regression_epsilon_px: Epsilon to use for initial hull approximation in regression mode.
        use_dominant_extension: If True (default), use dominant edge extension logic (validated, minimal data loss).

    Returns:
        MaskRectangleResult with ordered corners (TL, TR, BR, BL) or None on failure.
    """
    binary = _prepare_mask(mask)
    img_h, img_w = binary.shape[:2]
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
        fitted_quad, used_eps = _fit_quadrilateral_dominant_extension(hull, epsilon_px=regression_epsilon_px)
    elif use_regression:
        fitted_quad, used_eps = _fit_quadrilateral_regression(hull, epsilon_px=regression_epsilon_px)
    else:
        fitted_quad, used_eps = _fit_quadrilateral_from_hull(hull, max_epsilon_px=max_epsilon_px, strict_mode=strict_mode)

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
        synthesized_corners = _geometric_synthesis(ordered, ordered_bbox, (img_h, img_w))

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
    validation_failed = False
    validation_reason = None

    if not _validate_edge_angles(ordered, angle_tolerance_deg=45.0):
        validation_failed = True
        validation_reason = "invalid_edge_angles"
    elif not _validate_edge_lengths(ordered, min_aspect_ratio=0.1, max_aspect_ratio=10.0):
        validation_failed = True
        validation_reason = "invalid_edge_proportions"
    elif not _validate_contour_alignment(ordered, largest, mask_bbox, alignment_tolerance=0.25):
        validation_failed = True
        validation_reason = "poor_bbox_alignment"

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
    line_quality_metrics["parallelism_misalignment_deg"] = float(parallelism_misalignment) if parallelism_misalignment is not None else None
    parallel_ok = parallelism_misalignment is not None and parallelism_misalignment <= parallelism_threshold
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


__all__ = ["fit_mask_rectangle"]
