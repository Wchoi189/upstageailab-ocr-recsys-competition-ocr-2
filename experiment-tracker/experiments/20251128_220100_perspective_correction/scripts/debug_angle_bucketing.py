#!/usr/bin/env python3
"""
Diagnostic script to visualize angle-bucketing bin assignments for invalid-angle failures.

This helps identify why samples fail _validate_edge_angles by showing:
- Which segments are assigned to which bins (Top/Bottom/Left/Right)
- Segment angles and bin assignments
- Borrowed segments (when bins are empty)
- Final fitted lines and their intersections
"""
import sys
import cv2
import numpy as np
import math
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

# Setup experiment paths - auto-detect tracker root and experiment context
script_path = Path(__file__).resolve()
tracker_root = script_path.parent.parent.parent.parent.parent / "src"
sys.path.insert(0, str(tracker_root))
from experiment_tracker.utils.path_utils import setup_script_paths, ExperimentPaths

# Setup OCR project paths
workspace_root = tracker_root.parent.parent
sys.path.insert(0, str(workspace_root))
from ocr.utils.path_utils import get_path_resolver

# Auto-detect experiment context
TRACKER_ROOT, EXPERIMENT_ID, EXPERIMENT_PATHS = setup_script_paths(script_path)
OCR_RESOLVER = get_path_resolver()

# Import from local script
sys.path.insert(0, str(script_path.parent))

from mask_only_edge_detector import _prepare_mask, _fit_quadrilateral_dominant_extension, _intersect_lines

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _fit_quadrilateral_dominant_extension_with_diagnostics(
    hull: np.ndarray,
    epsilon_px: float = 10.0,
) -> tuple[Optional[np.ndarray], float, Dict[str, Any]]:
    """
    Same as _fit_quadrilateral_dominant_extension but returns diagnostic info.

    Returns: (quad, used_epsilon, diagnostics)
    """
    diagnostics: Dict[str, Any] = {
        "segments": [],
        "bins_initial": {},
        "bins_after_borrow": {},
        "borrowed_segments": {},
        "fitted_lines": {},
        "centroid": None,
    }

    if hull is None or len(hull) < 3:
        return None, 0.0, diagnostics

    eps = epsilon_px
    approx = cv2.approxPolyDP(hull, eps, True)
    if len(approx) < 4:
        return None, eps, diagnostics

    pts = approx.reshape(-1, 2).astype(np.float32)
    num_pts = len(pts)
    centroid = np.mean(pts, axis=0)
    cx, cy = float(centroid[0]), float(centroid[1])
    diagnostics["centroid"] = centroid.tolist()

    segments: List[Dict[str, Any]] = []
    for i in range(num_pts):
        p1 = pts[i]
        p2 = pts[(i + 1) % num_pts]
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        mid = (p1 + p2) / 2.0
        mid_x, mid_y = float(mid[0]), float(mid[1])
        length = float(np.linalg.norm(p2 - p1))
        # Keep angle for visualization/debugging
        vec = mid - centroid
        angle = (math.degrees(math.atan2(vec[1], vec[0])) + 360.0) % 360.0
        seg_info = {
            "p1": p1.tolist(),
            "p2": p2.tolist(),
            "mid": mid.tolist(),
            "dx": dx,
            "dy": dy,
            "mid_x": mid_x,
            "mid_y": mid_y,
            "angle": float(angle),  # For visualization
            "length": length,
            "bin_initial": None,
            "borrowed": False,
        }
        segments.append(seg_info)
        diagnostics["segments"].append(seg_info)

    if not segments:
        return None, eps, diagnostics

    bins: Dict[str, List[np.ndarray]] = {
        "top": [],
        "right": [],
        "bottom": [],
        "left": [],
    }

    def assign_bin(seg: Dict[str, Any]) -> str:
        """
        Classify segment as Top/Bottom/Left/Right based on dominant direction.
        - If |dx| > |dy|: Horizontal segment → Top or Bottom (based on y < cy)
        - If |dy| > |dx|: Vertical segment → Left or Right (based on x < cx)
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

    # Initial binning
    for seg in segments:
        bin_name = assign_bin(seg)
        seg["bin_initial"] = bin_name
        bins[bin_name].append(np.array(seg["p1"]))
        bins[bin_name].append(np.array(seg["p2"]))

    # Record initial bin counts
    for bin_name in bins:
        diagnostics["bins_initial"][bin_name] = len(bins[bin_name])

    # Borrow segments if a bin is empty
    # Prefer borrowing from adjacent bins
    adjacent_map = {
        "top": ["left", "right"],
        "bottom": ["left", "right"],
        "left": ["top", "bottom"],
        "right": ["top", "bottom"],
    }

    borrowed_info: Dict[str, List[int]] = {}
    for bin_name, points in bins.items():
        if len(points) >= 2:
            continue

        # Try to borrow from adjacent bins first
        best_seg_idx = None
        best_source_bin = None

        # First, try adjacent bins
        for adj_bin in adjacent_map.get(bin_name, []):
            if len(bins[adj_bin]) >= 4:  # Has at least 2 points (1 segment)
                # Find a segment from this adjacent bin
                for idx, seg in enumerate(segments):
                    if assign_bin(seg) == adj_bin:
                        best_seg_idx = idx
                        best_source_bin = adj_bin
                        break
                if best_seg_idx is not None:
                    break

        # If no adjacent bin available, borrow from any bin with points
        if best_seg_idx is None:
            for other_bin, other_points in bins.items():
                if other_bin != bin_name and len(other_points) >= 2:
                    for idx, seg in enumerate(segments):
                        if assign_bin(seg) == other_bin:
                            best_seg_idx = idx
                            best_source_bin = other_bin
                            break
                    if best_seg_idx is not None:
                        break

        if best_seg_idx is not None:
            best_seg = segments[best_seg_idx]
            bins[bin_name].append(np.array(best_seg["p1"]))
            bins[bin_name].append(np.array(best_seg["p2"]))
            best_seg["borrowed"] = True
            if bin_name not in borrowed_info:
                borrowed_info[bin_name] = []
            borrowed_info[bin_name].append(best_seg_idx)

    diagnostics["borrowed_segments"] = borrowed_info
    for bin_name in bins:
        diagnostics["bins_after_borrow"][bin_name] = len(bins[bin_name])

    def fit_line(points: List[np.ndarray]) -> Optional[Tuple[float, float, float, float]]:
        if len(points) < 2:
            return None
        pts_array = np.array(points, dtype=np.float32)
        [vx, vy, x, y] = cv2.fitLine(pts_array, cv2.DIST_L12, 0, 0.01, 0.01)
        return float(vx), float(vy), float(x), float(y)

    l_top = fit_line(bins["top"])
    l_bottom = fit_line(bins["bottom"])
    l_left = fit_line(bins["left"])
    l_right = fit_line(bins["right"])

    diagnostics["fitted_lines"] = {
        "top": list(l_top) if l_top else None,
        "bottom": list(l_bottom) if l_bottom else None,
        "left": list(l_left) if l_left else None,
        "right": list(l_right) if l_right else None,
    }

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
        return bbox_quad, eps, diagnostics

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
        return bbox_quad, eps, diagnostics

    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    diagnostics["corners"] = {
        "tl": tl.tolist() if tl is not None else None,
        "tr": tr.tolist() if tr is not None else None,
        "br": br.tolist() if br is not None else None,
        "bl": bl.tolist() if bl is not None else None,
    }
    return quad, eps, diagnostics


def visualize_bin_assignments(
    mask: np.ndarray,
    hull: np.ndarray,
    diagnostics: Dict[str, Any],
    fitted_quad: Optional[np.ndarray],
) -> np.ndarray:
    """
    Create a visualization showing:
    - Mask (gray background)
    - Hull (blue outline)
    - Segments color-coded by bin (Top=red, Right=green, Bottom=blue, Left=yellow)
    - Borrowed segments (thicker lines)
    - Fitted lines (extended)
    - Final quad corners (cyan)
    - Centroid (white circle)
    """
    binary = _prepare_mask(mask)
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Color scheme for bins
    bin_colors = {
        "top": (0, 0, 255),      # Red
        "right": (0, 255, 0),    # Green
        "bottom": (255, 0, 0),   # Blue
        "left": (0, 255, 255),   # Yellow
    }

    # Draw hull
    if hull is not None and len(hull) > 0:
        cv2.drawContours(vis, [hull], -1, (128, 128, 128), 1)

    # Draw centroid
    centroid = diagnostics.get("centroid")
    if centroid:
        cv2.circle(vis, (int(centroid[0]), int(centroid[1])), 5, (255, 255, 255), -1)

    # Draw segments color-coded by bin
    segments = diagnostics.get("segments", [])
    for seg in segments:
        p1 = (int(seg["p1"][0]), int(seg["p1"][1]))
        p2 = (int(seg["p2"][0]), int(seg["p2"][1]))
        bin_name = seg.get("bin_initial", "unknown")
        color = bin_colors.get(bin_name, (128, 128, 128))
        thickness = 3 if seg.get("borrowed", False) else 1
        cv2.line(vis, p1, p2, color, thickness)

        # Label segment with angle
        mid = seg.get("mid", [0, 0])
        angle = seg.get("angle", 0.0)
        cv2.putText(
            vis,
            f"{angle:.0f}°",
            (int(mid[0]) + 5, int(mid[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
        )

    # Draw fitted lines (extended)
    fitted_lines = diagnostics.get("fitted_lines", {})
    for bin_name, line_params in fitted_lines.items():
        if line_params is None:
            continue
        vx, vy, x, y = line_params
        color = bin_colors.get(bin_name, (128, 128, 128))
        # Extend line across image
        if abs(vx) > 1e-6:
            t1 = (0 - x) / vx
            t2 = (vis.shape[1] - x) / vx
        else:
            t1 = (0 - y) / vy
            t2 = (vis.shape[0] - y) / vy
        t_min = min(t1, t2)
        t_max = max(t1, t2)
        pt1 = (int(x + vx * t_min), int(y + vy * t_min))
        pt2 = (int(x + vx * t_max), int(y + vy * t_max))
        cv2.line(vis, pt1, pt2, color, 2)

    # Draw final quad
    if fitted_quad is not None:
        pts = fitted_quad.astype(int).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, (255, 255, 0), 2)
        for idx, corner in enumerate(fitted_quad):
            cv2.circle(vis, tuple(corner.astype(int)), 6, (255, 255, 0), -1)
            cv2.putText(
                vis,
                str(idx),
                (int(corner[0]) + 8, int(corner[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

    # Add legend
    y_offset = 20
    for bin_name, color in bin_colors.items():
        cv2.rectangle(vis, (10, y_offset - 10), (30, y_offset + 5), color, -1)
        cv2.putText(
            vis,
            bin_name.upper(),
            (35, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y_offset += 20

    return vis


def debug_angle_bucketing(
    worst_case_dir: str,
    output_dir: str,
    target_ids: Optional[List[str]] = None,
) -> None:
    """
    Debug angle bucketing for specific samples (focus on invalid-angle failures).

    Args:
        worst_case_dir: Directory containing mask files
        output_dir: Output directory for diagnostic visualizations
        target_ids: List of image IDs to debug (default: invalid-angle failures)
    """
    if target_ids is None:
        # Default to invalid-angle failure cases from assessment
        target_ids = [
            "drp.en_ko.in_house.selectstar_000078",
            "drp.en_ko.in_house.selectstar_000101",
            "drp.en_ko.in_house.selectstar_000140",
            "drp.en_ko.in_house.selectstar_000145",
            "drp.en_ko.in_house.selectstar_000153",
        ]

    worst_case_path = Path(worst_case_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Debugging angle bucketing for {len(target_ids)} samples")
    logger.info(f"Output directory: {output_path}")

    for img_id in target_ids:
        mask_file = worst_case_path / f"{img_id}_mask.jpg"
        if not mask_file.exists():
            logger.warning(f"Mask file not found: {mask_file}")
            continue

        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error(f"Could not load mask: {mask_file}")
            continue

        # Get hull
        binary = _prepare_mask(mask)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning(f"No contours found for {img_id}")
            continue

        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)

        # Run fitting with diagnostics
        fitted_quad, used_eps, diagnostics = _fit_quadrilateral_dominant_extension_with_diagnostics(
            hull, epsilon_px=10.0
        )

        # Create visualization
        vis = visualize_bin_assignments(mask, hull, diagnostics, fitted_quad)

        # Save visualization
        output_file = output_path / f"{img_id}_bin_debug.jpg"
        cv2.imwrite(str(output_file), vis)
        logger.info(f"Saved diagnostic visualization: {output_file}")

        # Print summary
        logger.info(f"\n{img_id} Diagnostics:")
        logger.info(f"  Segments: {len(diagnostics['segments'])}")
        logger.info(f"  Initial bin counts: {diagnostics['bins_initial']}")
        logger.info(f"  After borrow: {diagnostics['bins_after_borrow']}")
        logger.info(f"  Borrowed segments: {diagnostics['borrowed_segments']}")


if __name__ == "__main__":
    worst_case_dir = str(OCR_RESOLVER.config.output_dir / "improved_edge_approach" / "worst_force_improved")
    output_dir = str(EXPERIMENT_PATHS.get_artifacts_path() if EXPERIMENT_PATHS else Path.cwd() / "artifacts")

    debug_angle_bucketing(worst_case_dir, output_dir)

