from __future__ import annotations

"""Quality metrics computation for fitted rectangles."""

import math
from typing import Any

import cv2
import numpy as np

from .geometry import _compute_edge_vectors


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

        mask = (proj >= -distance_threshold) & (proj <= edge_len + distance_threshold) & (dist <= distance_threshold)

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
) -> dict[str, float] | None:
    """Measure maximum and mean deviation from 90 degrees using hull neighbors."""
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


def _compute_parallelism_misalignment(corners: np.ndarray) -> float | None:
    """Return maximum angular deviation between opposite edges."""
    if corners is None or len(corners) != 4:
        return None

    edges, _ = _compute_edge_vectors(corners)
    if len(edges) != 4:
        return None

    def _parallel_angle(v1: np.ndarray, v2: np.ndarray) -> float | None:
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


__all__ = [
    "_collect_edge_support_data",
    "_compute_edge_support_metrics",
    "_compute_linearity_rmse",
    "_compute_solidity_metrics",
    "_compute_corner_sharpness_deviation",
    "_compute_parallelism_misalignment",
]
