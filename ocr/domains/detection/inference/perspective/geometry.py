from __future__ import annotations

"""Geometric calculations for perspective correction."""

import math

import cv2
import numpy as np


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


def _intersect_lines(line1: tuple[float, float, float, float], line2: tuple[float, float, float, float]) -> np.ndarray | None:
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


def _geometric_synthesis(
    fitted_quad: np.ndarray,
    bbox_corners: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
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
        approx = np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ],
            dtype=np.float32,
        )
    elif len(approx) == 4:
        approx = approx.reshape(-1, 2).astype(np.float32)
    else:
        return None

    return _order_points(approx)


__all__ = [
    "_order_points",
    "_compute_edge_vectors",
    "_intersect_lines",
    "_blend_corners",
    "_geometric_synthesis",
]
