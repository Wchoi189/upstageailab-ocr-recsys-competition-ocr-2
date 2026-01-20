from __future__ import annotations

"""Prediction post-processing utilities."""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np


# from .dependencies import torch

LOGGER = logging.getLogger(__name__)


# BugRef: BUG-20251116-001 — inference inverse-mapping must match padding position
# Report: docs/bug_reports/BUG-20251116-001_DEBUGGING_HANDOVER.md
# Date: 2025-11-17
# IndexSig: pad=top_left; scale=640/max(H0,W0); translation=0
# BUG-2025-001: Original bug report (superseded by BUG-20251116-001)
def compute_inverse_matrix(processed_tensor, original_shape: Sequence[int]):
    """Compute inverse mapping from processed (padded 640x640) tensor space to original image space.

    🚨 CRITICAL FUNCTION - DO NOT MODIFY WITHOUT TESTS

    BUG-20251116-001: This function MUST match the padding position used in transforms.
    The transforms use PadIfNeeded with position="top_left", so padding is at bottom/right only.
    Therefore, there is NO translation needed (translation = 0, 0).

    Assumptions per transforms:
    - LongestMaxSize(max_size=640) followed by PadIfNeeded(640, 640, position="top_left").
    - Therefore, there is no translation (padding is at bottom/right only), and we only need 1/scale.
    - scale = 640 / max(original_h, original_w)

    See: docs/bug_reports/BUG-20251116-001_DEBUGGING_HANDOVER.md

    NOTE: This is a compatibility wrapper. New code should use coordinate_manager module.
    """
    import torch

    if torch is None:
        return [np.eye(3, dtype=np.float32)]

    # Use the new coordinate_manager for consistent transformation logic
    from .coordinate_manager import compute_inverse_matrix as _compute_inverse_matrix

    matrix = _compute_inverse_matrix(original_shape, target_size=640)
    return [matrix]


def decode_polygons_with_head(model, processed_tensor, predictions, original_shape: Sequence[int]) -> dict[str, Any] | None:
    head = getattr(model, "head", None)
    if head is None or not hasattr(head, "get_polygons_from_maps"):
        return None

    inverse_matrix = compute_inverse_matrix(processed_tensor, original_shape)
    batch = {
        "images": processed_tensor,
        "shape": [tuple(original_shape)],
        "filename": ["input"],
        "inverse_matrix": inverse_matrix,
    }

    polygons_result = head.get_polygons_from_maps(batch, predictions)

    if not polygons_result:
        return None

    boxes_batch, scores_batch = polygons_result
    if not boxes_batch:
        return None

    polygons: list[str] = []
    texts: list[str] = []
    confidences: list[float] = []

    for index, box in enumerate(boxes_batch[0]):
        if not box or len(box) < 4:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        polygon_coords = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
        # Competition format uses space-separated coordinates
        polygons.append(" ".join(map(str, polygon_coords)))
        texts.append(f"Text_{index + 1}")
        score = scores_batch[0][index] if scores_batch and index < len(scores_batch[0]) else 0.0
        confidences.append(float(score))

    return {
        "polygons": "|".join(polygons) if polygons else "",
        "texts": texts,
        "confidences": confidences,
    }



