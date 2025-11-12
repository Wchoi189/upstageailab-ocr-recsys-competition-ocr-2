from __future__ import annotations

"""Prediction post-processing utilities."""

import logging
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from .config_loader import PostprocessSettings
from .dependencies import torch

LOGGER = logging.getLogger(__name__)


# BugRef: BUG-2025-001 — inference inverse-mapping uses pre-pad scale
# Report: docs/bug_reports/BUG-2025-001_inference_padding_scaling_mismatch.md
# Date: 2025-10-20
# IndexSig: pad=top_left; scale=640/max(H0,W0); translation=0
def compute_inverse_matrix(processed_tensor, original_shape: Sequence[int]):
    """Compute inverse mapping from processed (padded 640x640) tensor space to original image space.

    Assumptions per transforms:
    - LongestMaxSize(max_size=640) followed by PadIfNeeded(640, 640, position="top_left").
    - Therefore, there is no translation (padding is at bottom/right only), and we only need 1/scale.
    - scale = 640 / max(original_h, original_w)
    """
    if torch is None:
        return [np.eye(3, dtype=np.float32)]

    # original_height = int(original_shape[0])
    # original_width = int(original_shape[1])
    # if original_height <= 0 or original_width <= 0:
    #     return [np.eye(3, dtype=np.float32)]

    # # Pre-pad resize scale used by LongestMaxSize
    # max_side = float(max(original_height, original_width))
    # if max_side == 0:
    #     return [np.eye(3, dtype=np.float32)]
    # scale = 640.0 / max_side
    # inv_scale = 1.0 / scale

    # matrix = np.array(
    #     [[inv_scale, 0.0, 0.0], [0.0, inv_scale, 0.0], [0.0, 0.0, 1.0]],
    #     dtype=np.float32,
    # )
    # return [matrix]
    # Reverting the bug fix above due to low performance
    # 1. Calculate the scale and the intermediate (pre-pad) dimensions
    original_height, original_width = original_shape[:2]
    scale = 640.0 / max(original_height, original_width)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))

    # 2. Calculate the padding that Albumentations *would have added*
    pad_left = (640 - resized_width) // 2
    pad_top = (640 - resized_height) // 2

    # 3. Create the inverse transformation matrix
    # This matrix first translates to remove the padding, then scales up.
    inv_scale = 1.0 / scale
    matrix = np.array(
        [
            [inv_scale, 0.0, -pad_left * inv_scale],
            [0.0, inv_scale, -pad_top * inv_scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

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


# BugRef: BUG-2025-001 — fallback scales by original/resized (pre-pad) dims
# Report: docs/bug_reports/BUG-2025-001_inference_padding_scaling_mismatch.md
# Date: 2025-10-20
# IndexSig: scale_x=W0/W1; scale_y=H0/H1; W1,H1=round(scale*[W0,H0])
def fallback_postprocess(predictions: Any, original_shape: Sequence[int], settings: PostprocessSettings) -> dict[str, Any]:
    prob_map = predictions.get("prob_maps")
    if prob_map is None:
        raise ValueError("'prob_maps' key not found in model predictions.")

    if torch is not None and isinstance(prob_map, torch.Tensor):
        prob_map = prob_map.detach().cpu().numpy()

    prob_map = np.squeeze(prob_map)
    binary_map = (prob_map > settings.binarization_thresh).astype(np.uint8)

    original_height, original_width, _ = original_shape
    # Compute pre-pad resized dims (W1, H1) consistent with LongestMaxSize(640)
    max_side = float(max(original_height, original_width))
    if max_side == 0:
        raise ValueError("Invalid original image size")
    scale = 640.0 / max_side
    resized_height = int(round(original_height * scale))
    resized_width = int(round(original_width * scale))
    if resized_height <= 0 or resized_width <= 0:
        raise ValueError(f"Invalid resized dims: {resized_width}x{resized_height}")

    # binary_map is sized like the processed map (after model head). We assume it matches 640x640 padded canvas.
    # With top-left padding, unpadding is no-op for coordinates (content starts at 0,0), so we scale by original/pre-pad dims.
    scale_x = original_width / float(resized_width)
    scale_y = original_height / float(resized_height)

    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons: list[str] = []
    texts: list[str] = []
    confidences: list[float] = []

    for index, contour in enumerate(contours):
        if len(polygons) >= settings.max_candidates:
            LOGGER.info("Reached maximum candidates limit: %s", settings.max_candidates)
            break

        x, y, w, h = cv2.boundingRect(contour)
        if w < settings.min_detection_size or h < settings.min_detection_size:
            continue

        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)
        orig_w = int(w * scale_x)
        orig_h = int(h * scale_y)

        polygon_coords = [
            orig_x,
            orig_y,
            orig_x + orig_w,
            orig_y,
            orig_x + orig_w,
            orig_y + orig_h,
            orig_x,
            orig_y + orig_h,
        ]
        # Competition format uses space-separated coordinates
        polygons.append(" ".join(map(str, polygon_coords)))
        texts.append(f"Text_{index + 1}")

        prob_slice = prob_map[y : y + h, x : x + w]
        confidence = float(prob_slice.mean()) if prob_slice.size else 0.0
        confidences.append(confidence)

    return {
        "polygons": "|".join(polygons),
        "texts": texts,
        "confidences": confidences,
    }
