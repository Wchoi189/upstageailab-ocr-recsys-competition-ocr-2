# ocr/domains/detection/callbacks/wandb.py
# Detection domain-specific WandB image logging
# Extracted from ocr/core/utils/wandb_utils.py during Phase 2 surgical refactor

from __future__ import annotations

import hashlib
import os

from ocr.core.utils.wandb_base import _crop_to_content, _to_u8_bgr


def log_validation_images(images, gt_bboxes, pred_bboxes, epoch, limit=8, seed: int = 42, filenames=None):
    """Logs images with ground truth (green) and predicted (red) boxes to W&B.

    Adds a compact legend overlay and samples up to `limit` images with a fixed seed
    for diversity across epochs.

    This is Detection domain-specific: handles bounding boxes and polygons.
    """
    # Lazy imports to reduce module coupling
    import cv2
    import numpy as np
    import wandb

    from ocr.core.utils.text_rendering import put_text_utf8

    if not wandb.run:
        return

    log_images = []
    drawn_images = []
    sizes = []
    captions = []
    # Rank images to prefer those with BOTH GT and PRED present; then fall back
    N = min(len(images), len(gt_bboxes), len(pred_bboxes))
    pairs = []
    for i in range(N):
        g = len(gt_bboxes[i]) if gt_bboxes[i] is not None else 0
        p = len(pred_bboxes[i]) if pred_bboxes[i] is not None else 0
        pairs.append((i, g, p))

    # Deterministic shuffle per epoch
    try:
        epoch_int = int(epoch)
        local_seed = seed + max(0, epoch_int)
    except Exception:
        local_seed = seed
    rng = np.random.RandomState(local_seed)
    rng.shuffle(pairs)

    both = [i for (i, g, p) in pairs if g > 0 and p > 0]
    only_gt = [i for (i, g, p) in pairs if g > 0 and p == 0]
    only_pred = [i for (i, g, p) in pairs if g == 0 and p > 0]

    ordered = both + only_gt + only_pred
    idxs = ordered[: min(limit, len(ordered))]

    # For optional integrity table
    table_rows = []
    for rank, i in enumerate(idxs):
        image, gt_boxes, pred_boxes = images[i], gt_bboxes[i], pred_bboxes[i]

        # Convert to BGR uint8 for OpenCV drawing
        img_to_draw, needs_rgb_conversion = _to_u8_bgr(image)

        # Prepare counts and safe iterables
        g = len(gt_boxes) if gt_boxes is not None else 0
        p = len(pred_boxes) if pred_boxes is not None else 0
        gt_iter = gt_boxes if gt_boxes is not None else []
        pred_iter = pred_boxes if pred_boxes is not None else []

        # Draw ground truth boxes (in green)
        for box in gt_iter:
            box_array = np.array(box).reshape(-1, 2).astype(np.int32)
            # Handle polygons with different numbers of points
            if len(box_array) >= 3:  # Need at least 3 points for a polygon
                cv2.polylines(
                    img_to_draw,
                    [box_array],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

        # Draw predicted boxes (in red)
        # BUG-20251116-001: In BGR format, red is (0, 0, 255), not (255, 0, 0)
        for box in pred_iter:
            box_array = np.array(box).reshape(-1, 2).astype(np.int32)
            # Handle polygons with different numbers of points
            if len(box_array) >= 3:  # Need at least 3 points for a polygon
                cv2.polylines(
                    img_to_draw,
                    [box_array],
                    isClosed=True,
                    color=(0, 0, 255),  # BGR: Red = (0, 0, 255)
                    thickness=2,
                )

        # Add small legend (top-left)
        legend_h = 36
        legend_w = 160
        overlay = img_to_draw.copy()
        cv2.rectangle(overlay, (0, 0), (legend_w, legend_h), (0, 0, 0), thickness=-1)
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, img_to_draw, 1 - alpha, 0, dst=img_to_draw)
        # Green = GT, Red = Pred
        cv2.line(img_to_draw, (8, 12), (32, 12), (0, 255, 0), 3)
        img_to_draw = put_text_utf8(
            img_to_draw,
            "GT",
            (38, 10),
            font_size=12,
            color=(220, 220, 220),
        )
        cv2.line(img_to_draw, (8, 26), (32, 26), (0, 0, 255), 3)  # BGR: Red = (0, 0, 255)
        img_to_draw = put_text_utf8(
            img_to_draw,
            "Pred",
            (38, 24),
            font_size=12,
            color=(220, 220, 220),
        )

        # BUG-20251116-001: Convert BGR back to RGB for WandB logging (WandB expects RGB)
        if needs_rgb_conversion and img_to_draw.ndim == 3 and img_to_draw.shape[2] == 3:
            img_to_draw = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)

        # Ensure image is uint8 and in [0,255] range
        img_uint8 = np.clip(img_to_draw, 0, 255).astype(np.uint8)
        cropped = _crop_to_content(img_uint8)
        drawn_images.append(cropped)
        sizes.append(cropped.shape[:2])  # (H, W)
        # Filename & original dims if provided.
        # NOTE: Use original sampled index `i` (not the display ordering `rank`).
        # Previous implementation used `rank`, which reorders images by
        # (GT/PRED presence) causing filename/image mismatches in W&B captions.
        fname = "(unknown)"
        orig_w = -1
        orig_h = -1
        if filenames and i < len(filenames):
            fname, orig_w, orig_h = filenames[i]
            meta_prefix = f"{fname} ({orig_w}x{orig_h})"
        else:
            meta_prefix = "(unknown)"
        caption = f"{meta_prefix} | Ep {epoch} | GT={g} Pred={p}"
        captions.append(caption)
        if os.environ.get("LOG_VAL_IMAGE_TABLE", "0") == "1":
            # Lightweight perceptual fingerprint: SHA1 of raw bytes
            img_hash = hashlib.sha1(cropped.tobytes()).hexdigest()[:12]
            table_rows.append(
                [
                    rank,  # display order
                    i,  # original sample index
                    fname if (filenames and i < len(filenames)) else "(unknown)",
                    orig_w if (filenames and i < len(filenames)) else -1,
                    orig_h if (filenames and i < len(filenames)) else -1,
                    g,
                    p,
                    cropped.shape[1],
                    cropped.shape[0],  # logged W,H
                    img_hash,
                    caption,
                ]
            )

    # Pad images to the same size to avoid W&B UI warnings
    if drawn_images:
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)
        for idx, img in enumerate(drawn_images):
            h, w = img.shape[:2]
            pad_bottom = max_h - h
            pad_right = max_w - w
            if pad_bottom > 0 or pad_right > 0:
                img_padded = cv2.copyMakeBorder(
                    img,
                    0,
                    pad_bottom,
                    0,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                )
            else:
                img_padded = img
            log_images.append(wandb.Image(img_padded, caption=captions[idx]))

    if log_images:
        payload = {"validation_images": log_images}
        if table_rows:
            table = wandb.Table(
                columns=[  # type: ignore[arg-type]
                    "display_rank",
                    "orig_index",
                    "filename",
                    "orig_w",
                    "orig_h",
                    "gt_count",
                    "pred_count",
                    "logged_w",
                    "logged_h",
                    "sha1_12",
                    "caption",
                ],
                data=table_rows,
            )
            payload["validation_image_table"] = table  # type: ignore[assignment]
        wandb.log(payload)
