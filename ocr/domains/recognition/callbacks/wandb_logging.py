# ocr/domains/recognition/callbacks/wandb_logging.py
# Recognition domain-specific WandB image logging (function-based)
# Extracted from ocr/core/utils/wandb_utils.py during Phase 2 surgical refactor

from __future__ import annotations

import logging
import os
from pathlib import Path

from ocr.core.lightning.processors.image_processor import ImageProcessor
from ocr.core.utils.wandb_base import _to_u8_bgr


logger = logging.getLogger(__name__)


def _font_supports_korean(font_path: Path) -> bool:
    try:
        from fontTools.ttLib import TTFont
    except Exception:
        return True

    try:
        font = TTFont(str(font_path), lazy=True)
        cmap = {}
        for table in font["cmap"].tables:
            cmap.update(table.cmap)
    except Exception:
        return False

    required_chars = ["한", "변", "김"]
    return all(ord(ch) in cmap for ch in required_chars)


def _load_korean_font(font_size: int):
    from PIL import ImageFont

    env_font = os.getenv("OCR_WANDB_FONT_PATH")
    candidates = [
        env_font,
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists() or not path.is_file():
            continue
        if not _font_supports_korean(path):
            continue

        return ImageFont.truetype(str(path), font_size)

    raise RuntimeError(
        "No Korean-capable font found for WandB recognition image logging. "
        "Install a Korean font (e.g., fonts-nanum or Noto Sans CJK) and/or set OCR_WANDB_FONT_PATH "
        "to a valid .ttf/.otf/.ttc file with Hangul glyphs."
    )


def log_recognition_images(
    images,
    pred_texts: list[str],
    gt_texts: list[str] | None,
    epoch: int,
    limit: int = 8,
    seed: int = 42,
    filenames: list[str] | None = None,
    caption_prefix: str = "val_rec_samples",
    max_image_side: int | None = 640,
    patch_native_view: bool = False,
    wandb_run=None,
) -> None:
    """Logs recognition images with ground truth (Green) and predicted (Red/Blue) text.

    [VERIFY-BUG-20260115-001] Added to visualize recognition outputs during debug.
    Uses PIL for text rendering to support Korean characters (Mojibake fix).

    This is Recognition domain-specific: handles text rendering with Korean font support.

    Args:
        images: Batch of images (Tensor, Numpy, or List of PIL)
        pred_texts: List of predicted text strings
        gt_texts: List of ground truth text strings (optional)
        epoch: Current epoch
        limit: Max images to log
        seed: Random seed for sampling
        filenames: Optional filenames
        caption_prefix: Prefix for WandB caption
    """
    import cv2
    import numpy as np
    import wandb
    from PIL import Image, ImageDraw

    if gt_texts is None:
        logger.warning("Skipping WandB logging: gt_texts is missing.")
        return

    if len(pred_texts) != len(gt_texts):
        logger.warning(
            "Skipping WandB logging: pred_texts (%s) and gt_texts (%s) length mismatch.",
            len(pred_texts),
            len(gt_texts),
        )
        return

    num_samples = len(images)
    if num_samples == 0:
        return

    # Sample indices
    indices = list(range(num_samples))
    if num_samples > limit:
        rng = np.random.default_rng(seed + epoch)
        indices = sorted(rng.choice(indices, size=limit, replace=False))

    wandb_images = []

    font = None
    if not patch_native_view:
        font = _load_korean_font(24)

    for idx in indices:
        # Convert image to u8 BGR. _to_u8_bgr handles tensors/PIL/numpy
        img_bgr, _ = _to_u8_bgr(images[idx])
        # Convert to RGB for PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Prepare text info
        pred_text = pred_texts[idx] if idx < len(pred_texts) else ""
        gt_text = gt_texts[idx] if idx < len(gt_texts) else ""
        filename = filenames[idx] if filenames and idx < len(filenames) else f"sample_{idx}"

        if patch_native_view:
            output_image = pil_img
        else:
            # Create canvas with extended bottom
            w, h = pil_img.size
            extension = 80
            new_h = h + extension
            canvas = Image.new("RGB", (w, new_h), (255, 255, 255))
            canvas.paste(pil_img, (0, 0))

            draw = ImageDraw.Draw(canvas)

            # Draw GT (Green)
            draw.text((10, h + 10), f"GT: {gt_text}", font=font, fill=(0, 200, 0))

            # Draw Pred (Red if mismatch, Blue if match)
            color = (255, 0, 0) if pred_text != gt_text else (0, 0, 255)
            draw.text((10, h + 45), f"Pr: {pred_text}", font=font, fill=color)
            output_image = canvas

        wandb_images.append(
            wandb.Image(
                ImageProcessor.prepare_wandb_image(output_image, max_image_side),
                caption=f"Epoch {epoch} | {filename} | GT: {gt_text} | Pr: {pred_text}"
            )
        )

    # Log to WandB (strict mode: explicit run required)
    if wandb_run is None:
        raise RuntimeError("wandb_run must be provided for recognition image logging.")

    if wandb_images:
        payload = {caption_prefix: wandb_images, "epoch": epoch}
        wandb_run.log(payload)
