# ocr/domains/recognition/callbacks/wandb_logging.py
# Recognition domain-specific WandB image logging (function-based)
# Extracted from ocr/core/utils/wandb_utils.py during Phase 2 surgical refactor

from __future__ import annotations

from ocr.core.utils.wandb_base import _to_u8_bgr


def log_recognition_images(
    images,
    pred_texts: list[str],
    gt_texts: list[str] | None,
    epoch: int,
    limit: int = 8,
    seed: int = 42,
    filenames: list[str] | None = None,
    caption_prefix: str = "val_rec_samples",
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
    from PIL import Image, ImageDraw, ImageFont

    # Ensure list types
    if gt_texts is None:
        gt_texts = [""] * len(pred_texts)

    num_samples = len(images)
    if num_samples == 0:
        return

    # Sample indices
    indices = list(range(num_samples))
    if num_samples > limit:
        rng = np.random.default_rng(seed + epoch)
        indices = sorted(rng.choice(indices, size=limit, replace=False))

    wandb_images = []

    # Font handling
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    try:
        font = ImageFont.truetype(font_path, 24)
    except OSError:
        print(f"Warning: Korean font not found at {font_path}. Using default.")
        font = ImageFont.load_default()

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

        wandb_images.append(
            wandb.Image(
                canvas,
                caption=f"Epoch {epoch} | {filename} | GT: {gt_text} | Pr: {pred_text}"
            )
        )

    # Log to WandB
    if wandb_images:
        wandb.log({caption_prefix: wandb_images, "epoch": epoch})
