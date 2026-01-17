from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

import lightning.pytorch as pl
import numpy as np
import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RecognitionWandbImageLogger(pl.Callback):
    """
    Callback to log Recognition predictions to WandB during validation.
    Annotates images with Ground Truth (Green) and Predicted (Red) text.
    """

    def __init__(self, num_samples: int = 8, log_every_n_epochs: int = 1):
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_epochs = log_every_n_epochs
        self.log_next_batch = False

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self.log_next_batch = True

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.log_next_batch:
            return

        # Only look at the first batch effectively
        self.log_next_batch = False

        if not isinstance(trainer.logger, pl.loggers.WandbLogger):
            return

        try:
            self._log_images(trainer, pl_module, batch, outputs)
        except Exception as e:
            logger.error(f"Failed to log images to WandB: {e}", exc_info=True)

    def _log_images(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        outputs: Any,
    ) -> None:
        # Lazy imports to reduce module load time
        import wandb
        from PIL import Image

        try:
            from ocr.core.utils.text_rendering import put_text_utf8
        except ImportError:
            # Fallback if util is missing (though it should exist)
            logger.warning("ocr.core.utils.text_rendering not found, falling back to basic PIL drawing")
            put_text_utf8 = None  # type: ignore


        images_to_log = []

        # STRICT DATA CONTRACT: Expecting a dictionary with 'images' key
        if not isinstance(batch, dict):
            logger.warning(
                f"Skipping image logging: Expected batch to be 'dict', got '{type(batch).__name__}'. "
                "Check data loader contract."
            )
            return

        images = batch.get("images")  # Tensor [B, C, H, W]
        targets = batch.get("targets")  # Tensor or List
        filenames = batch.get("filenames", [])

        if images is None:
            logger.warning("Skipping image logging: 'images' key missing in batch.")
            return

        if not isinstance(images, torch.Tensor):
            logger.warning(f"Skipping image logging: Expected 'images' to be Tensor, got '{type(images).__name__}'.")
            return

        # Get Predictions
        preds = None
        if isinstance(outputs, dict):
            if "preds" in outputs:
                preds = outputs["preds"]
            elif "logits" in outputs:
                logits = outputs["logits"]
                if hasattr(pl_module, "tokenizer"):
                    try:
                        preds = pl_module.tokenizer.decode(logits)
                    except Exception as e:
                        logger.warning(f"Failed to decode logits: {e}")

        # Fallback inference if predictions unavailable
        if preds is None:
            with torch.no_grad():
                try:
                    if hasattr(pl_module, "get_predictions"):
                        preds = pl_module.get_predictions(batch)
                    elif hasattr(pl_module, "tokenizer"):
                        # Basic forward pass assumption
                        model_out = pl_module(images)
                        preds = pl_module.tokenizer.decode(model_out)
                except Exception:
                    pass  # Fail silently on inference attempt

        # Determine indices
        batch_size = images.shape[0]
        n = min(batch_size, self.num_samples)

        # Deterministic sampling
        local_seed = 42 + trainer.current_epoch
        indices = list(range(batch_size))
        random.Random(local_seed).shuffle(indices)
        selected_indices = indices[:n]

        # Decode Ground Truth
        gt_texts = []
        if targets is not None:
            if hasattr(pl_module, "tokenizer"):
                try:
                    gt_texts = pl_module.tokenizer.decode(targets)
                    # Handle single string return case
                    if isinstance(gt_texts, str):
                        gt_texts = [gt_texts] * batch_size
                except Exception:
                     # Fallback if decode fails or targets are already strings
                    if isinstance(targets, (list, tuple)) and len(targets) > 0 and isinstance(targets[0], str):
                        gt_texts = list(targets)
                    else:
                        gt_texts = ["?"] * batch_size
            elif isinstance(targets, (list, tuple)) and len(targets) > 0 and isinstance(targets[0], str):
                gt_texts = list(targets)
            else:
                gt_texts = ["?"] * batch_size
        else:
            gt_texts = ["?"] * batch_size

        if preds is None:
             preds = ["(no pred)"] * batch_size
        elif isinstance(preds, str): # Handle single string return from decode
             preds = [preds] * batch_size

        display_data = []

        for idx in selected_indices:
            # 1. Prepare Image (Tensor -> Numpy [0, 255] uint8)
            img_tensor = images[idx]
            if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]:
                 # CHW -> HWC
                 img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            else:
                 logger.warning(f"Unexpected image shape: {img_tensor.shape}")
                 continue

            # Denormalize
            # Assumption: Input is either [0, 1] or [-1, 1]
            if img_np.min() < 0:
                img_np = (img_np * 0.5 + 0.5) * 255.0
            else:
                img_np = img_np * 255.0

            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            # Handle grayscale (1 channel) -> RGB (3 channels) for display
            if img_np.shape[2] == 1:
                img_np = np.concatenate([img_np] * 3, axis=2)

            # 2. Resize for visibility (Recognition crops are short)
            h, w, _ = img_np.shape
            target_h = 64
            if h < target_h:
                scale = target_h / h
                new_w = int(w * scale)
                # Use PIL for high-quality resize
                pil_img = Image.fromarray(img_np)
                pil_img = pil_img.resize((new_w, target_h), Image.NEAREST)
                img_np = np.array(pil_img)

            # 3. Create Canvas and Draw Text
            # We use a canvas to draw text *below* the image to avoid occlusion
            h, w, _ = img_np.shape
            canvas_h = h + 60 # Space for 2 lines
            canvas_w = max(w, 400) # Minimum width for text

            # Create dark background
            canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8) # Dark gray RGB(20,20,20)

            # Place image centered
            x_offset = (canvas_w - w) // 2
            canvas[0:h, x_offset:x_offset+w] = img_np # BGR or RGB?
            # Note: WandB expects RGB. OpenCV is BGR.
            # If we used PIL.fromarray(img_np) earlier, img_np was RGB.
            # put_text_utf8 expects BGR by default convention of OpenCV,
            # BUT our img_np came from PyTorch (RGB) -> Permute -> RGB.
            # So img_np is RGB.
            # put_text_utf8 converts input to RGB internally, draws, then converts back to BGR.
            # So if we pass RGB, it treats it as BGR (swaps channels to RGB), draws, swaps back to BGR.
            # This double swap cancels out if we are consistently "wrong".
            # Let's be explicit: Convert to BGR for drawing function, then back to RGB for WandB.

            import cv2
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

            gt_text = gt_texts[idx] if idx < len(gt_texts) else "?"
            pred_text = preds[idx] if idx < len(preds) else "?"

            gt_color = (0, 255, 0) # Green
            pred_color = (255, 100, 100) if pred_text != gt_text else (100, 100, 255) # Red if wrong, Blue if right

            if put_text_utf8:
                canvas_bgr = put_text_utf8(canvas_bgr, f"GT:   {gt_text}", (10, h + 20), color=gt_color, font_size=16)
                canvas_bgr = put_text_utf8(canvas_bgr, f"Pred: {pred_text}", (10, h + 45), color=pred_color, font_size=16)
            else:
                # Fallback to cv2
                cv2.putText(canvas_bgr, f"GT: {gt_text}", (10, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1)
                cv2.putText(canvas_bgr, f"Pred: {pred_text}", (10, h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)

            # Convert back to RGB for WandB
            canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

            caption = f"Val Sample {idx}"
            if idx < len(filenames):
                caption += f" | {filenames[idx]}"

            # Lazy import Image again just to be safe or use existing
            display_data.append(wandb.Image(Image.fromarray(canvas_rgb), caption=caption))

        if display_data and trainer.logger:
             trainer.logger.experiment.log(
                 {"validation/recognition_samples": display_data},
                 step=trainer.global_step
             )
