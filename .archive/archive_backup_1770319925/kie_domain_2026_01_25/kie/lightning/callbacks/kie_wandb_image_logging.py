import logging
from typing import Any

import lightning.pytorch as pl
import torch
from PIL import Image, ImageDraw

import wandb

logger = logging.getLogger(__name__)


class WandBKeyInformationExtractionImageLogger(pl.Callback):
    """
    Callback to log KIE predictions to WandB during validation.
    Annotates images with Ground Truth (Green) and Predicted (Red) bounding boxes.
    """

    def __init__(self, num_samples: int = 4, label_list: list[str] | None = None):
        super().__init__()
        self.num_samples = num_samples
        self.label_list = label_list
        self.log_next_batch = False

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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
            images_to_log = []

            # Unpack batch
            # We need raw images or paths.
            # The KIEDataset returns 'image_path'. We can reload from there for best quality.
            image_paths = batch.get("image_path", [])
            bboxes = batch.get("bbox")
            labels = batch.get("labels")

            # Get Predictions
            # outputs is loss if dict, or whatever validation_step returns if aggregated?
            # In validation_step, we usually just return loss. We need to run inference here again or grab logits?
            # Running inference is cleaner for visualization.

            with torch.no_grad():
                model_out = pl_module(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    bbox=batch["bbox"],
                    pixel_values=batch.get("pixel_values"),
                )
                logits = model_out["logits"]
                predictions = torch.argmax(logits, dim=2)

            # Limit samples
            n = min(len(image_paths), self.num_samples)

            # Storage for intermediate images
            display_data = []  # List of (image, caption)

            for i in range(n):
                img_path = image_paths[i]

                try:
                    image = Image.open(img_path).convert("RGB")
                    draw = ImageDraw.Draw(image)
                except Exception as e:
                    logger.warning(f"Could not load image for logging: {img_path} - {e}")
                    continue

                width, height = image.size

                # Iterate tokens
                # Boxes are normalized 0-1000
                batch_boxes = bboxes[i]
                batch_labels = labels[i]
                batch_preds = predictions[i]

                for box, label, pred in zip(batch_boxes, batch_labels, batch_preds, strict=False):
                    # Skip padding
                    if label == -100:
                        continue

                    # Unnormalize box
                    # box is [x1, y1, x2, y2] in 0-1000 scale
                    x1 = int(box[0].item() / 1000 * width)
                    y1 = int(box[1].item() / 1000 * height)
                    x2 = int(box[2].item() / 1000 * width)
                    y2 = int(box[3].item() / 1000 * height)

                    label_name = self.label_list[label.item()] if self.label_list else str(label.item())
                    pred_name = self.label_list[pred.item()] if self.label_list else str(pred.item())

                    # Draw GT (Green) if not 'O' or background
                    if label_name != "O":
                        draw.rectangle((x1, y1, x2, y2), outline="green", width=2)

                    # Draw Pred (Red) if mismatch or important
                    if pred_name != "O":
                        color = "red" if pred_name != label_name else "blue"  # Blue if correct, Red if wrong
                        # Offset slightly to see both
                        draw.rectangle((x1 - 2, y1 - 2, x2 + 2, y2 + 2), outline=color, width=2)

                        # Add text
                        text = f"P:{pred_name} (T:{label_name})"
                        draw.text((x1, y1), text, fill=color)

                # Resize image for logging if too large
                max_dim = 768  # Reduced from 1024 as requested
                if width > max_dim or height > max_dim:
                    image.thumbnail((max_dim, max_dim))

                display_data.append((image, f"Val Sample {i}"))

            # Pad images to same size to avoid WandB "Images sizes do not match" warning
            if display_data:
                max_w = max(img.width for img, _ in display_data)
                max_h = max(img.height for img, _ in display_data)

                for img, caption in display_data:
                    # Pad if needed
                    if img.width < max_w or img.height < max_h:
                        new_img = Image.new("RGB", (max_w, max_h), (0, 0, 0))
                        new_img.paste(img, (0, 0))
                        img = new_img

                    # Log as JPEG using standard file_type support
                    images_to_log.append(wandb.Image(img, caption=caption, file_type="jpg"))  # type: ignore[attr-defined]

            if images_to_log:
                trainer.logger.experiment.log({"validation/predictions": images_to_log})

        except Exception as e:
            logger.error(f"Failed to log images to WandB: {e}", exc_info=True)
