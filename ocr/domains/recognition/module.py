"""Recognition-specific PyTorch Lightning Module."""

import torch
from pydantic import ValidationError

from ocr.core.lightning.base import OCRPLModule
from ocr.core.validation import ValidatedTensorData


class RecognitionPLModule(OCRPLModule):
    """Recognition-specific Lightning Module for text recognition tasks.

    Implements recognition-specific validation logic:
    - Text decoding from logits
    - Character Error Rate (CER) computation
    - Exact match accuracy
    - WandB recognition image logging

    Attributes:
        rec_cer: CharErrorRate metric for validation
    """

    def __init__(self, model, dataset, config, metric_cfg=None):
        super().__init__(model, dataset, config, metric_cfg)

        # Recognition-specific initialization
        # Lazy import - defers ~25s torchmetrics loading until training starts
        from torchmetrics.text import CharErrorRate
        self.rec_cer = CharErrorRate()

    def training_step(self, batch, batch_idx):
        """Recognition-specific training step with optional tensor validation."""
        pred = self.model(**batch)

        # Validate model outputs only in debug mode (BUG-20251112-001/013 prevention)
        # NOTE: Pydantic validation causes GPU sync - disabled by default for performance
        if getattr(getattr(self.config, "global", None), "debug", False):
            try:
                ValidatedTensorData(tensor=pred["loss"], expected_device=batch["images"].device, allow_nan=False, allow_inf=False)
            except ValidationError as exc:
                raise ValueError(f"Training step model output validation failed at step {batch_idx}: {exc}") from exc

        self.log("train/loss", pred["loss"], batch_size=batch["images"].shape[0])
        for key, value in pred["loss_dict"].items():
            self.log(f"train/{key}", value, batch_size=batch["images"].shape[0])
        return pred["loss"]

    def validation_step(self, batch, batch_idx):
        """Recognition-specific validation step.

        Decodes predicted tokens to text and computes character-level metrics.
        """
        pred = self.model(**batch)

        # Validate model outputs only in debug mode
        if getattr(getattr(self.config, "global", None), "debug", False):
            try:
                ValidatedTensorData(
                    tensor=pred["loss"], expected_device=batch["images"].device, allow_nan=False, allow_inf=False
                )
            except ValidationError as exc:
                raise ValueError(f"Validation step model output validation failed at step {batch_idx}: {exc}") from exc

        self.log("val_loss", pred["loss"], batch_size=batch["images"].shape[0])
        for key, value in pred["loss_dict"].items():
            self.log(f"val_{key}", value, batch_size=batch["images"].shape[0])

        # Run inference for metrics
        with torch.no_grad():
            inference_out = self.model(images=batch["images"], return_loss=False)

        if "tokens" in inference_out:
            tokenizer = self._get_tokenizer()
            if tokenizer:
                pred_texts = self._decode_predictions(inference_out, tokenizer)
                gt_texts = self._decode_ground_truth(batch, tokenizer)

                if gt_texts:
                    self._compute_metrics(pred_texts, gt_texts)
                    self._log_validation_images(batch, pred_texts, gt_texts, batch_idx)

        return pred["loss"]

    def on_validation_epoch_end(self):
        """Recognition has no epoch-level evaluator - metrics are already logged per-step."""
        # CER metric is automatically aggregated by TorchMetrics
        pass

    def _decode_predictions(self, inference_out, tokenizer):
        """Decode predicted tokens to text."""
        pred_tokens = inference_out["tokens"]
        if isinstance(pred_tokens, torch.Tensor):
            pred_tokens = pred_tokens.tolist()
        return tokenizer.batch_decode(pred_tokens)

    def _decode_ground_truth(self, batch, tokenizer):
        """Decode ground truth tokens or labels."""
        if "text_tokens" in batch:
            gt_tokens = batch["text_tokens"]
            if isinstance(gt_tokens, torch.Tensor):
                gt_tokens = gt_tokens.tolist()
            return tokenizer.batch_decode(gt_tokens)
        elif "label" in batch:
            return batch["label"]
        return None
    def _get_tokenizer(self):
        """Retrieve tokenizer from validation dataset."""
        if "val" in self.dataset and hasattr(self.dataset["val"], "tokenizer"):
            return self.dataset["val"].tokenizer
        return None

    def _compute_metrics(self, pred_texts, gt_texts):
        """Compute and log recognition metrics."""
        self.rec_cer(pred_texts, gt_texts)

        # Exact Match Accuracy
        matches = sum([1 for p, g in zip(pred_texts, gt_texts, strict=True) if p == g])
        batch_acc = matches / len(pred_texts) if len(pred_texts) > 0 else 0.0

        self.log("val/acc", batch_acc, batch_size=len(pred_texts), prog_bar=True)
        self.log("val/cer", self.rec_cer, batch_size=len(pred_texts), prog_bar=True)

        # Debug logging
        if self.trainer.is_global_zero:
             print(f"\\n[DEBUG] Step {self.trainer.global_step} Predictions:")
             for i in range(min(3, len(pred_texts))):
                 tokenizer = self._get_tokenizer()
                 if tokenizer:
                     gt_tokens = tokenizer.encode(gt_texts[i])
                     print(f"  GT Text:   '{gt_texts[i]}'")
                     print(f"  GT Tokens: {gt_tokens}")
                 print(f"  Pred Text: '{pred_texts[i]}'")

    def _log_validation_images(self, batch, pred_texts, gt_texts, batch_idx):
        """Log validation images to WandB if enabled."""
        if batch_idx >= 2:
            return

        use_wandb = False
        try:
            if hasattr(self.config, "train") and hasattr(self.config.train, "logger"):
                if "wandb" in self.config.train.logger:
                    use_wandb = self.config.train.logger.wandb.get("enabled", False)
        except Exception:
            pass

        if use_wandb:
            from ocr.domains.recognition.callbacks.wandb_logging import log_recognition_images

            log_recognition_images(
                images=batch["images"],
                pred_texts=pred_texts,
                gt_texts=gt_texts,
                epoch=self.current_epoch,
                limit=8,
                seed=42,
                filenames=batch.get("image_filename", None),
                caption_prefix="val_recognition_samples",
            )
