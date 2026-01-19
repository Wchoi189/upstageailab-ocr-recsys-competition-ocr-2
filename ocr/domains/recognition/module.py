"""Recognition-specific PyTorch Lightning Module."""

import torch
from pydantic import ValidationError
from torchmetrics.text import CharErrorRate

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

        # Validate model outputs only in debug mode (BUG-20251112-001/013 prevention)
        if getattr(getattr(self.config, "global", None), "debug", False):
            try:
                ValidatedTensorData(tensor=pred["loss"], expected_device=batch["images"].device, allow_nan=False, allow_inf=False)
            except ValidationError as exc:
                raise ValueError(f"Validation step model output validation failed at step {batch_idx}: {exc}") from exc

        self.log("val_loss", pred["loss"], batch_size=batch["images"].shape[0])
        for key, value in pred["loss_dict"].items():
            self.log(f"val_{key}", value, batch_size=batch["images"].shape[0])

        # Run inference for metrics
        with torch.no_grad():
            inference_out = self.model(images=batch["images"], return_loss=False)

        if "tokens" in inference_out:
            tokenizer = None
            if "val" in self.dataset and hasattr(self.dataset["val"], "tokenizer"):
                tokenizer = self.dataset["val"].tokenizer

            if tokenizer:
                # Decode predictions
                pred_tokens = inference_out["tokens"]
                if isinstance(pred_tokens, torch.Tensor):
                    pred_tokens = pred_tokens.tolist()
                pred_texts = tokenizer.batch_decode(pred_tokens)

                # Decode Ground Truth if available
                gt_texts = None
                if "text_tokens" in batch:
                    gt_tokens = batch["text_tokens"]
                    if isinstance(gt_tokens, torch.Tensor):
                        gt_tokens = gt_tokens.tolist()
                    gt_texts = tokenizer.batch_decode(gt_tokens)
                elif "label" in batch:  # Some datasets might pass raw labels
                    gt_texts = batch["label"]

                # Compute and Log Metrics
                if gt_texts:
                    self.rec_cer(pred_texts, gt_texts)
                    # Exact Match Accuracy
                    matches = sum([1 for p, g in zip(pred_texts, gt_texts, strict=True) if p == g])
                    batch_acc = matches / len(pred_texts) if len(pred_texts) > 0 else 0.0

                    self.log("val/acc", batch_acc, batch_size=len(pred_texts), prog_bar=True)
                    self.log("val/cer", self.rec_cer, batch_size=len(pred_texts), prog_bar=True)

                    # Debug logging
                    print(f"\n[DEBUG] Step {self.trainer.global_step} Predictions:")
                    for i in range(min(3, len(pred_texts))):
                        if hasattr(self.dataset["val"], "tokenizer"):
                            gt_tokens = self.dataset["val"].tokenizer.encode(gt_texts[i])
                            print(f"  GT Text:   '{gt_texts[i]}'")
                            print(f"  GT Tokens: {gt_tokens}")
                        print(f"  Pred Text: '{pred_texts[i]}'")

                # WandB image logging (limit to first 2 batches)
                if batch_idx < 2 and self.config.logger.wandb.get("enabled", False):
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

        return pred["loss"]

    def on_validation_epoch_end(self):
        """Recognition has no epoch-level evaluator - metrics are already logged per-step."""
        # CER metric is automatically aggregated by TorchMetrics
        pass
