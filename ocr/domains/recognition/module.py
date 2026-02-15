"""Recognition-specific PyTorch Lightning Module."""

import logging

import torch
from pydantic import ValidationError

from ocr.core.lightning.base import OCRPLModule
from ocr.core.data.schemas import CacheConfig, ImageLoadingConfig
from ocr.core.utils.config_utils import is_config
from ocr.domains.recognition.models.flash_attention import enable_flash_attention_kernel


logger = logging.getLogger(__name__)


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

    def _plm_enabled(self) -> bool:
        decoder = getattr(self.model, "decoder", None)
        return bool(getattr(decoder, "plm", None))

    def training_step(self, batch, batch_idx):
        """Recognition-specific training step with optional tensor validation."""
        with enable_flash_attention_kernel(plm_enabled=self._plm_enabled()):
            pred = self.model(**batch)

        # Validate model outputs only in debug mode (BUG-20251112-001/013 prevention)
        # NOTE: Pydantic validation causes GPU sync - disabled by default for performance
        if self.config.get("global", {}).get("debug", False):
            try:
                ValidatedTensorData(tensor=pred["loss"], expected_device=batch["images"].device, allow_nan=False, allow_inf=False)
            except ValidationError as exc:
                raise ValueError(f"Training step model output validation failed at step {batch_idx}: {exc}") from exc

        self.log("train/loss", pred["loss"], batch_size=batch["images"].shape[0])
        for key, value in pred["loss_dict"].items():
            self.log(f"train/{key}", value, batch_size=batch["images"].shape[0])
        return pred["loss"]

    def on_after_backward(self):
        """Log gradient norms for debugging."""
        pass
        # if self.global_step % 10 == 0:
        #     if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "pos_encoder"):
        #         grad = self.model.decoder.pos_encoder.grad
        #         if grad is not None:
        #             print(f"\n[Grad Debug] Step {self.global_step} - Pos Encoder Grad Norm: {grad.norm():.4f}")
        #         else:
        #             print(f"\n[Grad Debug] Step {self.global_step} - Pos Encoder Grad is None!")
        #
        #     if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "embed_tokens"):
        #         grad = self.model.decoder.embed_tokens.weight.grad
        #         if grad is not None:
        #              print(f"[Grad Debug] Step {self.global_step} - Embed Tokens Grad Norm: {grad.norm():.4f}")


    def validation_step(self, batch, batch_idx):
        """Recognition-specific validation step.

        Decodes predicted tokens to text and computes character-level metrics.
        """
        with enable_flash_attention_kernel(plm_enabled=self._plm_enabled()):
            pred = self.model(**batch)

        # Validate model outputs only in debug mode
        if self.config.get("global", {}).get("debug", False):
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

                    # DEBUG: Print first few samples
                    if batch_idx == 0:
                        print(f"\n[Validation Debug] Samples:")
                        print(f"  Pred Type: {type(inference_out)}")
                        if is_config(inference_out):
                             print(f"  Pred Keys: {list(inference_out.keys())}")
                             if "tokens" in inference_out:
                                 for i in range(min(5, len(inference_out['tokens']))):
                                     print(f"  Pred IDs: {inference_out['tokens'][i].tolist()}")
                             else:
                                 print("  WARNING: 'tokens' key missing in inference_out dict!")
                        else:
                             print(f"  WARNING: inference_out is not a dict!")

                        for i in range(min(5, len(pred_texts))):
                             print(f"  Pred: '{pred_texts[i]}' | GT: '{gt_texts[i]}'")
                        print(f"  Match Count: {sum([1 for p, g in zip(pred_texts, gt_texts) if p == g])}/{len(pred_texts)}")

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

    def _log_validation_images(self, batch, pred_texts, gt_texts, batch_idx):
        """Log validation images to WandB if enabled."""
        if batch_idx >= 2:
            return

        if not pred_texts or not gt_texts:
            logger.warning("Skipping WandB logging: missing pred_texts or gt_texts.")
            return

        if len(pred_texts) != len(gt_texts):
            logger.warning(
                "Skipping WandB logging: pred_texts (%s) and gt_texts (%s) length mismatch.",
                len(pred_texts),
                len(gt_texts),
            )
            return

        if self._wandb_enabled() and self._wandb_image_logging_enabled():
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

    def _wandb_image_logging_enabled(self) -> bool:
        wandb_cfg = self._get_wandb_cfg()
        if is_config(wandb_cfg):
            return wandb_cfg.get("log_recognition_images", False)
        return False
