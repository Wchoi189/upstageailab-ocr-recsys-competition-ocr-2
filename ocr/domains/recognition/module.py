"""Recognition-specific PyTorch Lightning Module."""

import heapq
import logging
import math
from dataclasses import dataclass

import torch
from pydantic import ValidationError

from ocr.core.lightning.base import OCRPLModule
from ocr.core.data.schemas import CacheConfig, ImageLoadingConfig
from ocr.core.utils.config_utils import is_config
from ocr.domains.recognition.models.flash_attention import enable_flash_attention_kernel


logger = logging.getLogger(__name__)


@dataclass
class HighLossSample:
    epoch: int
    global_step: int
    batch_idx: int
    sample_idx: int
    loss: float
    gt_text: str
    pred_text: str
    filename: str | None
    image_ref: torch.Tensor


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
        self._high_loss_epoch_buffer: list[tuple[float, int, HighLossSample]] = []
        self._high_loss_counter = 0

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
                    self._collect_high_loss_samples(
                        batch=batch,
                        pred=pred,
                        inference_out=inference_out,
                        pred_texts=pred_texts,
                        gt_texts=gt_texts,
                        batch_idx=batch_idx,
                    )

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

    def on_validation_epoch_start(self):
        """Reset metrics at start of validation epoch."""
        # Reset CharErrorRate metric to prevent accumulation across epochs
        self.rec_cer.reset()
        
        if self._high_loss_audit_enabled():
            self._clear_high_loss_epoch_buffer()

    def on_validation_epoch_end(self):
        """Recognition has no epoch-level evaluator - metrics are already logged per-step."""
        # CER metric is automatically aggregated by TorchMetrics
        self._log_high_loss_audit()
        
        # Set checkpoint metrics for consistent checkpoint saving
        # Get the aggregated epoch metrics from trainer.callback_metrics
        if hasattr(self, "trainer") and self.trainer:
            self._checkpoint_metrics = {
                "val/acc": float(self.trainer.callback_metrics.get("val/acc", 0.0)),
                "val/cer": float(self.trainer.callback_metrics.get("val/cer", 0.0)),
            }
        
        # Reset metrics at end of validation epoch (ensures clean state for next epoch)
        self.rec_cer.reset()

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

        # Log with on_epoch=True to ensure proper aggregation for checkpointing
        self.log("val/acc", batch_acc, batch_size=len(pred_texts), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/cer", self.rec_cer, batch_size=len(pred_texts), prog_bar=True, on_epoch=True, sync_dist=True)

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

        if self._wandb_image_logging_enabled():
            wandb_run = self._get_wandb_experiment()
            if wandb_run is None:
                raise RuntimeError(
                    "Recognition image logging is enabled, but no active WandB experiment is attached to the trainer. "
                    "Set up a WandbLogger explicitly before enabling train.logger.wandb.log_recognition_images=true."
                )

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
                max_image_side=self._wandb_image_max_side(),
                patch_native_view=self._wandb_image_patch_native_view(),
                wandb_run=wandb_run,
            )

    def _wandb_image_logging_enabled(self) -> bool:
        wandb_cfg = self._get_wandb_cfg()
        if is_config(wandb_cfg):
            return wandb_cfg.get("log_recognition_images", False)
        return False

    def _wandb_image_max_side(self) -> int | None:
        wandb_cfg = self._get_wandb_cfg()
        if not is_config(wandb_cfg):
            return 640

        max_side = wandb_cfg.get("recognition_image_max_side", 640)
        if max_side is None:
            return None

        try:
            max_side = int(max_side)
        except (TypeError, ValueError):
            raise ValueError(
                "train.logger.wandb.recognition_image_max_side must be an integer or null."
            ) from None

        if max_side <= 0:
            return None
        return max_side

    def _wandb_image_patch_native_view(self) -> bool:
        wandb_cfg = self._get_wandb_cfg()
        if not is_config(wandb_cfg):
            return False
        return bool(wandb_cfg.get("recognition_patch_native_view", False))

    def _collect_high_loss_samples(
        self,
        batch,
        pred,
        inference_out,
        pred_texts: list[str],
        gt_texts: list[str],
        batch_idx: int,
    ) -> None:
        if not self._should_collect_high_loss_samples():
            return

        images = batch.get("images")
        if not isinstance(images, torch.Tensor) or images.ndim < 4:
            return

        per_sample_loss = self._compute_per_sample_validation_loss(pred, inference_out, batch)
        if per_sample_loss is None:
            return

        top_k = self._high_loss_audit_top_k()
        include_correct = self._high_loss_audit_include_correct()
        filenames = batch.get("image_filename", None)
        max_len = min(images.shape[0], len(pred_texts), len(gt_texts), len(per_sample_loss))

        for sample_idx in range(max_len):
            loss_val = float(per_sample_loss[sample_idx].item())
            if not math.isfinite(loss_val):
                continue

            pred_text = pred_texts[sample_idx]
            gt_text = gt_texts[sample_idx]
            if not include_correct and pred_text == gt_text:
                continue

            filename = None
            if isinstance(filenames, (list, tuple)) and sample_idx < len(filenames):
                filename = str(filenames[sample_idx])

            sample = HighLossSample(
                epoch=self._current_epoch(),
                global_step=self._current_global_step(),
                batch_idx=batch_idx,
                sample_idx=sample_idx,
                loss=loss_val,
                gt_text=gt_text,
                pred_text=pred_text,
                filename=filename,
                image_ref=images[sample_idx].detach().cpu(),
            )
            self._push_high_loss_sample(sample, top_k)

    def _push_high_loss_sample(self, sample: HighLossSample, top_k: int) -> None:
        entry = (sample.loss, self._high_loss_counter, sample)
        self._high_loss_counter += 1

        if len(self._high_loss_epoch_buffer) < top_k:
            heapq.heappush(self._high_loss_epoch_buffer, entry)
            return

        if sample.loss > self._high_loss_epoch_buffer[0][0]:
            heapq.heapreplace(self._high_loss_epoch_buffer, entry)

    def _log_high_loss_audit(self) -> None:
        if not self._should_collect_high_loss_samples():
            self._clear_high_loss_epoch_buffer()
            return

        if not self._high_loss_epoch_buffer:
            return

        wandb_run = self._get_wandb_experiment()
        if wandb_run is None:
            logger.warning(
                "Skipping high-loss audit logging: no active WandB experiment attached."
            )
            self._clear_high_loss_epoch_buffer()
            return

        ranked_samples = [
            item[2] for item in sorted(self._high_loss_epoch_buffer, key=lambda item: item[0], reverse=True)
        ]

        from ocr.domains.recognition.callbacks.wandb_logging import log_recognition_images

        log_recognition_images(
            images=[sample.image_ref for sample in ranked_samples],
            pred_texts=[sample.pred_text for sample in ranked_samples],
            gt_texts=[sample.gt_text for sample in ranked_samples],
            epoch=self._current_epoch(),
            limit=len(ranked_samples),
            seed=42,
            filenames=[sample.filename for sample in ranked_samples],
            caption_prefix="audit/high_loss_samples",
            max_image_side=self._high_loss_audit_max_image_side(),
            patch_native_view=self._high_loss_audit_patch_native_view(),
            wandb_run=wandb_run,
        )

        if self._high_loss_audit_include_table():
            self._log_high_loss_table(ranked_samples, wandb_run)

        self._clear_high_loss_epoch_buffer()

    def _log_high_loss_table(self, samples: list[HighLossSample], wandb_run) -> None:
        import wandb

        table = wandb.Table(
            columns=[
                "epoch",
                "global_step",
                "batch_idx",
                "sample_idx",
                "loss",
                "gt_text",
                "pred_text",
                "filename",
                "is_exact_match",
            ]
        )

        for sample in samples:
            table.add_data(
                sample.epoch,
                sample.global_step,
                sample.batch_idx,
                sample.sample_idx,
                sample.loss,
                sample.gt_text,
                sample.pred_text,
                sample.filename,
                sample.pred_text == sample.gt_text,
            )

        wandb_run.log({"audit/high_loss_table": table, "epoch": self._current_epoch()})

    def _compute_per_sample_validation_loss(self, pred, inference_out, batch):
        import torch.nn.functional as F

        text_tokens = batch.get("text_tokens", None)
        if not isinstance(text_tokens, torch.Tensor) or text_tokens.ndim != 2:
            return None

        logits = None
        if is_config(pred) and "logits" in pred:
            logits = pred["logits"]
        elif is_config(inference_out) and "logits" in inference_out:
            logits = inference_out["logits"]

        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            return None

        targets = text_tokens[:, 1:].to(logits.device)
        max_steps = min(logits.size(1), targets.size(1))
        if max_steps <= 0:
            return None

        logits = logits[:, :max_steps, :]
        targets = targets[:, :max_steps]

        loss_flat = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=0,
            reduction="none",
        )
        token_losses = loss_flat.view(logits.size(0), max_steps)
        valid_mask = targets != 0

        valid_counts = valid_mask.sum(dim=1)
        loss_sums = (token_losses * valid_mask).sum(dim=1)
        per_sample_loss = loss_sums / valid_counts.clamp_min(1)

        per_sample_loss = torch.where(
            valid_counts > 0,
            per_sample_loss,
            torch.full_like(per_sample_loss, float("nan")),
        )
        return per_sample_loss.detach().cpu()

    def _high_loss_audit_cfg(self):
        wandb_cfg = self._get_wandb_cfg()
        if not is_config(wandb_cfg):
            return {}
        cfg = wandb_cfg.get("high_loss_audit", {})
        return cfg if is_config(cfg) else {}

    def _high_loss_audit_enabled(self) -> bool:
        return bool(self._high_loss_audit_cfg().get("enabled", False))

    def _high_loss_audit_top_k(self) -> int:
        raw_value = self._high_loss_audit_cfg().get("top_k", 16)
        try:
            top_k = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError("train.logger.wandb.high_loss_audit.top_k must be an integer.") from None

        if top_k <= 0:
            raise ValueError("train.logger.wandb.high_loss_audit.top_k must be > 0.")
        return top_k

    def _high_loss_audit_log_every_n_epochs(self) -> int:
        raw_value = self._high_loss_audit_cfg().get("log_every_n_epochs", 1)
        try:
            every_n = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                "train.logger.wandb.high_loss_audit.log_every_n_epochs must be an integer."
            ) from None

        if every_n < 1:
            raise ValueError(
                "train.logger.wandb.high_loss_audit.log_every_n_epochs must be >= 1."
            )
        return every_n

    def _high_loss_audit_min_global_step(self) -> int:
        raw_value = self._high_loss_audit_cfg().get("min_global_step", 0)
        try:
            min_step = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                "train.logger.wandb.high_loss_audit.min_global_step must be an integer."
            ) from None

        if min_step < 0:
            raise ValueError(
                "train.logger.wandb.high_loss_audit.min_global_step must be >= 0."
            )
        return min_step

    def _high_loss_audit_max_image_side(self) -> int | None:
        raw_value = self._high_loss_audit_cfg().get("max_image_side", 640)
        if raw_value is None:
            return None

        try:
            max_side = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                "train.logger.wandb.high_loss_audit.max_image_side must be an integer or null."
            ) from None

        if max_side <= 0:
            return None
        return max_side

    def _high_loss_audit_include_table(self) -> bool:
        return bool(self._high_loss_audit_cfg().get("include_table", True))

    def _high_loss_audit_include_correct(self) -> bool:
        return bool(self._high_loss_audit_cfg().get("include_correct_but_high_loss", False))

    def _high_loss_audit_patch_native_view(self) -> bool:
        return bool(self._high_loss_audit_cfg().get("patch_native_view", True))

    def _should_collect_high_loss_samples(self) -> bool:
        if not self._high_loss_audit_enabled():
            return False

        if self._current_global_step() < self._high_loss_audit_min_global_step():
            return False

        every_n = self._high_loss_audit_log_every_n_epochs()
        return (self._current_epoch() + 1) % every_n == 0

    def _clear_high_loss_epoch_buffer(self) -> None:
        self._high_loss_epoch_buffer.clear()
        self._high_loss_counter = 0

    def _current_global_step(self) -> int:
        try:
            return int(self.global_step)
        except Exception:
            trainer = getattr(self, "trainer", None) or getattr(self, "_trainer", None)
            return int(getattr(trainer, "global_step", 0))

    def _current_epoch(self) -> int:
        try:
            return int(self.current_epoch)
        except Exception:
            trainer = getattr(self, "trainer", None) or getattr(self, "_trainer", None)
            return int(getattr(trainer, "current_epoch", 0))
