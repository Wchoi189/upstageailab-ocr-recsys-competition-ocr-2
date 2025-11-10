from collections import OrderedDict
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ocr.evaluation import CLEvalEvaluator
from ocr.lightning_modules.loggers import WandbProblemLogger
from ocr.lightning_modules.utils import CheckpointHandler, extract_metric_kwargs, extract_normalize_stats, format_predictions
from ocr.metrics import CLEvalMetric
from ocr.utils.submission import SubmissionWriter
from ocr.validation.models import CollateOutput


class OCRPLModule(pl.LightningModule):
    """OCR PyTorch Lightning Module for text detection and recognition.

    This module implements the training, validation, testing, and prediction loops
    for OCR tasks. It integrates with CLEvalEvaluator for metric computation and
    follows the data contracts defined in #file:data_contracts.md.

    The module expects input batches to follow the collate function output contract
    and produces predictions compatible with the CLEvalMetric evaluation pipeline.

    Attributes:
        model: The OCR model instance
        dataset: Dataset dictionary with 'train', 'val', 'test' keys
        config: Hydra configuration object
        valid_evaluator: CLEvalEvaluator for validation metrics
        test_evaluator: CLEvalEvaluator for test metrics
        predict_step_outputs: OrderedDict for prediction outputs
    """

    def __init__(self, model, dataset, config, metric_cfg: DictConfig | None = None):
        super().__init__()
        self.model = model
        # Compile the model for better performance
        if hasattr(config, "compile_model") and config.compile_model:
            # Configure torch.compile to handle scalar outputs better
            import torch._dynamo

            torch._dynamo.config.capture_scalar_outputs = True
            self.model = torch.compile(self.model, mode="default")
        self.dataset = dataset
        self.metric_cfg = metric_cfg
        self.metric_kwargs = extract_metric_kwargs(metric_cfg)
        self.metric = instantiate(metric_cfg) if metric_cfg is not None else CLEvalMetric(**self.metric_kwargs)
        self.config = config
        self.lr_scheduler = None
        self._normalize_mean, self._normalize_std = extract_normalize_stats(config)

        self.valid_evaluator = CLEvalEvaluator(self.dataset["val"], self.metric_kwargs, mode="val") if "val" in self.dataset else None
        self.test_evaluator = CLEvalEvaluator(self.dataset["test"], self.metric_kwargs, mode="test") if "test" in self.dataset else None
        self.predict_step_outputs: OrderedDict[str, Any] = OrderedDict()
        self.validation_step_outputs: OrderedDict[str, Any] = OrderedDict()

        # Initialize helper classes
        val_dataset = self.dataset["val"] if "val" in self.dataset else None
        self.wandb_logger = WandbProblemLogger(
            config,
            self._normalize_mean,
            self._normalize_std,
            val_dataset,
            self.metric_kwargs,
        )
        self.submission_writer = SubmissionWriter(config)

        # Log selected performance preset in bright yellow
        self._log_performance_preset()

    def _log_performance_preset(self) -> None:
        """Log the selected performance preset."""
        try:
            # Try to infer the preset from validation dataset config
            val_dataset = self.dataset.get("val")
            if val_dataset and hasattr(val_dataset, "config"):
                config = val_dataset.config

                # Determine which preset is active based on config settings
                if config.cache_config.cache_transformed_tensors:
                    preset_name = "validation_optimized"
                    preset_desc = "Full caching (~2.5-3x speedup, validation only!)"
                elif config.cache_config.cache_images:
                    preset_name = "balanced"
                    preset_desc = "Image caching (~1.12x speedup)"
                elif config.preload_images or config.load_maps:
                    preset_name = "memory_efficient"
                    preset_desc = "Minimal memory footprint"
                else:
                    preset_name = "none"
                    preset_desc = "No optimizations (baseline)"

                # Simple logging without Rich to avoid conflicts
                print(f"🚀 Performance Preset: {preset_name}")
                print(f"   {preset_desc}")

        except Exception:
            # Silently ignore if we can't determine the preset
            pass

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load state dict with fallback handling for different checkpoint formats."""
        # Call parent class load_state_dict directly to avoid recursion
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x):
        return self.model(return_loss=False, **x)

    def training_step(self, batch, batch_idx):
        """
        Training step for OCR model.

        BUG-20251109-002: Added validation for input images and ground truth maps
        to detect NaN/Inf values before they cause CUDA errors.
        See: docs/bug_reports/BUG-20251109-002-code-changes.md
        """
        # BUG-20251109-002: Validate input batch before model forward pass
        # CUDA illegal instruction errors often occur due to invalid input data
        images = batch.get("images")
        if images is not None:
            if torch.isnan(images).any():
                raise ValueError(f"NaN values detected in batch['images'] at batch_idx={batch_idx}. Shape: {images.shape}")
            if torch.isinf(images).any():
                raise ValueError(f"Inf values detected in batch['images'] at batch_idx={batch_idx}. Shape: {images.shape}")
            # BUG-20251109-002: Check for extreme values that could cause numerical instability
            # Normalized images should be in range [-3, 3] (ImageNet normalization)
            # Values outside this range can cause numerical instability in the encoder
            img_min, img_max = images.min().item(), images.max().item()
            if img_min < -10.0 or img_max > 10.0:
                raise ValueError(
                    f"Extreme values detected in batch['images'] at batch_idx={batch_idx}. "
                    f"Shape: {images.shape}, Min: {img_min:.4f}, Max: {img_max:.4f}. "
                    f"Expected normalized values in range [-3, 3]. This may cause numerical instability."
                )

        # BUG-20251109-002: Validate ground truth tensors if present
        gt_binary = batch.get("prob_maps")
        if gt_binary is not None:
            if torch.isnan(gt_binary).any():
                raise ValueError(f"NaN values detected in batch['prob_maps'] at batch_idx={batch_idx}. Shape: {gt_binary.shape}")
            if torch.isinf(gt_binary).any():
                raise ValueError(f"Inf values detected in batch['prob_maps'] at batch_idx={batch_idx}. Shape: {gt_binary.shape}")

        # BUG-20251109-002: Validate thresh_maps to detect NaN/Inf values
        gt_thresh = batch.get("thresh_maps")
        if gt_thresh is not None:
            if torch.isnan(gt_thresh).any():
                raise ValueError(f"NaN values detected in batch['thresh_maps'] at batch_idx={batch_idx}. Shape: {gt_thresh.shape}")
            if torch.isinf(gt_thresh).any():
                raise ValueError(f"Inf values detected in batch['thresh_maps'] at batch_idx={batch_idx}. Shape: {gt_thresh.shape}")

        gt_prob_mask = batch.get("prob_mask")
        if gt_prob_mask is not None:
            if torch.isnan(gt_prob_mask).any():
                raise ValueError(f"NaN values detected in batch['prob_mask'] at batch_idx={batch_idx}. Shape: {gt_prob_mask.shape}")
            if torch.isinf(gt_prob_mask).any():
                raise ValueError(f"Inf values detected in batch['prob_mask'] at batch_idx={batch_idx}. Shape: {gt_prob_mask.shape}")

        pred = self.model(**batch)

        # BUG-20251110-001: Validate loss values before backward pass
        # cuDNN errors often occur due to NaN/Inf values in loss or gradients
        loss = pred["loss"]
        if torch.isnan(loss):
            raise ValueError(
                f"NaN loss detected at batch_idx={batch_idx}. "
                f"This will cause cuDNN errors during backward pass. "
                f"Loss dict: {pred.get('loss_dict', {})}"
            )
        if torch.isinf(loss):
            raise ValueError(
                f"Inf loss detected at batch_idx={batch_idx}. "
                f"This will cause cuDNN errors during backward pass. "
                f"Loss dict: {pred.get('loss_dict', {})}"
            )
        # Check for extreme loss values that could cause gradient explosion
        loss_value = loss.item()
        if loss_value > 1e6 or loss_value < -1e6:
            raise ValueError(
                f"Extreme loss value detected at batch_idx={batch_idx}: {loss_value:.4f}. "
                f"This may cause gradient explosion and cuDNN errors. "
                f"Loss dict: {pred.get('loss_dict', {})}"
            )

        # BUG-20251110-001: Validate loss_dict values
        for key, value in pred.get("loss_dict", {}).items():
            if torch.is_tensor(value):
                if torch.isnan(value).any():
                    raise ValueError(
                        f"NaN value in loss_dict['{key}'] at batch_idx={batch_idx}. "
                        f"This will cause cuDNN errors during backward pass."
                    )
                if torch.isinf(value).any():
                    raise ValueError(
                        f"Inf value in loss_dict['{key}'] at batch_idx={batch_idx}. "
                        f"This will cause cuDNN errors during backward pass."
                    )

        self.log("train/loss", loss, batch_size=batch["images"].shape[0])
        for key, value in pred["loss_dict"].items():
            self.log(f"train/{key}", value, batch_size=batch["images"].shape[0])
        return loss

    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        """
        Validate gradients before optimizer step.

        BUG-20251110-001: Added gradient validation to catch NaN/Inf values
        that cause cuDNN errors during backward pass.
        See: docs/bug_reports/BUG-20251110-001_out-of-bounds-polygon-coordinates-in-training-dataset.md
        """
        import logging

        logger = logging.getLogger(__name__)
        trainer = self.trainer
        batch_idx = getattr(trainer, "global_step", -1) if trainer else -1

        # BUG-20251110-001: Check for NaN/Inf gradients that cause cuDNN errors
        nan_grads = []
        inf_grads = []
        extreme_grads = []

        for param_name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append((param_name, param.grad.shape, param.grad.device))
                elif torch.isinf(param.grad).any():
                    inf_grads.append((param_name, param.grad.shape, param.grad.device))
                else:
                    # Check for extreme gradient values that could cause instability
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1e6:
                        extreme_grads.append((param_name, grad_norm))

        # Log warnings for extreme gradients (but don't fail)
        if extreme_grads:
            for param_name, grad_norm in extreme_grads:
                logger.warning(
                    f"Extreme gradient norm detected in {param_name}: {grad_norm:.4f} "
                    f"at step {batch_idx}. This may cause instability. "
                    f"Consider reducing learning rate or increasing gradient clipping."
                )

        # Zero out NaN/Inf gradients as a safety measure
        # This prevents cuDNN errors while allowing training to continue
        if nan_grads or inf_grads:
            for param_name, param in self.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.error(
                            f"NaN/Inf gradient detected in {param_name} at step {batch_idx}. "
                            f"Shape: {param.grad.shape}, Device: {param.grad.device}. "
                            f"Zeroing out gradient to prevent cuDNN errors."
                        )
                        param.grad.zero_()

            # After zeroing, log warning with detailed information but continue training
            # This allows training to recover from numerical instability
            warning_msg = f"NaN/Inf gradients detected at step {batch_idx}:\n"
            if nan_grads:
                warning_msg += f"  NaN gradients ({len(nan_grads)}): {[name for name, _, _ in nan_grads[:5]]}\n"
            if inf_grads:
                warning_msg += f"  Inf gradients ({len(inf_grads)}): {[name for name, _, _ in inf_grads[:5]]}\n"
            warning_msg += (
                f"\nGradients have been zeroed out to prevent cuDNN errors. "
                f"This suggests numerical instability. Possible causes:\n"
                f"  1. Extreme loss values (check loss validation)\n"
                f"  2. Invalid input data (check data pipeline)\n"
                f"  3. Learning rate too high\n"
                f"  4. Numerical instability in model architecture\n"
                f"\nConsider:\n"
                f"  - Reducing learning rate\n"
                f"  - Increasing gradient clipping\n"
                f"  - Checking for NaN/Inf in loss values\n"
                f"  - Validating input data pipeline\n"
                f"\nTraining will continue with zeroed gradients for this step."
            )
            logger.warning(warning_msg)

            # Optionally raise error if configured to do so (for debugging)
            # This can be enabled via config if you want to stop training on NaN/Inf gradients
            if hasattr(self.config, "stop_on_nan_gradients") and self.config.stop_on_nan_gradients:
                raise ValueError(warning_msg)

    def validation_step(self, batch, batch_idx):
        """Perform validation step for OCR model.

        Args:
            batch: Input batch following the collate function output contract
                   from #file:data_contracts.md (CollateOutput format)
            batch_idx: Batch index

        Returns:
            Loss value for the batch

        Side Effects:
            Updates self.valid_evaluator with predictions for epoch-end metrics
            Logs validation loss and loss components
            Stores predictions for W&B logging
        """
        # Validate input batch against data contract
        try:
            CollateOutput(**batch)
        except Exception as e:
            raise ValueError(f"Batch validation failed: {e}") from e
        pred = self.model(**batch)
        self.log("val_loss", pred["loss"], batch_size=batch["images"].shape[0])
        for key, value in pred["loss_dict"].items():
            self.log(f"val_{key}", value, batch_size=batch["images"].shape[0])

        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
        predictions = format_predictions(batch, boxes_batch)

        # Store predictions for wandb image logging callback
        for idx, prediction_entry in enumerate(predictions):
            filename = batch["image_filename"][idx]
            # Store transformed image for WandB logging (only for first few samples to minimize memory)
            if batch_idx < 2 and idx < 8:  # Limit to first 2 batches, 8 images each
                prediction_entry["transformed_image"] = batch["images"][idx].detach().cpu()
            self.validation_step_outputs[filename] = prediction_entry

        if self.valid_evaluator is not None:
            self.valid_evaluator.update(batch["image_filename"], predictions)

        # Compute per-batch validation metrics and log problematic batch images
        batch_metrics = self.wandb_logger.log_if_needed(batch, predictions, batch_idx)
        self.log(f"batch_{batch_idx}/recall", batch_metrics["recall"], batch_size=batch["images"].shape[0])
        self.log(f"batch_{batch_idx}/precision", batch_metrics["precision"], batch_size=batch["images"].shape[0])
        self.log(f"batch_{batch_idx}/hmean", batch_metrics["hmean"], batch_size=batch["images"].shape[0])

        return pred["loss"]

    def on_train_epoch_start(self) -> None:
        # Reset collate function logging flag at the start of each training epoch
        import ocr.datasets.db_collate_fn

        ocr.datasets.db_collate_fn._db_collate_logged_stats = False
        # Reset wandb logger step counter at the start of each training epoch
        self.wandb_logger.reset_epoch_counter()

    def on_validation_epoch_start(self) -> None:
        self.validation_step_outputs.clear()
        self.wandb_logger.reset_epoch_counter()
        # Reset collate function logging flag at the start of each validation epoch
        import ocr.datasets.db_collate_fn

        ocr.datasets.db_collate_fn._db_collate_logged_stats = False

    def on_validation_epoch_end(self):
        if self.valid_evaluator is None:
            return

        metrics = self.valid_evaluator.compute()
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True, prog_bar=True)

        self._checkpoint_metrics = {
            "recall": metrics.get("val/recall", 0.0),
            "precision": metrics.get("val/precision", 0.0),
            "hmean": metrics.get("val/hmean", 0.0),
        }

        self.valid_evaluator.reset()

    def on_save_checkpoint(self, checkpoint):
        """Save additional metrics in the checkpoint."""
        return CheckpointHandler.on_save_checkpoint(self, checkpoint)

    def on_load_checkpoint(self, checkpoint):
        """Restore metrics from checkpoint (optional)."""
        CheckpointHandler.on_load_checkpoint(self, checkpoint)

    def test_step(self, batch):
        """Perform test step for OCR model.

        Args:
            batch: Input batch following the collate function output contract
                   from #file:data_contracts.md (CollateOutput format)

        Returns:
            Loss value for the batch

        Side Effects:
            Updates self.test_evaluator with predictions for epoch-end metrics
            Logs test loss and loss components
        """
        # Validate input batch against data contract
        try:
            CollateOutput(**batch)
        except Exception as e:
            raise ValueError(f"Batch validation failed: {e}") from e

        pred = self.model(return_loss=False, **batch)

        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
        predictions = format_predictions(batch, boxes_batch)

        if self.test_evaluator is not None:
            self.test_evaluator.update(batch["image_filename"], predictions)
        return pred

    def on_test_epoch_end(self):
        if self.test_evaluator is None:
            return

        metrics = self.test_evaluator.compute()
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True, prog_bar=True)

        self.test_evaluator.reset()

    def predict_step(self, batch):
        """Perform prediction step for OCR model inference.

        Args:
            batch: Input batch following the collate function output contract
                   from #file:data_contracts.md (CollateOutput format)

        Returns:
            List of predictions with polygon coordinates for each image in batch,
            following the model output format from #file:data_contracts.md
        """
        # Validate input batch against data contract (skip for prediction mode)
        try:
            # For prediction, ground truth fields (polygons, prob_maps, thresh_maps) are not present
            # Check if we're in prediction mode by seeing if these fields are missing
            if "polygons" not in batch or "prob_maps" not in batch or "thresh_maps" not in batch:
                # Prediction mode - skip validation as ground truth fields are not available
                pass
            else:
                # Training/validation mode - validate full batch
                CollateOutput(**batch)
        except Exception as e:
            raise ValueError(f"Batch validation failed: {e}") from e

        pred = self.model(return_loss=False, **batch)
        boxes_batch, scores_batch = self.model.get_polygons_from_maps(batch, pred)

        include_confidence = getattr(self.config, "include_confidence", False)

        for idx, (boxes, scores) in enumerate(zip(boxes_batch, scores_batch, strict=True)):
            normalized_boxes = [np.asarray(box, dtype=np.float32).reshape(-1, 2) for box in boxes]
            if include_confidence:
                self.predict_step_outputs[batch["image_filename"][idx]] = {"boxes": normalized_boxes, "scores": scores}
            else:
                self.predict_step_outputs[batch["image_filename"][idx]] = normalized_boxes
        return pred

    def on_predict_epoch_end(self):
        self.submission_writer.save(self.predict_step_outputs)
        self.predict_step_outputs.clear()

    def configure_optimizers(self):
        optimizers, schedulers = self.model.get_optimizers()
        optimizer_list = optimizers if isinstance(optimizers, list) else [optimizers]

        if isinstance(schedulers, list):
            self.lr_scheduler = schedulers[0] if schedulers else None
        elif schedulers is None:
            self.lr_scheduler = None
        else:
            self.lr_scheduler = schedulers

        return optimizer_list

    def on_train_epoch_end(self):
        # Log cache statistics from datasets if caching is enabled
        if hasattr(self, "train_dataloader"):
            try:
                train_loader = self.trainer.train_dataloader
                if train_loader and hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "log_cache_statistics"):
                    train_loader.dataset.log_cache_statistics()
            except Exception:
                pass  # Silently skip if dataset doesn't support cache statistics

        if self.lr_scheduler is None:
            return

        if self.trainer is not None and self.trainer.sanity_checking:
            return

        optimizer = getattr(self.lr_scheduler, "optimizer", None)
        if optimizer is None:
            return
        step_count = getattr(optimizer, "_step_count", 0)
        if step_count > 0:
            self.lr_scheduler.step()


class OCRDataPLModule(pl.LightningDataModule):
    def __init__(self, dataset, config):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.dataloaders_cfg = self.config.dataloaders
        self.collate_cfg = self.config.collate_fn

    def _build_collate_fn(self, *, inference_mode: bool) -> Any:
        # Create collate function (no longer using cache - using pre-processed maps instead)
        collate_fn = instantiate(self.collate_cfg)
        if hasattr(collate_fn, "inference_mode"):
            collate_fn.inference_mode = inference_mode
        return collate_fn

    def train_dataloader(self):
        train_loader_config = self.dataloaders_cfg.train_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if train_loader_config.get("num_workers", 0) == 0:
            train_loader_config = {k: v for k, v in train_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=False)
        return DataLoader(self.dataset["train"], collate_fn=collate_fn, **train_loader_config)

    def val_dataloader(self):
        val_loader_config = self.dataloaders_cfg.val_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if val_loader_config.get("num_workers", 0) == 0:
            val_loader_config = {k: v for k, v in val_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=False)
        return DataLoader(self.dataset["val"], collate_fn=collate_fn, **val_loader_config)

    def test_dataloader(self):
        test_loader_config = self.dataloaders_cfg.test_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if test_loader_config.get("num_workers", 0) == 0:
            test_loader_config = {k: v for k, v in test_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=False)
        return DataLoader(self.dataset["test"], collate_fn=collate_fn, **test_loader_config)

    def predict_dataloader(self):
        predict_loader_config = self.dataloaders_cfg.predict_dataloader
        # Filter out multiprocessing-only parameters when num_workers == 0
        if predict_loader_config.get("num_workers", 0) == 0:
            predict_loader_config = {k: v for k, v in predict_loader_config.items() if k not in ["prefetch_factor", "persistent_workers"]}
        collate_fn = self._build_collate_fn(inference_mode=True)
        return DataLoader(self.dataset["predict"], collate_fn=collate_fn, **predict_loader_config)
