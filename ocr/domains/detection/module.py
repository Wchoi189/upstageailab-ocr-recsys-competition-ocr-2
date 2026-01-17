"""Detection-specific PyTorch Lightning Module."""

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from pydantic import ValidationError

from ocr.core.lightning.base import OCRPLModule
from ocr.core.lightning.loggers import WandbProblemLogger
from ocr.core.lightning.utils import format_predictions
from ocr.domains.detection.evaluation import CLEvalEvaluator
from ocr.core.utils.submission import SubmissionWriter
from ocr.core.validation import CollateOutput, ValidatedTensorData



class DetectionPLModule(OCRPLModule):
    """Detection-specific Lightning Module for text detection tasks.

    Implements detection-specific validation logic:
    - Polygon extraction from probability/threshold maps
    - CLEval metric computation
    - WandB detection image logging

    Attributes:
        valid_evaluator: CLEvalEvaluator for validation metrics
        test_evaluator: CLEvalEvaluator for test metrics
        validation_step_outputs: Storage for validation predictions
        predict_step_outputs: Storage for prediction outputs
        wandb_logger: WandB problem logger for detection
        submission_writer: Handles submission file generation
    """

    def __init__(self, model, dataset, config, metric_cfg=None):
        super().__init__(model, dataset, config, metric_cfg)

        # Detection-specific initialization
        self._valid_evaluator = None
        self._test_evaluator = None
        self.validation_step_outputs: OrderedDict[str, Any] = OrderedDict()
        self.predict_step_outputs: OrderedDict[str, Any] = OrderedDict()

        # Initialize detection helper classes
        val_dataset = self.dataset["val"] if "val" in self.dataset else None
        self.wandb_logger = WandbProblemLogger(
            config,
            self._normalize_mean,
            self._normalize_std,
            val_dataset,
            self.metric_kwargs,
        )
        self.submission_writer = SubmissionWriter(config)

    @property
    def valid_evaluator(self):
        """Lazy-initialized validation evaluator (Performance optimization)."""
        if self._valid_evaluator is None and "val" in self.dataset:
            self._valid_evaluator = CLEvalEvaluator(self.dataset["val"], self.metric_kwargs, mode="val")
        return self._valid_evaluator

    @property
    def test_evaluator(self):
        """Lazy-initialized test evaluator (Performance optimization)."""
        if self._test_evaluator is None and "test" in self.dataset:
            self._test_evaluator = CLEvalEvaluator(self.dataset["test"], self.metric_kwargs, mode="test")
        return self._test_evaluator

    def training_step(self, batch, batch_idx):
        """Detection-specific training step with tensor validation."""
        pred = self.model(**batch)

        # Validate model outputs (BUG-20251112-001/013 prevention)
        try:
            ValidatedTensorData(tensor=pred["loss"], expected_device=batch["images"].device, allow_nan=False, allow_inf=False)

            if "prob_maps" in pred and isinstance(pred["prob_maps"], torch.Tensor):
                ValidatedTensorData(
                    tensor=pred["prob_maps"],
                    expected_device=batch["images"].device,
                    value_range=(0.0, 1.0),
                    allow_nan=False,
                    allow_inf=False,
                )

            if "thresh_maps" in pred and isinstance(pred["thresh_maps"], torch.Tensor):
                ValidatedTensorData(tensor=pred["thresh_maps"], expected_device=batch["images"].device, allow_nan=False, allow_inf=False)
        except ValidationError as exc:
            raise ValueError(f"Training step model output validation failed at step {batch_idx}: {exc}") from exc

        self.log("train/loss", pred["loss"], batch_size=batch["images"].shape[0])
        for key, value in pred["loss_dict"].items():
            self.log(f"train/{key}", value, batch_size=batch["images"].shape[0])
        return pred["loss"]

    def validation_step(self, batch, batch_idx):
        """Detection-specific validation step.

        Extracts polygons from probability/threshold maps and computes detection metrics.
        """
        # Validate input batch against detection data contract
        try:
            CollateOutput(**batch)
        except Exception as e:
            raise ValueError(f"Batch validation failed: {e}") from e

        pred = self.model(**batch)

        # Validate model outputs (BUG-20251112-001/013 prevention)
        try:
            ValidatedTensorData(tensor=pred["loss"], expected_device=batch["images"].device, allow_nan=False, allow_inf=False)

            if "prob_maps" in pred and isinstance(pred["prob_maps"], torch.Tensor):
                ValidatedTensorData(
                    tensor=pred["prob_maps"],
                    expected_device=batch["images"].device,
                    value_range=(0.0, 1.0),
                    allow_nan=False,
                    allow_inf=False,
                )

            if "thresh_maps" in pred and isinstance(pred["thresh_maps"], torch.Tensor):
                ValidatedTensorData(tensor=pred["thresh_maps"], expected_device=batch["images"].device, allow_nan=False, allow_inf=False)
        except ValidationError as exc:
            raise ValueError(f"Validation step model output validation failed at step {batch_idx}: {exc}") from exc

        self.log("val_loss", pred["loss"], batch_size=batch["images"].shape[0])
        for key, value in pred["loss_dict"].items():
            self.log(f"val_{key}", value, batch_size=batch["images"].shape[0])

        # Extract polygons from detection maps
        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
        predictions = format_predictions(batch, boxes_batch)

        # Store predictions for wandb image logging callback
        for idx, prediction_entry in enumerate(predictions):
            filename = batch["image_filename"][idx]
            # Store transformed image for WandB logging (limit to minimize memory)
            if batch_idx < 2 and idx < 8:  # First 2 batches, 8 images each
                prediction_entry["transformed_image"] = batch["images"][idx].detach()
            self.validation_step_outputs[filename] = prediction_entry

        if self.valid_evaluator is not None:
            self.valid_evaluator.update(batch["image_filename"], predictions)

        # Compute per-batch validation metrics and log problematic images
        batch_metrics = self.wandb_logger.log_if_needed(batch, predictions, batch_idx)
        self.log(f"batch_{batch_idx}/recall", batch_metrics["recall"], batch_size=batch["images"].shape[0])
        self.log(f"batch_{batch_idx}/precision", batch_metrics["precision"], batch_size=batch["images"].shape[0])
        self.log(f"batch_{batch_idx}/hmean", batch_metrics["hmean"], batch_size=batch["images"].shape[0])

        return pred["loss"]

    def on_validation_epoch_start(self) -> None:
        """Reset validation outputs and wandb logger at epoch start."""
        super().on_validation_epoch_start()
        self.validation_step_outputs.clear()
        self.wandb_logger.reset_epoch_counter()

    def on_train_epoch_start(self) -> None:
        """Reset wandb logger at training epoch start."""
        super().on_train_epoch_start()
        self.wandb_logger.reset_epoch_counter()

    def on_validation_epoch_end(self):
        """Compute and log epoch-level detection metrics."""
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

    def test_step(self, batch):
        """Detection-specific test step."""
        # Validate input batch
        try:
            CollateOutput(**batch)
        except Exception as e:
            raise ValueError(f"Batch validation failed: {e}") from e

        pred = self.model(return_loss=False, **batch)

        # Extract polygons and update test evaluator
        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
        predictions = format_predictions(batch, boxes_batch)

        if self.test_evaluator is not None:
            self.test_evaluator.update(batch["image_filename"], predictions)

        return pred

    def on_test_epoch_end(self):
        """Compute and log epoch-level test metrics."""
        if self.test_evaluator is None:
            return

        metrics = self.test_evaluator.compute()
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True, prog_bar=True)

        self.test_evaluator.reset()

    def predict_step(self, batch):
        """Detection-specific prediction step."""
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
        """Write submission file and clear prediction outputs."""
        self.submission_writer.save(self.predict_step_outputs)
        self.predict_step_outputs.clear()
