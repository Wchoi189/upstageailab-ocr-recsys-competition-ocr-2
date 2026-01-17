"""Base PyTorch Lightning Module for OCR tasks.

This provides shared functionality for detection and recognition modules.
Domain-specific logic is implemented in:
- ocr.domains.detection.module (DetectionPLModule)
- ocr.domains.recognition.module (RecognitionPLModule)
"""

from abc import abstractmethod

import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from ocr.core.lightning.utils import CheckpointHandler, extract_metric_kwargs, extract_normalize_stats
from ocr.core.metrics import CLEvalMetric


class OCRPLModule(pl.LightningModule):
    """Base OCR PyTorch Lightning Module with shared functionality.

    This abstract base class provides:
    - Model initialization and compilation
    - Optimizer configuration
    - Checkpoint handling
    - Performance preset logging
    - Normalization stats extraction

    Subclasses MUST override:
    - validation_step(): Domain-specific validation logic
    - on_validation_epoch_end(): Domain-specific metric computation

    Attributes:
        model: The OCR model instance
        dataset: Dataset dictionary with 'train', 'val', 'test' keys
        config: Hydra configuration object
        metric_cfg: Metric configuration
        lr_scheduler: Learning rate scheduler instance
    """

    def __init__(self, model, dataset, config, metric_cfg: DictConfig | None = None):
        super().__init__()
        self.model = model

        # Compile the model for better performance if explicitly enabled
        # NOTE: torch.compile adds 10-20s startup overhead - disable during development
        # Enable with: compile_model=true in config
        if hasattr(config, "compile_model") and config.compile_model:
            import torch
            import torch._dynamo

            torch._dynamo.config.capture_scalar_outputs = True
            print("⚡ Compiling model with torch.compile() - this will take 10-20s...")
            self.model = torch.compile(self.model, mode="default")
            print("✓ Model compilation complete")

        self.dataset = dataset
        self.metric_cfg = metric_cfg
        self.metric_kwargs = extract_metric_kwargs(metric_cfg)
        self.metric = instantiate(metric_cfg) if metric_cfg is not None else CLEvalMetric(**self.metric_kwargs)
        self.config = config
        self.lr_scheduler = None
        self._normalize_mean, self._normalize_std = extract_normalize_stats(config)

        # Log selected performance preset
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

                print(f"\n🚀 Performance Preset: {preset_name}")
                print(f"   {preset_desc}\n")

        except Exception:
            # Silently ignore if we can't determine the preset
            pass

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load state dict with fallback handling for different checkpoint formats."""
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x):
        return self.model(return_loss=False, **x)

    def training_step(self, batch, batch_idx):
        """Shared training step implementation."""
        pred = self.model(**batch)

        self.log("train/loss", pred["loss"], batch_size=batch["images"].shape[0])
        for key, value in pred["loss_dict"].items():
            self.log(f"train/{key}", value, batch_size=batch["images"].shape[0])
        return pred["loss"]

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """Domain-specific validation logic.

        Must be implemented by subclasses (DetectionPLModule, RecognitionPLModule).
        """
        pass

    @abstractmethod
    def on_validation_epoch_end(self):
        """Domain-specific epoch-end metric computation.

        Must be implemented by subclasses.
        """
        pass

    def on_train_epoch_start(self) -> None:
        """Reset collate function logging at epoch start."""
        import ocr.data.datasets.db_collate_fn

        ocr.data.datasets.db_collate_fn._db_collate_logged_stats = False

    def on_validation_epoch_start(self) -> None:
        """Reset collate function logging at validation epoch start."""
        import ocr.data.datasets.db_collate_fn

        ocr.data.datasets.db_collate_fn._db_collate_logged_stats = False

    def on_save_checkpoint(self, checkpoint):
        """Save additional metrics in the checkpoint."""
        return CheckpointHandler.on_save_checkpoint(self, checkpoint)

    def on_load_checkpoint(self, checkpoint):
        """Restore metrics from checkpoint."""
        CheckpointHandler.on_load_checkpoint(self, checkpoint)

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
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
        """Handle cache statistics logging and LR scheduler step."""
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
