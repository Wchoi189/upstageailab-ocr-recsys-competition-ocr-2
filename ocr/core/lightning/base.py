"""Base PyTorch Lightning Module for OCR tasks.

This provides shared functionality for detection and recognition modules.
Domain-specific logic is implemented in:
- ocr.domains.detection.module (DetectionPLModule)
- ocr.domains.recognition.module (RecognitionPLModule)
"""

from abc import abstractmethod
import logging

import torch

import lightning.pytorch as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ocr.core.lightning.utils import CheckpointHandler, extract_metric_kwargs, extract_normalize_stats
from ocr.core.utils.config_utils import is_config
from ocr.core.utils.wandb_base import finalize_run


logger = logging.getLogger(__name__)


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
        self.metric = instantiate(metric_cfg) if metric_cfg is not None else None
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

    def on_fit_end(self):
        """Finalize W&B run name using the best available metric."""
        if not self._wandb_enabled():
            return

        metrics = {}
        if self.trainer is not None:
            metrics = dict(self.trainer.callback_metrics)

        try:
            finalize_run(metrics)
        except Exception:
            logger.exception("Failed to finalize W&B run name.")

    def _get_wandb_cfg(self):
        if hasattr(self.config, "train") and hasattr(self.config.train, "logger"):
            logger_cfg = self.config.train.logger
            if is_config(logger_cfg) and "wandb" in logger_cfg:
                return logger_cfg.get("wandb")
        return None

    def _wandb_enabled(self) -> bool:
        wandb_cfg = self._get_wandb_cfg()
        if is_config(wandb_cfg):
            return wandb_cfg.get("enabled", False)
        return False

    def _get_wandb_experiment(self):
        """Return active WandB experiment from trainer loggers when available."""
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None

        loggers = getattr(trainer, "loggers", None)
        if not loggers:
            return None

        for logger_instance in loggers:
            if logger_instance is None:
                continue

            logger_type = type(logger_instance).__name__
            logger_module = getattr(type(logger_instance), "__module__", "").lower()
            if logger_type == "WandbLogger" or "wandb" in logger_module:
                try:
                    return logger_instance.experiment
                except Exception:
                    return None

        return None

    def _wandb_image_logging_enabled(self) -> bool:
        """Override in subclasses to enable image logging."""
        return False



    def on_save_checkpoint(self, checkpoint):
        """Save additional metrics in the checkpoint."""
        return CheckpointHandler.on_save_checkpoint(self, checkpoint)

    def on_load_checkpoint(self, checkpoint):
        """Restore metrics from checkpoint."""
        CheckpointHandler.on_load_checkpoint(self, checkpoint)

    def configure_optimizers(self):
        """Configure optimizers from V5 Hydra config ONLY.

        V5 Standard: config.train.optimizer (Hydra _target_)
        NO LEGACY SUPPORT. NO FALLBACKS. FAIL FAST.

        Raises:
            ValueError: If config.train.optimizer is missing or invalid
        """
        if not hasattr(self.config, "train") or not hasattr(self.config.train, "optimizer"):
            raise ValueError(
                "V5 Hydra config missing: config.train.optimizer is required.\n"
                "Legacy model.get_optimizers() is no longer supported.\n"
                "See configs/train/optimizer/adam.yaml for template."
            )

        opt_cfg = self.config.train.optimizer

        # Hydra instantiate ONLY - no manual fallbacks
        optimizer = instantiate(opt_cfg, params=self.model.parameters())

        if hasattr(self.config.train, "lr_scheduler") and self.config.train.lr_scheduler:
            scheduler_cfg = self.config.train.lr_scheduler
            if is_config(scheduler_cfg):
                scheduler_cfg = OmegaConf.to_container(scheduler_cfg, resolve=True)

            # Extract Lightning-specific metadata (not passed to scheduler __init__)
            lightning_metadata = {}
            warmup_epochs = 0
            warmup_start_factor = 0.1

            if isinstance(scheduler_cfg, dict):
                # Extract Lightning scheduler config keys
                lightning_metadata['monitor'] = scheduler_cfg.pop("monitor", None)
                lightning_metadata['interval'] = scheduler_cfg.pop("interval", "epoch")
                lightning_metadata['frequency'] = scheduler_cfg.pop("frequency", 1)

                # Extract warmup config
                warmup_epochs = int(scheduler_cfg.pop("warmup_epochs", 0))
                warmup_start_factor = float(scheduler_cfg.pop("warmup_start_factor", warmup_start_factor))

            scheduler = instantiate(scheduler_cfg, optimizer=optimizer)

            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=warmup_start_factor,
                    total_iters=warmup_epochs,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[warmup_epochs],
                )

            self.lr_scheduler = scheduler

            # Return Lightning format for ReduceLROnPlateau and other metric-based schedulers
            if lightning_metadata.get('monitor'):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": lightning_metadata['monitor'],
                        "interval": lightning_metadata['interval'],
                        "frequency": lightning_metadata['frequency'],
                    }
                }

        return optimizer

    def on_train_epoch_end(self):
        """Handle cache statistics logging.

        Note: LR scheduler stepping is now handled by Lightning when using
        ReduceLROnPlateau or other metric-based schedulers via the dict return
        format in configure_optimizers. For simple schedulers, manual stepping
        is still performed here.
        """
        # Log cache statistics from datasets if caching is enabled
        if hasattr(self, "train_dataloader"):
            try:
                train_loader = self.trainer.train_dataloader
                if train_loader and hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "log_cache_statistics"):
                    train_loader.dataset.log_cache_statistics()
            except Exception:
                pass  # Silently skip if dataset doesn't support cache statistics

        # Only manually step scheduler if Lightning isn't managing it
        # (i.e., when configure_optimizers returned just an optimizer, not a dict)
        if self.lr_scheduler is None:
            return

        if self.trainer is not None and self.trainer.sanity_checking:
            return

        # Check if scheduler is ReduceLROnPlateau (Lightning manages these)
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return  # Lightning handles stepping via the dict config

        optimizer = getattr(self.lr_scheduler, "optimizer", None)
        if optimizer is None:
            return
        step_count = getattr(optimizer, "_step_count", 0)
        if step_count > 0:
            self.lr_scheduler.step()
