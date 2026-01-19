"""OCR Project Orchestrator - Bridges V5.0 Hydra configs to PyTorch Lightning.

This orchestrator implements the "Domains First" architecture by:
1. Delegating to existing model/dataset factories
2. Domain-specific Lightning module routing
3. Trainer configuration from merged Hydra tiers
4. Vocab size injection for recognition models
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import Trainer
import logging

from ocr.core.models import get_model_by_cfg
from ocr.data.datasets import get_datasets_by_cfg
from ocr.data.lightning_data import OCRDataPLModule

logger = logging.getLogger(__name__)


class OCRProjectOrchestrator:
    """Orchestrates OCR training/evaluation pipeline with V5.0 Hydra configs.

    This class bridges V5.0 "Domains First" configs to the existing
    infrastructure, handling vocab injection and domain routing.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize orchestrator with Hydra configuration.

        Args:
            cfg: Resolved Hydra configuration
        """
        self.cfg = cfg
        self.domain = cfg.get("domain", cfg.get("task", "detection"))
        self.mode = cfg.get("mode", "train")

        logger.info("🎯 OCRProjectOrchestrator initialized")
        logger.info(f"   Domain: {self.domain}")
        logger.info(f"   Mode: {self.mode}")

    def _inject_vocab_size(self):
        """Inject vocab_size into recognition model config.

        This handles the Recognition-specific dependency where the
        model head and decoder need to know the tokenizer vocab size.
        """
        if "tokenizer" not in self.cfg.data:
            return

        logger.info("💉 Injecting vocab_size for recognition model...")
        tokenizer = hydra.utils.instantiate(self.cfg.data.tokenizer)

        # Disable struct mode to allow runtime injection
        OmegaConf.set_struct(self.cfg, False)

        if "component_overrides" in self.cfg.model and self.cfg.model.component_overrides:
            # Inject into head (output layer)
            if "head" in self.cfg.model.component_overrides:
                if "params" not in self.cfg.model.component_overrides.head:
                    self.cfg.model.component_overrides.head.params = {}
                self.cfg.model.component_overrides.head.params.out_channels = tokenizer.vocab_size
                logger.info(f"   ✓ Head out_channels = {tokenizer.vocab_size}")

            # Inject into decoder (transformer embeddings)
            if "decoder" in self.cfg.model.component_overrides:
                if "params" not in self.cfg.model.component_overrides.decoder:
                    self.cfg.model.component_overrides.decoder.params = {}
                self.cfg.model.component_overrides.decoder.params.vocab_size = tokenizer.vocab_size
                logger.info(f"   ✓ Decoder vocab_size = {tokenizer.vocab_size}")

    def setup_modules(self):
        """Create Lightning modules using existing factories.

        Returns:
            Tuple of (pl_module, data_module)
        """
        logger.info("🏗️ Building model and datasets...")

        # Inject vocab size for recognition domain
        if self.domain == "recognition":
            self._inject_vocab_size()

        # Use existing model factory
        model = get_model_by_cfg(self.cfg.model)
        logger.info(f"   ✓ Model created: {type(model).__name__}")

        # Use existing dataset factory
        data_config = getattr(self.cfg, "data", None)
        dataset = get_datasets_by_cfg(self.cfg.data, data_config, self.cfg)
        logger.info("   ✓ Datasets created")

        # Extract metric config
        metric_cfg = None
        if "metrics" in self.cfg and "eval" in self.cfg.metrics:
            metric_cfg = self.cfg.metrics.eval

        # Domain-specific Lightning module routing
        if self.domain == "detection":
            from ocr.domains.detection.module import DetectionPLModule
            pl_module = DetectionPLModule(
                model=model,
                dataset=dataset,
                config=self.cfg,
                metric_cfg=metric_cfg
            )
            logger.info("   ✓ DetectionPLModule created")
        elif self.domain == "recognition":
            from ocr.domains.recognition.module import RecognitionPLModule
            pl_module = RecognitionPLModule(
                model=model,
                dataset=dataset,
                config=self.cfg,
                metric_cfg=metric_cfg
            )
            logger.info("   ✓ RecognitionPLModule created")
        else:
            raise ValueError(
                f"Unknown domain: {self.domain}. "
                f"Must be 'detection' or 'recognition'."
            )

        # Create data module
        data_module = OCRDataPLModule(dataset=dataset, config=self.cfg)
        logger.info("   ✓ DataModule created")

        return pl_module, data_module

    def setup_trainer(self):
        """Build PyTorch Lightning Trainer from V5.0 Hydra configs.

        Returns:
            Configured Trainer instance
        """
        logger.info("⚡ Configuring Lightning Trainer...")

        # Merge configs from multiple tiers
        trainer_kwargs = {}

        # Tier 1: Global trainer defaults
        if hasattr(self.cfg, "trainer"):
            trainer_kwargs.update(self.cfg.trainer)

        # Tier 2: Hardware settings (accelerator, devices, precision)
        if hasattr(self.cfg, "hardware"):
            if hasattr(self.cfg.hardware, "accelerator"):
                trainer_kwargs["accelerator"] = self.cfg.hardware.accelerator
            if hasattr(self.cfg.hardware, "devices"):
                trainer_kwargs["devices"] = self.cfg.hardware.devices
            if hasattr(self.cfg.hardware, "precision"):
                trainer_kwargs["precision"] = self.cfg.hardware.precision

        # Tier 7: Training configs (loggers, callbacks)
        if hasattr(self.cfg, "train"):
            # Instantiate loggers
            if hasattr(self.cfg.train, "logger") and self.cfg.train.logger:
                loggers = [
                    hydra.utils.instantiate(logger_cfg)
                    for logger_cfg in self.cfg.train.logger.values()
                ]
                trainer_kwargs["logger"] = loggers
                logger.info(f"   ✓ {len(loggers)} logger(s) configured")

            # Instantiate callbacks
            if hasattr(self.cfg.train, "callbacks") and self.cfg.train.callbacks:
                callbacks = [
                    hydra.utils.instantiate(cb_cfg)
                    for cb_cfg in self.cfg.train.callbacks.values()
                ]
                trainer_kwargs["callbacks"] = callbacks
                logger.info(f"   ✓ {len(callbacks)} callback(s) configured")

        trainer = Trainer(**trainer_kwargs)
        logger.info("   ✓ Trainer ready")
        return trainer

    def run(self):
        """Execute the orchestrated pipeline."""
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting {self.mode.upper()} for {self.domain} domain")
        logger.info(f"{'='*60}\n")

        # Setup components
        pl_module, data_module = self.setup_modules()
        trainer = self.setup_trainer()

        # Execute based on mode
        checkpoint_path = self.cfg.get("checkpoint_path", None)

        if self.mode == "train":
            logger.info("🏋️ Starting training...\n")
            trainer.fit(pl_module, data_module, ckpt_path=checkpoint_path)
            logger.info("\n✅ Training complete!")

        elif self.mode == "eval" or self.mode == "test":
            if not checkpoint_path:
                raise ValueError("checkpoint_path required for eval/test mode")
            logger.info(f"🧪 Starting evaluation from {checkpoint_path}...\n")
            trainer.test(pl_module, data_module, ckpt_path=checkpoint_path)
            logger.info("\n✅ Evaluation complete!")

        elif self.mode == "predict":
            logger.info("🔮 Starting prediction...\n")
            trainer.predict(pl_module, data_module, ckpt_path=checkpoint_path)
            logger.info("\n✅ Prediction complete!")

        else:
            raise ValueError(
                f"Unknown mode: {self.mode}. "
                f"Must be 'train', 'eval', 'test', or 'predict'."
            )


__all__ = ["OCRProjectOrchestrator"]
