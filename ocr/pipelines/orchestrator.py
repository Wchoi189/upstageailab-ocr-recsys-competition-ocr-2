"""OCR Project Orchestrator - Bridges V5.0 Hydra configs to PyTorch Lightning.

This orchestrator implements the "Domains First" architecture by:
1. Delegating to existing model/dataset factories
2. Domain-specific Lightning module routing
3. Trainer configuration from merged Hydra tiers
4. Vocab size injection for recognition models
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from ocr.core.models import get_model_by_cfg
from ocr.data.datasets import get_datasets_by_cfg
from ocr.data.lightning_data import OCRDataPLModule
from ocr.core.utils.config_utils import ensure_dict, is_config

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
        # Fix for V5 Structs: If domain is a config/dict, extract 'task'
        domain_cfg = cfg.get("domain", cfg.get("task", "detection"))

        if is_config(domain_cfg):
            self.domain = domain_cfg.get("task", "detection")
        else:
            self.domain = domain_cfg

        # Fallback: Check model target if domain seems wrong (e.g. detection with PARSeq)
        try:
            model_target = str(cfg.model.get("_target_", ""))
            arch_target = str(cfg.model.get("architectures", {}).get("_target_", ""))
            if "PARSeq" in model_target or "PARSeq" in arch_target:
                if self.domain != "recognition":
                    logger.warning(f"⚠️ Mismatch detected! Domain={self.domain} but Model=PARSeq.")
                    logger.warning("⚠️ Forcing domain='recognition' to prevent runtime errors.")
                    self.domain = "recognition"
        except Exception:
            pass # Be safe

        self.mode = cfg.get("mode", "train")

        self._validate_config_structure()

        logger.info("🎯 OCRProjectOrchestrator initialized")

    def _get_required_splits(self):
        """
        Determine which dataset splits are needed for the current mode.

        This enables lazy dataset loading to avoid instantiating unused splits.

        Returns:
            List of required split names
        """
        mode_to_splits = {
            "train": ["train", "val"],
            "eval": ["val"],
            "test": ["test"],
            "predict": ["predict"]
        }
        return mode_to_splits.get(self.mode, ["train", "val"])
        logger.info(f"   Domain: {self.domain}")
        logger.info(f"   Mode: {self.mode}")

    def _validate_config_structure(self):
        """Validate configuration structure to prevent common silent failures."""
        # Check Logger Structure
        if hasattr(self.cfg, "train") and hasattr(self.cfg.train, "logger") and self.cfg.train.logger:
            logging_conf = self.cfg.train.logger
            # If the container ITSELF has a target, it's likely a flat config (Bug!)
            if "_target_" in logging_conf:
                raise RuntimeError(
                    "CRITICAL CONFIG ERROR: 'train.logger' seems to be a single Logger config. "
                    "It MUST be a dictionary/list of loggers. "
                    "Did you forget a nesting wrapper like '@package _group_' or '@package train.logger.wandb'?"
                )

        # Check Callbacks Structure (similar pattern)
        if hasattr(self.cfg, "train") and hasattr(self.cfg.train, "callbacks") and self.cfg.train.callbacks:
            callbacks_conf = self.cfg.train.callbacks
            if "_target_" in callbacks_conf:
                raise RuntimeError(
                    "CRITICAL CONFIG ERROR: 'train.callbacks' seems to be a single Callback config. "
                    "It MUST be a dictionary/list of callbacks."
                )

    def setup_modules(self):
        """Create Lightning modules using existing factories.

        Returns:
            Tuple of (pl_module, data_module)
        """
        logger.info("🏗️ Building model and datasets...")

        # Inject vocab size for recognition domain
        if self.domain == "recognition":
            from ocr.pipelines.strategies.recognition_config import RecognitionConfigStrategy
            RecognitionConfigStrategy.inject_vocab_size(self.cfg)

        # Use existing model factory
        model = get_model_by_cfg(self.cfg.model)
        logger.info(f"   ✓ Model created: {type(model).__name__}")

        # Use existing dataset factory with lazy loading
        data_config = getattr(self.cfg, "data", None)
        required_splits = self._get_required_splits()
        dataset = get_datasets_by_cfg(self.cfg.data, data_config, self.cfg, splits=required_splits)
        logger.info(f"   ✓ Datasets created for splits: {required_splits}")

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
        # Lazy import - defers Lightning strategies loading until trainer setup
        from lightning.pytorch import Trainer

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
                loggers = []
                for logger_cfg in self.cfg.train.logger.values():
                    if is_config(logger_cfg):
                        target = logger_cfg.get("_target_", "")
                        # WandB Configuration Logging Constraint (CONFIG-WANDB-001)
                        # -----------------------------------------------------
                        # Full config logging (log_config=true) disabled by default to prevent
                        # serialization errors when Hydra DictConfig contains callable references
                        # (_target_ fields). Essential config visibility maintained via run naming.
                        # Override at own risk: train.logger.wandb.log_config=true
                        # Spec: AgentQMS/specs/tier2-framework/configuration.spec.md (Section 5)
                        if "WandbLogger" in str(target):
                            standardize_name = bool(logger_cfg.get("standardize_name", False))
                            log_config = bool(logger_cfg.get("log_config", False))

                            if standardize_name:
                                from ocr.core.utils.wandb_base import generate_run_name
                                logger_cfg["name"] = generate_run_name(self.cfg)

                            # Convert entire config to plain Python types to avoid WandB serialization issues
                            # This resolves interpolations and removes OmegaConf wrappers
                            logger_cfg_dict = OmegaConf.to_container(logger_cfg, resolve=True)

                            if log_config:
                                # Add Hydra config as YAML string in plain dict
                                yaml_str = OmegaConf.to_yaml(self.cfg, resolve=True)
                                logger_cfg_dict["config"] = {"hydra_config_yaml": yaml_str}

                            # Remove internal keys and problematic fields that aren't WandB params
                            for internal_key in (
                                "standardize_name",
                                "log_config",
                                "enabled",
                                "log_recognition_images",
                                "per_batch_image_logging",
                                "_recursive_",
                                "settings",  # Remove settings - causes serialization issues with WandB
                            ):
                                logger_cfg_dict.pop(internal_key, None)

                            # Manual instantiation for WandB to avoid Hydra's DictConfig serialization issues
                            from lightning.pytorch.loggers import WandbLogger
                            wandb_logger = WandbLogger(
                                project=logger_cfg_dict.get("project"),
                                name=logger_cfg_dict.get("name"),
                                save_dir=logger_cfg_dict.get("save_dir"),
                                log_model=logger_cfg_dict.get("log_model", False),
                                config=logger_cfg_dict.get("config"),
                            )
                            loggers.append(wandb_logger)
                        else:
                            # Use Hydra instantiation for other loggers
                            loggers.append(hydra.utils.instantiate(logger_cfg))
                    else:
                        # Instantiate non-WandB logger normally
                        loggers.append(hydra.utils.instantiate(logger_cfg))
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
        logger.info(f"\\n{'='*60}")
        logger.info(f"🚀 Starting {self.mode.upper()} for {self.domain} domain")
        logger.info(f"{'='*60}\\n")

        # Setup components
        pl_module, data_module = self.setup_modules()
        trainer = self.setup_trainer()

        # Execute based on mode
        checkpoint_path = self.cfg.get("checkpoint_path", None)

        if self.mode == "train":
            logger.info("🏋️ Starting training...\\n")
            trainer.fit(pl_module, data_module, ckpt_path=checkpoint_path)
            logger.info("\\n✅ Training complete!")

        elif self.mode == "eval" or self.mode == "test":
            if not checkpoint_path:
                raise ValueError("checkpoint_path required for eval/test mode")
            logger.info(f"🧪 Starting evaluation from {checkpoint_path}...\\n")
            trainer.test(pl_module, data_module, ckpt_path=checkpoint_path)
            logger.info("\\n✅ Evaluation complete!")

        elif self.mode == "predict":
            logger.info("🔮 Starting prediction...\\n")
            trainer.predict(pl_module, data_module, ckpt_path=checkpoint_path)
            logger.info("\\n✅ Prediction complete!")

        else:
            raise ValueError(
                f"Unknown mode: {self.mode}. "
                f"Must be 'train', 'eval', 'test', or 'predict'."
            )


__all__ = ["OCRProjectOrchestrator"]
