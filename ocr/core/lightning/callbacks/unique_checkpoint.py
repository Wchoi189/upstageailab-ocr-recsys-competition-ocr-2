from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from ocr.core.utils.config_utils import ensure_dict, is_config




class UniqueModelCheckpoint(ModelCheckpoint):
    """
    Enhanced ModelCheckpoint with index-based directory organization.

    Implements the naming scheme for index-based structure:
    outputs/<index>/checkpoints/<type>-<epoch>_<step>_<metric>.ckpt

    This callback extends PyTorch Lightning's ModelCheckpoint to provide:
    - Index-based directory organization (managed by Hydra)
    - Clear checkpoint naming with model information
    - Automatic metadata generation alongside checkpoints
    - Prevention of overwrites through unique identifiers
    - Improved searchability and filtering (easier to sort by date)

    Example checkpoint names:
    - Epoch checkpoint: epoch-03_step-000103.ckpt
    - Last checkpoint: last.ckpt
    - Best checkpoint: best-hmean-0.8920.ckpt
    """

    def __init__(
        self,
        *args,
        add_timestamp: bool = True,
        experiment_tag: str | None = None,
        training_phase: str = "training",
        config: dict | Any | None = None,
        **kwargs,
    ):
        """
        Initialize checkpoint callback with index-based organization.

        Args:
            add_timestamp: Whether to add timestamp to checkpoint filenames (legacy, kept for compatibility)
            experiment_tag: Optional experiment identifier (deprecated in favor of index-based structure)
            training_phase: Stage of the experiment (e.g., "training", "validation", "finetuning")
            config: Resolved training configuration to save alongside checkpoints
            **kwargs: Additional arguments passed to ModelCheckpoint
        """
        super().__init__(*args, **kwargs)
        self.add_timestamp = add_timestamp
        self.experiment_tag = experiment_tag
        self.training_phase = training_phase
        self._resolved_config = config

        # Generate unique identifier once at initialization (for legacy compatibility)
        if self.add_timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def format_checkpoint_name(
        self,
        metrics: dict | None = None,
        filename: str | None = None,
        ver: int | None = None,
        prefix: str | None = None,
    ) -> str:
        """
        Formats checkpoint name using hierarchical naming convention.

        Implements the scheme:
        - Epoch checkpoint: epoch-<epoch>_step-<step>.ckpt
        - Last checkpoint: last.ckpt
        - Best checkpoint: best-<metric_name>-<value>.ckpt

        Args:
            metrics: Dictionary of metric values
            filename: Template filename from config (used for determining checkpoint type)
            ver: Version number (optional)
            prefix: Additional prefix (optional)

        Returns:
            Formatted checkpoint filename path
        """
        # Trainer might not be attached during initialization
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return super().format_checkpoint_name(metrics or {}, filename, ver, prefix)

        # 1. Get authoritative epoch and step directly from the trainer
        epoch = trainer.current_epoch
        step = trainer.global_step

        # 2. Determine checkpoint type and build the filename
        is_best_checkpoint = filename and "best" in filename.lower()
        is_last_checkpoint = filename and filename.lower() == "last"

        if is_last_checkpoint:
            # Last checkpoint: simply "last.ckpt"
            stem = "last"
        elif is_best_checkpoint:
            # Best checkpoint: best-<metric_name>-<value>.ckpt
            stem = "best"
            if self.monitor and metrics:
                metric_val = metrics.get(self.monitor)
                if isinstance(metric_val, torch.Tensor):
                    # Clean up the metric name (e.g., "val/hmean" -> "hmean")
                    metric_name_clean = self.monitor.split("/")[-1]
                    stem = f"best-{metric_name_clean}-{metric_val.item():.4f}"
        else:
            # Epoch checkpoint: epoch-<epoch>_step-<step>.ckpt
            stem = f"epoch-{epoch:02d}_step-{step:06d}"

            # Optionally add metric value for epoch checkpoints
            if self.auto_insert_metric_name and metrics and self.monitor:
                metric_val = metrics.get(self.monitor)
                if isinstance(metric_val, torch.Tensor):
                    metric_name_clean = self.monitor.split("/")[-1]
                    stem = f"{stem}_{metric_name_clean}-{metric_val.item():.4f}"

        # 3. Add prefix if provided
        if prefix:
            stem = f"{prefix}_{stem}"

        # 4. Add version if provided
        if ver is not None:
            stem = f"{stem}_v{ver}"

        # 5. Combine and return the final path
        dirpath = self.dirpath or "."
        final_name = f"{stem}{self.FILE_EXTENSION}"
        return os.path.join(dirpath, final_name)

    def _get_model_info(self) -> dict[str, str | None]:
        """
        Extract model architecture and encoder information.

        Returns:
            Dictionary with 'architecture' and 'encoder' keys
        """
        info: dict[str, str | None] = {"architecture": None, "encoder": None}

        try:
            trainer = getattr(self, "trainer", None)
            if trainer is not None and hasattr(trainer, "model"):
                model = trainer.model

                # Extract architecture
                if hasattr(model, "architecture_name"):
                    info["architecture"] = model.architecture_name
                else:
                    info["architecture"] = getattr(model, "_architecture_name", None)

                # Extract encoder information
                if hasattr(model, "encoder") and hasattr(model.encoder, "model_name"):
                    info["encoder"] = model.encoder.model_name
                elif hasattr(model, "component_overrides") and "encoder" in model.component_overrides:
                    encoder_override = model.component_overrides["encoder"]
                    if is_config(encoder_override) and "model_name" in encoder_override:
                        info["encoder"] = encoder_override["model_name"]

        except Exception:
            # If anything fails, return None values
            pass

        return info

    def _generate_checkpoint_metadata(self, checkpoint_path: str, metrics: dict | None = None) -> None:
        """
        Generate metadata file alongside checkpoint using defined schema.

        Creates a JSON file with checkpoint metadata including:
        - Model architecture information (encoder, decoder, head)
        - Training configuration and hyperparameters
        - Checkpoint metrics and performance
        - Training progress (epoch, step)
        - Timestamp and environment information

        Args:
            checkpoint_path: Path to the checkpoint file
            metrics: Current training metrics
        """
        import json
        from pathlib import Path

        try:
            trainer = getattr(self, "trainer", None)
            if trainer is None:
                return

            def _json_ready(value):
                try:
                    import numpy as np  # type: ignore
                except Exception:  # pragma: no cover - optional dependency
                    np = None  # type: ignore[assignment]

                try:
                    from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore

                    omegaconf_types = (DictConfig, ListConfig)
                except Exception:  # pragma: no cover - optional dependency
                    OmegaConf = None  # type: ignore
                    omegaconf_types = ()

                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu()
                    if value.numel() == 1:
                        return value.item()
                    return value.tolist()

                if np is not None and isinstance(value, np.ndarray):  # type: ignore[attr-defined]
                    if value.size == 1:
                        return value.item()
                    return value.tolist()

                if omegaconf_types and isinstance(value, omegaconf_types):
                    try:
                        return _json_ready(OmegaConf.to_container(value, resolve=True))  # type: ignore[arg-type,union-attr]
                    except Exception:
                        return None

                if is_config(value):
                    return {str(k): _json_ready(v) for k, v in value.items()}

                if isinstance(value, list | tuple | set):
                    return [_json_ready(item) for item in value]

                if hasattr(value, "item"):
                    try:
                        return value.item()
                    except Exception:
                        pass

                if isinstance(value, Path):
                    return str(value)

                return value

            # Get model information
            model_info_extracted = self._get_model_info()

            # Build metadata using schema
            from ocr.core.utils.checkpoints.types import (
                CheckpointingConfig,
                CheckpointMetadataV1,
                DecoderInfo,
                EncoderInfo,
                HeadInfo,
                LossInfo,
                MetricsInfo,
                ModelInfo,
                TrainingInfo,
            )

            # Get model component details if available
            components = {}
            if hasattr(trainer, "model"):
                model = trainer.model
                if hasattr(model, "component_overrides"):
                    components = model.component_overrides

            safe_metrics = {}
            if metrics:
                for key, value in metrics.items():
                    safe_metrics[str(key)] = _json_ready(value)

            safe_components = _json_ready(components)
            if not is_config(safe_components):
                safe_components = {}

            # Construct V1 models for the newer schema
            metadata = CheckpointMetadataV1(
                checkpoint_path=checkpoint_path,
                exp_name=getattr(trainer.datamodule, "exp_name", "unknown"),
                created_at=datetime.now().isoformat(),
                training=TrainingInfo(
                    epoch=trainer.current_epoch,
                    global_step=trainer.global_step,
                    training_phase=self.training_phase,
                ),
                model=ModelInfo(
                    architecture=model_info_extracted.get("architecture") or "unknown",
                    encoder=EncoderInfo(
                        model_name=model_info_extracted.get("encoder") or "unknown",
                    ),
                    decoder=DecoderInfo(name="unknown"),
                    head=HeadInfo(name="unknown"),
                    loss=LossInfo(name="unknown"),
                ),
                metrics=MetricsInfo(additional_metrics=safe_metrics),
                checkpointing=CheckpointingConfig(
                    monitor=self.monitor or "unknown",
                    mode=self.mode or "max",
                    save_top_k=self.save_top_k,
                ),
            )

            # Save metadata file
            metadata_path = Path(checkpoint_path).with_suffix(".metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata.model_dump(mode="json"), f, indent=2)

            # Save resolved config file for inference compatibility
            # This eliminates the need for complex config resolution in inference
            try:
                # Get the resolved config from trainer (passed via callback setup)
                resolved_config = getattr(self, "_resolved_config", None)
                if resolved_config is not None:

                    # Convert to plain dict for JSON serialization
                    config_dict = ensure_dict(resolved_config, resolve=True)

                    # Ensure we have a dict to work with
                    if not is_config(config_dict):
                        config_dict = {}

                    # Save only the model section if it exists, otherwise save the full config
                    model_config = config_dict.get("model")
                    if model_config is not None:
                        config_to_save = {"model": model_config}
                    else:
                        # For configs without explicit model section, save key components
                        config_to_save = {
                            key: value
                            for key, value in config_dict.items()
                            if isinstance(key, str) and key in ["model", "architecture", "encoder", "decoder", "head", "backbone"]
                        }

                    if config_to_save:  # Only save if we have something useful
                        config_path = Path(checkpoint_path).with_suffix(".config.json")
                        with open(config_path, "w") as f:
                            json.dump(_json_ready(config_to_save), f, indent=2)

            except Exception:
                # Don't fail training if config saving fails
                pass

        except Exception:
            # Don't fail training if metadata generation fails
            pass

    def _setup_dirpath(self) -> str:
        """
        Set up the directory path for index-based structure.

        With the new index-based organization, Hydra creates the directory structure:
        outputs/<index>/checkpoints/

        This method now simply ensures the directory exists and can optionally
        add model information to checkpoint filenames.

        Returns:
            The formatted directory path
        """
        if not self.dirpath:
            return "."

        # Convert dirpath to string if it's a Path object
        dirpath_str = str(self.dirpath)

        # For index-based structure, use the dirpath as-is
        # Hydra already creates: outputs/<index>/checkpoints/
        # No need to manipulate directory names with experiment tags

        return dirpath_str

    def setup(self, trainer, pl_module, stage: str | None = None) -> None:
        """
        Setup callback for index-based directory structure.

        With the new structure, directory setup is simplified since Hydra
        handles the index-based organization.
        """
        # Call parent setup with stage
        if stage is not None:
            super().setup(trainer, pl_module, stage)
        else:
            # Avoid the type issue by not passing None
            super().setup(trainer, pl_module, "fit")

        # For index-based structure, dirpath is already set by Hydra
        # No need to enhance directory naming

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load state dict but handle dirpath mismatches gracefully.

        During prediction, the dirpath may differ from training, which causes
        warnings. We update the dirpath in the state_dict to match current
        dirpath to avoid warnings.
        """
        # Update dirpath in state_dict to match current dirpath
        if "dirpath" in state_dict and hasattr(self, "dirpath"):
            if state_dict["dirpath"] != self.dirpath:
                state_dict = state_dict.copy()
                state_dict["dirpath"] = self.dirpath

        super().load_state_dict(state_dict)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Generate metadata file and log checkpoint directory to wandb when a checkpoint is saved."""
        # Generate metadata file alongside checkpoint
        checkpoint_paths = []
        if hasattr(self, "best_model_path") and self.best_model_path:
            checkpoint_paths.append(self.best_model_path)
        if hasattr(self, "last_model_path") and self.last_model_path:
            checkpoint_paths.append(self.last_model_path)

        # Get current metrics for metadata
        metrics = {}
        if hasattr(trainer, "logged_metrics"):
            metrics = dict(trainer.logged_metrics)

        # Generate metadata for each checkpoint path
        for checkpoint_path in checkpoint_paths:
            self._generate_checkpoint_metadata(checkpoint_path, metrics)

        # Log checkpoint directory to wandb
        from ocr.core.utils.wandb_base import _get_wandb

        wandb = _get_wandb()
        if wandb.run and self.dirpath:
            wandb.log({"checkpoint_dir": self.dirpath})

        return checkpoint

    def on_train_end(self, trainer, pl_module):
        """Generate metadata for all saved checkpoints at the end of training."""
        # Generate metadata for all saved checkpoints
        checkpoint_paths = []
        if hasattr(self, "best_model_path") and self.best_model_path:
            checkpoint_paths.append(self.best_model_path)
        if hasattr(self, "last_model_path") and self.last_model_path:
            checkpoint_paths.append(self.last_model_path)

        # Get current metrics for metadata
        metrics = {}
        if hasattr(trainer, "logged_metrics"):
            metrics = dict(trainer.logged_metrics)

        # Generate metadata for each checkpoint path
        for checkpoint_path in checkpoint_paths:
            self._generate_checkpoint_metadata(checkpoint_path, metrics)
