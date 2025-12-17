"""Model lifecycle management for OCR inference.

This module consolidates model loading, caching, and cleanup logic into a
focused manager class. It handles:
1. Model instantiation from config
2. Checkpoint loading and state dict management
3. Model caching (avoid reloading same checkpoint)
4. Device management (CPU/GPU)
5. Cleanup and resource management

The manager provides a clean API for model lifecycle operations and ensures
proper resource cleanup.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ocr.utils.path_utils import get_path_resolver

from .config_loader import ModelConfigBundle, load_model_config, resolve_config_path
from .dependencies import OCR_MODULES_AVAILABLE, torch
from .model_loader import instantiate_model, load_checkpoint, load_state_dict

LOGGER = logging.getLogger(__name__)


class ModelManager:
    """Manages model lifecycle for OCR inference.

    This class handles all aspects of model management:
    - Loading models from checkpoints
    - Caching loaded models (avoid redundant loads)
    - Managing model state (train/eval mode)
    - Device placement (CPU/GPU)
    - Resource cleanup

    The manager maintains a single loaded model at a time and provides
    efficient caching to avoid reloading the same checkpoint.
    """

    def __init__(self, device: str | None = None):
        """Initialize model manager.

        Args:
            device: Device for model inference ("cuda" or "cpu").
                   Auto-detects if not specified.
        """
        if device is None:
            self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Any | None = None
        self.config: Any | None = None
        self._current_checkpoint_path: str | None = None
        self._config_bundle: ModelConfigBundle | None = None

        LOGGER.info(f"ModelManager initialized (device: {self.device})")

    def load_model(
        self,
        checkpoint_path: str,
        config_path: str | None = None,
    ) -> bool:
        """Load model from checkpoint.

        Implements intelligent caching: if the requested checkpoint is already
        loaded, reuses the cached model instead of reloading.

        Args:
            checkpoint_path: Path to model checkpoint file
            config_path: Optional path to config file (auto-detected if not provided)

        Returns:
            True if model loaded successfully, False otherwise

        Example:
            >>> manager = ModelManager()
            >>> success = manager.load_model("checkpoints/model.pth")
            >>> if success:
            ...     print("Model ready for inference")
        """
        if not OCR_MODULES_AVAILABLE:
            LOGGER.error("OCR modules not installed. Cannot load model.")
            return False

        total_start = time.perf_counter()
        normalized_path = str(Path(checkpoint_path).resolve())

        # Check cache: reuse already loaded model
        if self.model is not None and self._current_checkpoint_path == normalized_path:
            LOGGER.info(f"✅ Checkpoint already loaded; reusing cached model: {checkpoint_path}")
            return True

        # Resolve config file
        resolver = get_path_resolver()
        search_dirs = (resolver.config.config_dir,)
        resolved_config = resolve_config_path(checkpoint_path, config_path, search_dirs)

        if resolved_config is None:
            LOGGER.error(f"Could not find valid config file for checkpoint: {checkpoint_path}")
            return False

        # Load config bundle
        LOGGER.info(f"Loading model from checkpoint: {checkpoint_path}")
        bundle = load_model_config(resolved_config)
        self._config_bundle = bundle

        # Extract model config
        model_config = self._extract_model_config(bundle, resolved_config)
        if model_config is None:
            return False

        # Instantiate model
        try:
            model = instantiate_model(model_config)
        except Exception:
            LOGGER.exception(f"Failed to instantiate model from config {resolved_config}")
            return False

        # Load checkpoint weights
        load_start = time.perf_counter()
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        load_duration = time.perf_counter() - load_start
        LOGGER.info(f"Checkpoint weights loaded in {load_duration:.2f}s")

        if checkpoint is None:
            LOGGER.error(f"Failed to load checkpoint {checkpoint_path}")
            return False

        # Load state dict into model
        if not load_state_dict(model, checkpoint):
            LOGGER.error(f"Failed to load state dictionary for checkpoint {checkpoint_path}")
            return False

        # Setup model
        self.model = model.to(self.device)
        self.model.eval()
        self.config = bundle.raw_config
        self._current_checkpoint_path = normalized_path

        LOGGER.info(
            f"Model ready | checkpoint={checkpoint_path} | device={self.device} | "
            f"total_load={time.perf_counter() - total_start:.2f}s"
        )
        return True

    def get_config_bundle(self) -> ModelConfigBundle | None:
        """Get current model configuration bundle.

        Returns:
            ModelConfigBundle if model is loaded, None otherwise
        """
        return self._config_bundle

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded.

        Returns:
            True if model is loaded and ready for inference
        """
        return self.model is not None

    def get_current_checkpoint(self) -> str | None:
        """Get path to currently loaded checkpoint.

        Returns:
            Checkpoint path if model is loaded, None otherwise
        """
        return self._current_checkpoint_path

    def cleanup(self) -> None:
        """Clean up model and free resources.

        Unloads the model and clears all cached state. Should be called
        when the model is no longer needed to free GPU/CPU memory.
        """
        LOGGER.info("Cleaning up ModelManager resources...")

        if self.model is not None and torch is not None:
            # Move model to CPU before deletion to free GPU memory
            self.model = self.model.cpu()
            del self.model
            self.model = None

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Clear state
        self.config = None
        self._config_bundle = None
        self._current_checkpoint_path = None

        LOGGER.info("ModelManager cleanup completed")

    @staticmethod
    def _extract_model_config(bundle: ModelConfigBundle, config_path: str) -> Any | None:
        """Extract model configuration from bundle.

        Handles both standard configs (with 'model' section) and Hydra configs
        (with model attributes at root level).

        Args:
            bundle: Configuration bundle
            config_path: Path to config file (for error messages)

        Returns:
            Model config object, or None if extraction fails
        """
        model_config = getattr(bundle.raw_config, "model", None)

        if model_config is None:
            # Fallback: Try root level extraction (for Hydra configs)
            LOGGER.warning(
                f"Configuration missing direct 'model' section, "
                f"trying root level extraction: {config_path}"
            )
            model_config = bundle.raw_config

            # Verify this looks like a model config
            model_attrs = ["architecture", "encoder", "decoder", "head"]
            if not any(hasattr(model_config, attr) for attr in model_attrs):
                LOGGER.error(
                    f"Configuration has neither 'model' section nor model attributes "
                    f"at root level: {config_path}"
                )
                return None

        return model_config

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False


__all__ = [
    "ModelManager",
]
