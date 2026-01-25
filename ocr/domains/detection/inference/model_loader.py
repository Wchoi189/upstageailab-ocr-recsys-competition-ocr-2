from __future__ import annotations

"""Model loading utilities for OCR inference."""

import logging
from pathlib import Path
from typing import Any

from .dependencies import OCR_MODULES_AVAILABLE

# from .dependencies import ListConfig, get_model_by_cfg, torch

LOGGER = logging.getLogger(__name__)


def instantiate_model(model_config: Any):
    """Instantiate the OCR model from configuration."""
    if not OCR_MODULES_AVAILABLE:
        raise RuntimeError("OCR model modules are not available. Did you install training dependencies?")

    from ocr.core.models import get_model_by_cfg

    if get_model_by_cfg is None:
        raise RuntimeError("get_model_by_cfg not found!")

    architecture_name = "custom"
    if hasattr(model_config, "get") and callable(model_config.get):
        architecture_name = model_config.get("architecture_name", architecture_name)
    elif hasattr(model_config, "architecture_name"):
        architecture_name = model_config.architecture_name

    LOGGER.info("Instantiating model with architecture: %s", architecture_name)
    return get_model_by_cfg(model_config)


def load_checkpoint(checkpoint_path: str | Path, device: str) -> dict[str, Any] | None:
    """Load a checkpoint file into memory."""
    import torch

    if torch is None:
        return None

    register_safe_globals()
    path = Path(checkpoint_path)
    try:
        return torch.load(path, map_location=device)
    except TypeError:
        pass
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Initial torch.load failed for %s: %s", path, exc)

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Unable to load checkpoint %s: %s", path, exc)
        return None


def load_state_dict(model, checkpoint: dict[str, Any]) -> bool:  # type: ignore[override]
    """Load a model state dict from checkpoint data."""
    import torch

    if torch is None:
        return False

    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

    model_state = model.state_dict()
    filtered_state = {}
    prefixes = ("model._orig_mod.", "model.")
    original_to_new: dict[str, str] = {}
    for name, value in state_dict.items():
        new_name = name
        for prefix in prefixes:
            if new_name.startswith(prefix):
                new_name = new_name[len(prefix) :]
                break
        original_to_new[name] = new_name
        if new_name in model_state:
            filtered_state[new_name] = value

    dropped = {orig for orig, renamed in original_to_new.items() if renamed not in filtered_state}
    missing = set(model_state) - set(filtered_state)

    if dropped:
        LOGGER.warning("Dropped %d keys not present in current model: %s", len(dropped), sorted(dropped)[:10])
    
    # Load with strict=True to catch architecture mismatches
    try:
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=True)
        
        if missing_keys:
            raise RuntimeError(
                f"Checkpoint missing {len(missing_keys)} required keys: {sorted(missing_keys)[:10]}\\n"
                f"This checkpoint is incompatible with current model architecture.\\n"
                f"Convert it using scripts/checkpoints/convert_legacy_checkpoints.py"
            )
        
        if unexpected_keys:
            LOGGER.warning(f"Checkpoint has {len(unexpected_keys)} unexpected keys (will be ignored): {sorted(unexpected_keys)[:10]}")
        
        return True
        
    except RuntimeError as exc:
        if "size mismatch" in str(exc) or "Missing key(s)" in str(exc):
            LOGGER.error("Model architecture mismatch detected: %s", exc)
            return False
        raise


def register_safe_globals() -> None:
    import torch
    from omegaconf import ListConfig

    if torch is None or ListConfig is None:
        return

    try:
        from torch.serialization import add_safe_globals
    except (ImportError, AttributeError):  # pragma: no cover - torch internals
        return

    try:
        add_safe_globals([ListConfig])
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Could not register ListConfig as a safe global: %s", exc)
