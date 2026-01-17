# ocr/core/utils/wandb_base.py
# Shared WandB infrastructure (domain-agnostic)
# Extracted from wandb_utils.py during Phase 2 surgical refactor

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from omegaconf import DictConfig


def _get_wandb():
    """Return the wandb module for use in callbacks.

    This helper provides a consistent interface for accessing wandb
    in contexts where it may or may not be initialized.
    """
    import wandb

    return wandb


def load_env_variables():
    """Load environment variables from .env/.env.local if present."""

    def _load_file(env_file: Path) -> None:
        try:
            with env_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key:
                        os.environ[key] = value
                        if key == "WANDB_API_KEY" and value:
                            try:
                                import wandb

                                wandb.login(key=value)
                            except Exception as exc:
                                print(f"Warning: Failed to login to WandB: {exc}")
        except Exception as exc:
            print(f"Warning: Failed to load {env_file}: {exc}")

    for candidate in (Path(".env.local"), Path(".env")):
        if candidate.exists():
            _load_file(candidate)


_NAME_SANITIZE_PATTERN = re.compile(r"[^0-9a-zA-Z]+")
_MAX_RUN_NAME_LENGTH = 120

# Performance optimization: Cache registry to avoid repeated imports
_ARCHITECTURE_REGISTRY_CACHE = None


def _select(config: Any, path: Sequence[str], default: Any | None = None) -> Any | None:
    """Safely retrieve a nested value from a DictConfig or mapping."""
    from omegaconf import DictConfig

    current = config
    for key in path:
        if current is None:
            return default
        if isinstance(current, DictConfig):
            if key in current:
                current = current.get(key)
            elif hasattr(current, key):
                current = getattr(current, key)
            else:
                return default
        elif isinstance(current, dict):
            if key in current:
                current = current[key]
            else:
                return default
        else:
            current = getattr(current, key, default)
    return current if current is not None else default


def _sanitize_token(value: Any | None) -> str:
    """Convert a value into a lowercase token composed of safe characters."""

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    sanitized = _NAME_SANITIZE_PATTERN.sub("-", text)
    sanitized = sanitized.strip("-")
    return sanitized.lower()


def _format_lr_token(value: Any | None) -> str:
    """Format a learning rate value into a compact token."""

    if value is None:
        return ""
    try:
        lr_value = float(value)
    except (TypeError, ValueError):
        return ""
    if lr_value == 0:
        return "lr0"
    if abs(lr_value) >= 1e-2:
        lr_str = f"{lr_value:g}"
    else:
        lr_str = f"{lr_value:.0e}".replace("e-0", "e-").replace("e+0", "e")
    lr_token = _sanitize_token(lr_str)
    return f"lr{lr_token}" if lr_token else ""


def _format_batch_token(value: Any | None) -> str:
    """Format batch size information into a token."""

    if value is None:
        return ""
    try:
        batch_int = int(value)
    except (TypeError, ValueError):
        batch_int = None
    if batch_int is not None and batch_int > 0:
        return f"bs{batch_int}"
    batch_token = _sanitize_token(value)
    return f"bs{batch_token}" if batch_token else ""


def _deduplicate_labeled_tokens(tokens: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for label, token in tokens:
        if token and token not in seen:
            seen.add(token)
            result.append((label, token))
    return result


def _to_u8_bgr(image: Any) -> tuple[Any, bool]:
    """Convert image to BGR uint8 format for OpenCV drawing.

    Consolidates conversion logic for tensors, PIL images, and numpy arrays.
    Per visualization-logic.yaml: OpenCV uses BGR, WandB expects RGB.

    Args:
        image: Input image (torch.Tensor, PIL.Image, or np.ndarray)

    Returns:
        tuple: (bgr_image as np.ndarray, needs_rgb_conversion bool)
            - bgr_image: uint8 BGR image ready for cv2 drawing
            - needs_rgb_conversion: True if input was RGB and needs conversion back
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image as PILImage

    needs_rgb_conversion = False

    if torch.is_tensor(image):
        # Tensor expected as CHW normalized with mean=std=0.5
        arr = image.detach().cpu().float().numpy()  # C,H,W
        if arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))  # H,W,C
        # Un-normalize from (-1,1) back to (0,1)
        arr = arr * 0.5 + 0.5
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        # Tensor images are typically RGB, convert to BGR for OpenCV
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            needs_rgb_conversion = True

    elif isinstance(image, PILImage.Image):
        # PIL Images are in RGB format
        arr = np.array(image)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            needs_rgb_conversion = True
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    elif isinstance(image, np.ndarray):
        arr = image.copy()
        # Numpy arrays may be RGB or BGR depending on source
        # We assume they're already in the correct format
        if arr.dtype != np.uint8:
            maxv = float(arr.max()) if arr.size > 0 else 1.0
            if maxv <= 1.5:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    else:
        # Fallback: best-effort conversion
        arr = np.array(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    return np.ascontiguousarray(arr), needs_rgb_conversion


def _crop_to_content(image: np.ndarray, threshold: int = 4) -> np.ndarray:
    """Crop image to content, removing black borders.

    BUG-20251116-001: This function may crop too aggressively if images are incorrectly
    denormalized and appear dark. Ensure proper denormalization before calling this.
    """
    import numpy as np

    if image.size == 0:
        return image

    if image.ndim == 2:
        mask = image > threshold
    else:
        mask = np.max(image, axis=2) > threshold

    if not np.any(mask):
        # BUG-20251116-001: If no content found, return original image to avoid cropping everything
        return image

    coords = np.argwhere(mask)
    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0) + 1
    return image[top:bottom, left:right]


def _architecture_default_component(architecture_name: str | None, component: str) -> str:
    """Get default component from architecture registry (cached for performance).

    Performance: Caches registry to avoid repeated imports during run name generation.
    Called 4x per run (encoder/decoder/head/loss), so caching saves ~4 import cycles.
    """
    global _ARCHITECTURE_REGISTRY_CACHE

    if not architecture_name:
        return ""

    # Lazy-load and cache registry on first use
    if _ARCHITECTURE_REGISTRY_CACHE is None:
        try:
            from ocr.core import registry
            _ARCHITECTURE_REGISTRY_CACHE = registry
        except Exception:
            return ""

    try:
        mapping = _ARCHITECTURE_REGISTRY_CACHE.get_architecture(str(architecture_name))
    except Exception:
        return ""

    return _sanitize_token(mapping.get(component)) if mapping and component in mapping else ""


def _extract_component_token(
    model_cfg: Any,
    component: str,
    architecture_name: str | None,
    preference_keys: Sequence[str],
    fallback_keys: Sequence[str] = (),
) -> str:
    """Collect a representative token for a model component."""

    overrides = _select(model_cfg, ("component_overrides", component))

    def _pick_token(config_section: Any) -> str:
        if config_section is None:
            return ""
        for key in preference_keys:
            value = _select(config_section, (key,))
            if value is not None:
                token = _sanitize_token(value)
                if token:
                    return token
        for key in fallback_keys:
            value = _select(config_section, (key,))
            if value is not None:
                token = _sanitize_token(value)
                if token:
                    return token
        value = _select(config_section, ("_target_",))
        if value:
            return _sanitize_token(str(value).split(".")[-1])
        return ""

    component_cfg = _select(model_cfg, (component,))
    candidate = _pick_token(component_cfg)
    if not candidate:
        candidate = _pick_token(overrides)
    if not candidate:
        candidate = _architecture_default_component(architecture_name, component)
    return candidate


def generate_run_name(cfg: DictConfig) -> str:
    """Generate a descriptive, stable run name summarizing key components."""
    # Handle case where wandb config might not exist
    if hasattr(cfg, "wandb") and isinstance(cfg.wandb, bool):
        wandb_config: dict = {}
    elif hasattr(cfg, "wandb"):
        wandb_config = cfg.wandb or {}
    else:
        wandb_config = {}

    # Get user prefix from environment variable or config
    user_prefix = _sanitize_token(os.environ.get("WANDB_USER", wandb_config.get("user_prefix", "user"))) or "user"

    # Prefer wandb.experiment_tag; fall back to top-level or data tag
    tag = wandb_config.get("experiment_tag")
    if hasattr(cfg, "experiment_tag"):
        tag = tag or cfg.experiment_tag
    tag_token = _sanitize_token(tag)[:40] if tag else ""

    model_cfg = getattr(cfg, "model", None)
    architecture_name = None
    if model_cfg is not None:
        architecture_name = _select(model_cfg, ("architecture_name",))
    architecture_token = _sanitize_token(architecture_name)

    encoder_token = ""
    decoder_token = ""
    head_token = ""
    loss_token = ""
    if model_cfg is not None:
        encoder_token = _extract_component_token(
            model_cfg,
            "encoder",
            architecture_name,
            ("name", "model_name", "backbone", "type"),
            ("variant",),
        )
        decoder_token = _extract_component_token(
            model_cfg,
            "decoder",
            architecture_name,
            ("name", "decoder_name", "type", "variant"),
        )
        head_token = _extract_component_token(
            model_cfg,
            "head",
            architecture_name,
            ("name", "head_name", "type"),
        )
        loss_token = _extract_component_token(
            model_cfg,
            "loss",
            architecture_name,
            ("name", "loss_name", "type"),
        )

    batch_size = None
    if hasattr(cfg, "dataloaders") and hasattr(cfg.dataloaders, "train_dataloader"):
        batch_size = _select(cfg.dataloaders, ("train_dataloader", "batch_size"))
    if batch_size is None and hasattr(cfg, "data"):
        batch_size = _select(cfg, ("data", "batch_size"))

    batch_token = _format_batch_token(batch_size)

    lr_value = None
    if model_cfg is not None:
        lr_value = _select(model_cfg, ("optimizer", "lr"))
    lr_token = _format_lr_token(lr_value)

    hyper_tokens = [token for token in (batch_token, lr_token) if token]

    prefix_parts = [user_prefix]
    if tag_token:
        prefix_parts.append(tag_token)
    prefix = "_".join(filter(None, prefix_parts))

    component_tokens = _deduplicate_labeled_tokens(
        [
            ("arch", architecture_token),
            ("encoder", encoder_token),
            ("decoder", decoder_token),
            ("head", head_token),
            ("loss", loss_token),
        ]
    )

    labeled_tokens: list[tuple[str, str]] = list(component_tokens)

    for token in hyper_tokens:
        if token.startswith("bs"):
            labeled_tokens.append(("batch", token))
        elif token.startswith("lr"):
            labeled_tokens.append(("lr", token))

    descriptor = "-".join(token for _, token in labeled_tokens)
    name_without_suffix = prefix
    if descriptor:
        name_without_suffix = f"{prefix}_{descriptor}" if prefix else descriptor

    suffix = "_SCORE_PLACEHOLDER"
    run_name = f"{name_without_suffix}{suffix}"

    removal_priority = ["loss", "head", "lr", "batch", "decoder", "encoder", "arch"]

    def _rebuild(tokens: list[tuple[str, str]]) -> str:
        descriptor_part = "-".join(token for _, token in tokens if token)
        body = prefix
        if descriptor_part:
            body = f"{prefix}_{descriptor_part}" if prefix else descriptor_part
        return f"{body}{suffix}"

    while len(run_name) > _MAX_RUN_NAME_LENGTH:
        removed = False
        for label in removal_priority:
            for idx, (tok_label, _) in enumerate(labeled_tokens):
                if tok_label == label:
                    labeled_tokens.pop(idx)
                    run_name = _rebuild(labeled_tokens)
                    removed = True
                    break
            if removed:
                break
        if not removed:
            descriptor_fallback = "-".join(token for _, token in labeled_tokens)
            digest_source = descriptor_fallback or prefix or "run"
            digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:8]
            core_tokens = [token for _, token in labeled_tokens[:2]]
            core_tokens.append(digest)
            descriptor_part = "-".join(token for token in core_tokens if token)
            body = prefix
            if descriptor_part:
                body = f"{prefix}_{descriptor_part}" if prefix else descriptor_part
            run_name = f"{body}{suffix}"
            break

    return run_name


def finalize_run(metrics: Mapping[str, float] | float | None):
    """Finalize the active W&B run with the most relevant metric."""
    import wandb

    if not wandb.run:
        print("W&B run not initialized. Skipping finalization.")
        return

    metric_map: dict[str, float] = {}

    if isinstance(metrics, Mapping):
        for key, value in metrics.items():
            try:
                metric_map[key] = float(value)
            except (TypeError, ValueError):
                continue
    elif metrics is not None:
        try:
            metric_map["val/loss"] = float(metrics)
        except (TypeError, ValueError):
            metric_map["val/loss"] = 0.0

    final_loss = metric_map.get("val/loss")
    if final_loss is None:
        final_loss = metric_map.get("loss")
    if final_loss is None:
        final_loss = metric_map.get("train/loss")
    if final_loss is None:
        final_loss = 0.0

    metric_map.setdefault("final_mean_loss", final_loss)

    # Update summary metrics
    for key, value in metric_map.items():
        if math.isfinite(value):
            wandb.summary[key] = value

    rename_preferences: list[tuple[str, str, int]] = [
        ("test/hmean", "hmean", 3),
        ("val/hmean", "hmean", 3),
        ("test/recall", "recall", 3),
        ("val/recall", "recall", 3),
        ("test/precision", "precision", 3),
        ("val/precision", "precision", 3),
        ("val/loss", "loss", 4),
        ("test/loss", "loss", 4),
        ("final_mean_loss", "loss", 4),
    ]

    metric_label = "loss"
    metric_value = final_loss
    precision = 4

    for key, label, prec in rename_preferences:
        value = metric_map.get(key)
        if value is None or not math.isfinite(value):
            continue
        metric_label = label
        metric_value = value
        precision = prec
        break

    formatted_score = f"{metric_label}{metric_value:.{precision}f}"

    current_name = wandb.run.name or "run_SCORE_PLACEHOLDER"
    if "_SCORE_PLACEHOLDER" in current_name:
        final_name = current_name.replace("_SCORE_PLACEHOLDER", f"_{formatted_score}")
    else:
        final_name = f"{current_name}_{formatted_score}"

    wandb.run.name = final_name
    wandb.summary["final_run_name"] = final_name

    print(f"Finalized run name: {final_name}")

    wandb.finish()
