from __future__ import annotations

"""Shared dependency management for inference utilities."""

import logging
import sys

# Import PROJECT_ROOT from central path utility (stable, works from any location)
from ocr.utils.path_utils import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)

# Ensure project root is in sys.path (for imports)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import lightning.pytorch as pl  # type: ignore
    import torch
    import torchvision.transforms as transforms  # type: ignore
    import yaml  # type: ignore
    from omegaconf import DictConfig, ListConfig  # type: ignore

    from ocr.models import get_model_by_cfg  # type: ignore

    OCR_MODULES_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - optional dependency guard
    LOGGER.warning("Could not import OCR modules: %s. Falling back to mock predictions.", exc)
    OCR_MODULES_AVAILABLE = False
    torch = None  # type: ignore
    transforms = None  # type: ignore
    yaml = None  # type: ignore
    pl = None  # type: ignore
    DictConfig = dict  # type: ignore
    ListConfig = None  # type: ignore
    get_model_by_cfg = None  # type: ignore

__all__ = [
    "LOGGER",
    "PROJECT_ROOT",
    "OCR_MODULES_AVAILABLE",
    "torch",
    "transforms",
    "yaml",
    "pl",
    "DictConfig",
    "ListConfig",
    "get_model_by_cfg",
]
