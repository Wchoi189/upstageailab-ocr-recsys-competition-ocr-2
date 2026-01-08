from __future__ import annotations

"""Shared dependency management for inference utilities."""

import logging
import sys

# Import PROJECT_ROOT from central path utility (stable, works from any location)
from ocr.core.utils.path_utils import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)

# Ensure project root is in sys.path (for imports)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Lazy check for availability
import importlib.util

_torch_spec = importlib.util.find_spec("torch")
OCR_MODULES_AVAILABLE = _torch_spec is not None

if not OCR_MODULES_AVAILABLE:
    LOGGER.warning("Could not find 'torch' module. Falling back to mock predictions.")

# NOTE: Removed eager imports of torch, torchvision, lightning, etc.
# Consumers must import these locally inside methods/functions.

__all__ = [
    "LOGGER",
    "PROJECT_ROOT",
    "OCR_MODULES_AVAILABLE",
]
