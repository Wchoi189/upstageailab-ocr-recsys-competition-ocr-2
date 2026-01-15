"""Core abstract base classes and registry for OCR framework components."""

from .interfaces.losses import BaseLoss
from .interfaces.metrics import BaseMetric
from .interfaces.models import BaseDecoder, BaseEncoder, BaseHead
from .utils.registry import ComponentRegistry, get_registry, registry

__all__ = [
    "BaseEncoder",
    "BaseDecoder",
    "BaseHead",
    "BaseLoss",
    "BaseMetric",
    "ComponentRegistry",
    "get_registry",
    "registry",
]
