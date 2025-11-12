"""Service layer for unified OCR app.

Services handle business logic and are independent of UI framework.
"""

from .config_loader import ConfigLoader, load_unified_config

__all__ = [
    "ConfigLoader",
    "load_unified_config",
]
