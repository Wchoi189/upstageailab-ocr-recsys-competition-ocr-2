"""Core abstract base classes and registry for OCR framework components."""

# Explicit imports required to avoid heavy dependency loading (Torch) at package level.
# Users should import directly from submodules:
# from ocr.core.interfaces.losses import BaseLoss
# from ocr.core.utils.registry import registry

__all__ = []
