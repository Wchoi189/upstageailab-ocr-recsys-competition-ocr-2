"""OCR architecture implementations and registrations."""

from ocr.detection.models.architectures import craft, dbnet, dbnetpp  # noqa: F401
from . import shared_decoders  # noqa: F401

__all__ = ["dbnet", "craft", "dbnetpp", "shared_decoders"]
