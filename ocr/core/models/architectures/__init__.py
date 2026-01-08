"""OCR architecture implementations and registrations."""

from ocr.features.detection.models.architectures import craft, dbnet, dbnetpp  # noqa: F401
from ocr.features.recognition.models import architecture as recognition_arch  # noqa: F401
from . import shared_decoders  # noqa: F401

__all__ = ["dbnet", "craft", "dbnetpp", "shared_decoders", "recognition_arch"]
