"""Registration of decoder components shared across architectures."""

from ocr.core import registry
from ocr.features.detection.models.decoders.fpn_decoder import FPNDecoder
from ocr.core.models.decoder.pan_decoder import PANDecoder


def register_shared_decoders() -> None:
    """Register decoder implementations that can be used across architectures."""

    registry.register_decoder("fpn_decoder", FPNDecoder)
    registry.register_decoder("pan_decoder", PANDecoder)


register_shared_decoders()
