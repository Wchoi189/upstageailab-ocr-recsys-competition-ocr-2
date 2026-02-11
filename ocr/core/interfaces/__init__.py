"""Core Interface Layer - Domain-agnostic data contracts."""

from ocr.core.interfaces.decoder import AutoregressiveDecoder, DecoderConfig, DecoderMode, DecoderOutput
from ocr.core.interfaces.flash_constraints import FlashAttentionConfig, create_flash_config
from ocr.core.interfaces.plm import AttentionMasks, PLMConfig, PLMLossConfig, PLMModule
from ocr.core.interfaces.schemas import Box, DetectionResult, PageResult, RecognitionResult

__all__ = [
    # Schemas
    "Box",
    "DetectionResult",
    "RecognitionResult",
    "PageResult",
    # Decoder Interfaces (NEW)
    "AutoregressiveDecoder",
    "DecoderMode",
    "DecoderOutput",
    "DecoderConfig",
    # PLM Interfaces (NEW)
    "PLMModule",
    "PLMConfig",
    "PLMLossConfig",
    "AttentionMasks",
    # Flash Attention (NEW)
    "FlashAttentionConfig",
    "create_flash_config",
]
