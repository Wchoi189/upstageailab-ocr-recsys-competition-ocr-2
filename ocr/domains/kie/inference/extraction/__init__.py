"""Receipt data extraction module for OCR pipeline.

This module provides functionality for extracting structured data
from receipt images, including store info, items, and totals.
"""

from .field_extractor import (
    ExtractorConfig,
    ReceiptFieldExtractor,
)
from .normalizers import (
    normalize_currency,
    normalize_date,
    normalize_phone,
    normalize_time,
)
from .receipt_schema import (
    LineItem,
    ReceiptData,
    ReceiptMetadata,
)

__all__ = [
    # Schema
    "LineItem",
    "ReceiptData",
    "ReceiptMetadata",
    # Extractor
    "ReceiptFieldExtractor",
    "ExtractorConfig",
    # Normalizers
    "normalize_currency",
    "normalize_date",
    "normalize_phone",
    "normalize_time",
]
