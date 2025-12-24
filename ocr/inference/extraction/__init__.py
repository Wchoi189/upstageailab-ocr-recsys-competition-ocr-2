"""Receipt data extraction module for OCR pipeline.

This module provides functionality for extracting structured data
from receipt images, including store info, items, and totals.
"""

from .receipt_schema import (
    LineItem,
    ReceiptData,
    ReceiptMetadata,
)
from .field_extractor import (
    ReceiptFieldExtractor,
    ExtractorConfig,
)
from .normalizers import (
    normalize_currency,
    normalize_date,
    normalize_phone,
    normalize_time,
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
