"""Layout detection feature for OCR pipeline.

This module provides layout analysis capabilities for grouping text elements
into lines and blocks based on spatial relationships.
"""

from ocr.domains.layout.inference.contracts import (
    BoundingBox,
    LayoutResult,
    TextBlock,
    TextElement,
    TextLine,
)
from ocr.domains.layout.inference.grouper import (
    LineGrouper,
    LineGrouperConfig,
    create_text_element,
)

__all__ = [
    # Contracts
    "BoundingBox",
    "LayoutResult",
    "TextBlock",
    "TextElement",
    "TextLine",
    # Grouper
    "LineGrouper",
    "LineGrouperConfig",
    "create_text_element",
]
