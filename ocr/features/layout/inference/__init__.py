"""Layout detection module for OCR pipeline.

This module provides functionality for grouping detected text regions
into hierarchical structures (lines, paragraphs, tables).
"""

from .contracts import (
    BoundingBox,
    LayoutResult,
    TextBlock,
    TextElement,
    TextLine,
)
from .grouper import LineGrouper, LineGrouperConfig

__all__ = [
    "BoundingBox",
    "TextElement",
    "TextLine",
    "TextBlock",
    "LayoutResult",
    "LineGrouper",
    "LineGrouperConfig",
]
