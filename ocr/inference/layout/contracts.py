"""Pydantic contracts for layout detection.

This module defines the data structures for representing hierarchical
text layout: elements → lines → blocks → document.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BoundingBox(BaseModel):
    """Axis-aligned bounding box.

    Coordinates are in original image space (pixels).

    Attributes:
        x_min: Left edge coordinate
        y_min: Top edge coordinate
        x_max: Right edge coordinate
        y_max: Bottom edge coordinate
    """

    model_config = ConfigDict(frozen=True)

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @field_validator("x_max")
    @classmethod
    def x_max_greater_than_x_min(cls, v: float, info) -> float:
        """Ensure x_max >= x_min."""
        x_min = info.data.get("x_min", 0)
        if v < x_min:
            raise ValueError(f"x_max ({v}) must be >= x_min ({x_min})")
        return v

    @field_validator("y_max")
    @classmethod
    def y_max_greater_than_y_min(cls, v: float, info) -> float:
        """Ensure y_max >= y_min."""
        y_min = info.data.get("y_min", 0)
        if v < y_min:
            raise ValueError(f"y_max ({v}) must be >= y_min ({y_min})")
        return v

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        """Center point (x, y) of the bounding box."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
        )

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union with another bounding box.

        Args:
            other: Another BoundingBox

        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection
        x_min = max(self.x_min, other.x_min)
        y_min = max(self.y_min, other.y_min)
        x_max = min(self.x_max, other.x_max)
        y_max = min(self.y_max, other.y_max)

        if x_max < x_min or y_max < y_min:
            return 0.0

        intersection = (x_max - x_min) * (y_max - y_min)
        union = self.area + other.area - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def y_overlap_ratio(self, other: BoundingBox) -> float:
        """Calculate vertical overlap ratio.

        Returns the proportion of the smaller height that overlaps.

        Args:
            other: Another BoundingBox

        Returns:
            Overlap ratio between 0 and 1
        """
        overlap_min = max(self.y_min, other.y_min)
        overlap_max = min(self.y_max, other.y_max)

        if overlap_max <= overlap_min:
            return 0.0

        overlap_height = overlap_max - overlap_min
        min_height = min(self.height, other.height)

        if min_height <= 0:
            return 0.0

        return overlap_height / min_height


class TextElement(BaseModel):
    """Single text element with recognized content.

    Represents the smallest unit of text, typically a word or character group.

    Attributes:
        id: Unique identifier for this element
        polygon: List of [x, y] coordinates defining the text region
        bbox: Axis-aligned bounding box
        text: Recognized text content
        confidence: Recognition confidence (0-1)
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    polygon: list[list[float]]
    bbox: BoundingBox
    text: str
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, v: list[list[float]]) -> list[list[float]]:
        """Polygon must have at least 3 points."""
        if len(v) < 3:
            raise ValueError(f"Polygon must have at least 3 points, got {len(v)}")
        for point in v:
            if len(point) != 2:
                raise ValueError(f"Each polygon point must be [x, y], got {point}")
        return v


class TextLine(BaseModel):
    """Grouped text elements forming a line.

    A line is a horizontal sequence of text elements that should be
    read together from left to right.

    Attributes:
        id: Unique identifier for this line
        elements: List of TextElements in this line (sorted by x position)
        reading_order: Order of this line in the document (0-indexed)
        bbox: Axis-aligned bounding box enclosing all elements
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    elements: list[TextElement]
    reading_order: int = Field(ge=0)
    bbox: BoundingBox

    @property
    def text(self) -> str:
        """Concatenate all element texts with spaces."""
        return " ".join(el.text for el in self.elements)

    @property
    def confidence(self) -> float:
        """Average confidence of all elements."""
        if not self.elements:
            return 0.0
        return sum(el.confidence for el in self.elements) / len(self.elements)


BlockType = Literal["paragraph", "table_cell", "header", "list_item", "unknown"]


class TextBlock(BaseModel):
    """Paragraph or table cell containing lines.

    A block is a logical grouping of lines that form a coherent unit,
    such as a paragraph, table cell, or header.

    Attributes:
        id: Unique identifier for this block
        block_type: Type of block (paragraph, table_cell, header, list_item)
        lines: List of TextLines in this block (sorted by reading order)
        bbox: Axis-aligned bounding box enclosing all lines
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    block_type: BlockType = "paragraph"
    lines: list[TextLine]
    bbox: BoundingBox

    @property
    def text(self) -> str:
        """Concatenate all line texts with newlines."""
        return "\n".join(line.text for line in self.lines)

    @property
    def confidence(self) -> float:
        """Average confidence of all lines."""
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)


class LayoutResult(BaseModel):
    """Complete layout analysis result.

    The top-level result from layout analysis, containing all detected
    blocks with their hierarchical structure.

    Attributes:
        blocks: List of TextBlocks in reading order
        reading_order: List of block IDs in reading order
        tables: Optional list of structured table data (future use)
        raw_elements: Optional list of all elements before grouping
    """

    blocks: list[TextBlock] = Field(default_factory=list)
    reading_order: list[str] = Field(default_factory=list)
    tables: list[dict] | None = None
    raw_elements: list[TextElement] | None = None

    @property
    def text(self) -> str:
        """Concatenate all block texts with double newlines."""
        return "\n\n".join(block.text for block in self.blocks)

    @property
    def num_elements(self) -> int:
        """Total number of text elements across all blocks."""
        count = 0
        for block in self.blocks:
            for line in block.lines:
                count += len(line.elements)
        return count

    @property
    def num_lines(self) -> int:
        """Total number of text lines across all blocks."""
        return sum(len(block.lines) for block in self.blocks)

    def get_block_by_id(self, block_id: str) -> TextBlock | None:
        """Find a block by its ID."""
        for block in self.blocks:
            if block.id == block_id:
                return block
        return None


__all__ = [
    "BoundingBox",
    "TextElement",
    "TextLine",
    "TextBlock",
    "BlockType",
    "LayoutResult",
]
