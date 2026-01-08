"""Line grouping module for layout detection.

This module provides rule-based grouping of text elements into lines
based on spatial proximity and overlap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .contracts import (
    BoundingBox,
    LayoutResult,
    TextBlock,
    TextElement,
    TextLine,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class LineGrouperConfig:
    """Configuration for line grouping algorithm.

    Attributes:
        y_overlap_threshold: Minimum Y overlap ratio to consider elements on same line (0-1)
        x_gap_ratio: Maximum horizontal gap as ratio of average element height
        merge_close_lines: Whether to merge lines that are very close vertically
        line_merge_threshold: Vertical distance threshold for merging lines
    """

    y_overlap_threshold: float = 0.5
    x_gap_ratio: float = 1.5
    merge_close_lines: bool = False
    line_merge_threshold: float = 0.3


@dataclass
class LineGrouper:
    """Group text elements into lines based on spatial proximity.

    Uses a rule-based algorithm that:
    1. Sorts elements by Y-center position
    2. Groups elements with sufficient Y-overlap
    3. Sorts each line by X position
    4. Assigns reading order

    Example:
        >>> grouper = LineGrouper()
        >>> result = grouper.group_elements(elements)
        >>> print(f"Found {result.num_lines} lines")
    """

    config: LineGrouperConfig = field(default_factory=LineGrouperConfig)

    def group_elements(self, elements: list[TextElement]) -> LayoutResult:
        """Group text elements into lines and blocks.

        Args:
            elements: List of TextElement objects to group

        Returns:
            LayoutResult with hierarchical structure
        """
        if not elements:
            return LayoutResult(blocks=[], reading_order=[], raw_elements=[])

        LOGGER.debug("Grouping %d elements into lines", len(elements))

        # Group into lines
        lines = self._group_into_lines(elements)

        # Sort lines by reading order (top to bottom)
        lines.sort(key=lambda line: line.bbox.y_min)

        # Update reading order
        ordered_lines = []
        for i, line in enumerate(lines):
            # Create new line with updated reading order
            ordered_line = TextLine(
                id=line.id,
                elements=line.elements,
                reading_order=i,
                bbox=line.bbox,
            )
            ordered_lines.append(ordered_line)

        # Create a single block containing all lines (simple case)
        # Future: Implement paragraph detection
        if ordered_lines:
            block_bbox = self._compute_merged_bbox([line.bbox for line in ordered_lines])
            block = TextBlock(
                block_type="paragraph",
                lines=ordered_lines,
                bbox=block_bbox,
            )
            blocks = [block]
            reading_order = [block.id]
        else:
            blocks = []
            reading_order = []

        result = LayoutResult(
            blocks=blocks,
            reading_order=reading_order,
            raw_elements=elements,
        )

        LOGGER.debug(
            "Grouped into %d blocks with %d lines",
            len(blocks),
            result.num_lines,
        )

        return result

    def _group_into_lines(self, elements: list[TextElement]) -> list[TextLine]:
        """Group elements into lines based on Y-overlap.

        Args:
            elements: List of elements to group

        Returns:
            List of TextLine objects
        """
        if not elements:
            return []

        # Sort by Y-center for initial grouping
        sorted_elements = sorted(elements, key=lambda e: e.bbox.center[1])

        lines: list[list[TextElement]] = []
        current_line: list[TextElement] = [sorted_elements[0]]

        for element in sorted_elements[1:]:
            # Check if this element belongs to the current line
            if self._should_merge_to_line(element, current_line):
                current_line.append(element)
            else:
                # Start a new line
                lines.append(current_line)
                current_line = [element]

        # Don't forget the last line
        if current_line:
            lines.append(current_line)

        # Convert to TextLine objects
        text_lines: list[TextLine] = []
        for i, line_elements in enumerate(lines):
            # Sort elements within line by X position
            line_elements.sort(key=lambda e: e.bbox.x_min)

            # Compute line bounding box
            line_bbox = self._compute_merged_bbox([e.bbox for e in line_elements])

            text_line = TextLine(
                elements=line_elements,
                reading_order=i,  # Will be updated later
                bbox=line_bbox,
            )
            text_lines.append(text_line)

        return text_lines

    def _should_merge_to_line(
        self,
        element: TextElement,
        current_line: list[TextElement],
    ) -> bool:
        """Determine if an element should be added to the current line.

        Args:
            element: Element to check
            current_line: Current line elements

        Returns:
            True if element should be merged to current line
        """
        if not current_line:
            return True

        # Check Y-overlap with all elements in current line
        for line_element in current_line:
            overlap = element.bbox.y_overlap_ratio(line_element.bbox)
            if overlap >= self.config.y_overlap_threshold:
                return True

        # Additionally check if Y-center is within the line's Y range
        line_bbox = self._compute_merged_bbox([e.bbox for e in current_line])
        element_y_center = element.bbox.center[1]

        # Element's center should be within the line's vertical extent
        if line_bbox.y_min <= element_y_center <= line_bbox.y_max:
            return True

        return False

    @staticmethod
    def _compute_merged_bbox(bboxes: list[BoundingBox]) -> BoundingBox:
        """Compute bounding box that contains all input boxes.

        Args:
            bboxes: List of bounding boxes to merge

        Returns:
            Single bounding box containing all inputs
        """
        if not bboxes:
            return BoundingBox(x_min=0, y_min=0, x_max=0, y_max=0)

        return BoundingBox(
            x_min=min(b.x_min for b in bboxes),
            y_min=min(b.y_min for b in bboxes),
            x_max=max(b.x_max for b in bboxes),
            y_max=max(b.y_max for b in bboxes),
        )


def create_text_element(
    polygon: list[list[float]],
    text: str,
    confidence: float,
) -> TextElement:
    """Helper to create a TextElement with auto-computed bounding box.

    Args:
        polygon: List of [x, y] coordinates
        text: Recognized text
        confidence: Recognition confidence

    Returns:
        TextElement with computed bbox
    """
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    bbox = BoundingBox(
        x_min=min(xs),
        y_min=min(ys),
        x_max=max(xs),
        y_max=max(ys),
    )

    return TextElement(
        polygon=polygon,
        bbox=bbox,
        text=text,
        confidence=confidence,
    )


__all__ = [
    "LineGrouper",
    "LineGrouperConfig",
    "create_text_element",
]
