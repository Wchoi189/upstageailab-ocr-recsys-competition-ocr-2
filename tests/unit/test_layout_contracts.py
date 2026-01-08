"""Unit tests for layout detection contracts.

Tests the Pydantic models for layout structures and validates
contract shapes and invariants.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ocr.core.inference.layout.contracts import (
    BoundingBox,
    LayoutResult,
    TextBlock,
    TextElement,
    TextLine,
)


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_valid_bbox_creation(self):
        """Valid bounding box should be accepted."""
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)

        assert bbox.x_min == 0
        assert bbox.y_min == 0
        assert bbox.x_max == 100
        assert bbox.y_max == 50

    def test_width_property(self):
        """Width should be computed correctly."""
        bbox = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=70)
        assert bbox.width == 100

    def test_height_property(self):
        """Height should be computed correctly."""
        bbox = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=70)
        assert bbox.height == 50

    def test_center_property(self):
        """Center should be computed correctly."""
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        assert bbox.center == (50.0, 25.0)

    def test_area_property(self):
        """Area should be computed correctly."""
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        assert bbox.area == 5000

    def test_x_max_less_than_x_min_raises(self):
        """x_max < x_min should raise validation error."""
        with pytest.raises(ValueError, match="x_max"):
            BoundingBox(x_min=100, y_min=0, x_max=50, y_max=50)

    def test_y_max_less_than_y_min_raises(self):
        """y_max < y_min should raise validation error."""
        with pytest.raises(ValueError, match="y_max"):
            BoundingBox(x_min=0, y_min=100, x_max=50, y_max=50)

    def test_zero_size_bbox_valid(self):
        """Zero-size bounding box should be valid."""
        bbox = BoundingBox(x_min=50, y_min=50, x_max=50, y_max=50)
        assert bbox.width == 0
        assert bbox.height == 0
        assert bbox.area == 0

    def test_iou_full_overlap(self):
        """IoU of identical boxes should be 1.0."""
        bbox1 = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
        bbox2 = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
        assert bbox1.iou(bbox2) == 1.0

    def test_iou_no_overlap(self):
        """IoU of non-overlapping boxes should be 0.0."""
        bbox1 = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50)
        bbox2 = BoundingBox(x_min=100, y_min=100, x_max=150, y_max=150)
        assert bbox1.iou(bbox2) == 0.0

    def test_iou_partial_overlap(self):
        """IoU of partially overlapping boxes."""
        bbox1 = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)
        bbox2 = BoundingBox(x_min=50, y_min=50, x_max=150, y_max=150)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 ≈ 0.143
        iou = bbox1.iou(bbox2)
        assert 0.14 < iou < 0.15

    def test_y_overlap_ratio_full(self):
        """Y overlap ratio of vertically aligned boxes should be 1.0."""
        bbox1 = BoundingBox(x_min=0, y_min=10, x_max=50, y_max=30)
        bbox2 = BoundingBox(x_min=60, y_min=10, x_max=110, y_max=30)
        assert bbox1.y_overlap_ratio(bbox2) == 1.0

    def test_y_overlap_ratio_partial(self):
        """Y overlap ratio of partially overlapping boxes."""
        bbox1 = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=20)
        bbox2 = BoundingBox(x_min=60, y_min=10, x_max=110, y_max=30)
        # Overlap: 10 (from 10 to 20)
        # Min height: 20
        # Ratio: 10/20 = 0.5
        assert bbox1.y_overlap_ratio(bbox2) == 0.5

    def test_y_overlap_ratio_none(self):
        """Y overlap ratio of non-overlapping boxes should be 0.0."""
        bbox1 = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=20)
        bbox2 = BoundingBox(x_min=0, y_min=30, x_max=50, y_max=50)
        assert bbox1.y_overlap_ratio(bbox2) == 0.0

    def test_bbox_is_frozen(self):
        """BoundingBox should be immutable."""
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=50)
        with pytest.raises(ValidationError):  # Frozen model
            bbox.x_min = 10


class TestTextElement:
    """Tests for TextElement model."""

    def test_valid_element_creation(self):
        """Valid element should be accepted."""
        polygon = [[0, 0], [100, 0], [100, 30], [0, 30]]
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=30)

        element = TextElement(
            polygon=polygon,
            bbox=bbox,
            text="Hello",
            confidence=0.95,
        )

        assert element.text == "Hello"
        assert element.confidence == 0.95
        assert len(element.polygon) == 4

    def test_auto_generated_id(self):
        """Element should have auto-generated ID."""
        polygon = [[0, 0], [100, 0], [100, 30], [0, 30]]
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=30)

        element = TextElement(
            polygon=polygon,
            bbox=bbox,
            text="Test",
            confidence=0.9,
        )

        assert element.id is not None
        assert len(element.id) == 8  # UUID prefix

    def test_polygon_too_few_points(self):
        """Polygon with < 3 points should raise error."""
        polygon = [[0, 0], [100, 0]]  # Only 2 points
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=30)

        with pytest.raises(ValueError, match="at least 3"):
            TextElement(
                polygon=polygon,
                bbox=bbox,
                text="Test",
                confidence=0.9,
            )

    def test_invalid_polygon_point_format(self):
        """Polygon points must be [x, y] pairs."""
        polygon = [[0, 0, 0], [100, 0], [100, 30], [0, 30]]  # 3D point
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=30)

        with pytest.raises(ValueError, match="\\[x, y\\]"):
            TextElement(
                polygon=polygon,
                bbox=bbox,
                text="Test",
                confidence=0.9,
            )

    def test_confidence_out_of_range(self):
        """Confidence must be 0-1."""
        polygon = [[0, 0], [100, 0], [100, 30], [0, 30]]
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=30)

        with pytest.raises(ValueError):
            TextElement(
                polygon=polygon,
                bbox=bbox,
                text="Test",
                confidence=1.5,
            )


class TestTextLine:
    """Tests for TextLine model."""

    @pytest.fixture
    def sample_elements(self):
        """Create sample text elements for testing."""
        return [
            TextElement(
                polygon=[[0, 0], [50, 0], [50, 30], [0, 30]],
                bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=30),
                text="Hello",
                confidence=0.9,
            ),
            TextElement(
                polygon=[[60, 0], [110, 0], [110, 30], [60, 30]],
                bbox=BoundingBox(x_min=60, y_min=0, x_max=110, y_max=30),
                text="World",
                confidence=0.85,
            ),
        ]

    def test_valid_line_creation(self, sample_elements):
        """Valid line should be accepted."""
        bbox = BoundingBox(x_min=0, y_min=0, x_max=110, y_max=30)

        line = TextLine(
            elements=sample_elements,
            reading_order=0,
            bbox=bbox,
        )

        assert len(line.elements) == 2
        assert line.reading_order == 0

    def test_text_property(self, sample_elements):
        """Text should concatenate element texts."""
        bbox = BoundingBox(x_min=0, y_min=0, x_max=110, y_max=30)

        line = TextLine(
            elements=sample_elements,
            reading_order=0,
            bbox=bbox,
        )

        assert line.text == "Hello World"

    def test_confidence_property(self, sample_elements):
        """Confidence should average element confidences."""
        bbox = BoundingBox(x_min=0, y_min=0, x_max=110, y_max=30)

        line = TextLine(
            elements=sample_elements,
            reading_order=0,
            bbox=bbox,
        )

        expected = (0.9 + 0.85) / 2
        assert abs(line.confidence - expected) < 0.001

    def test_empty_line_confidence(self):
        """Empty line should have 0 confidence."""
        bbox = BoundingBox(x_min=0, y_min=0, x_max=0, y_max=0)

        line = TextLine(
            elements=[],
            reading_order=0,
            bbox=bbox,
        )

        assert line.confidence == 0.0


class TestTextBlock:
    """Tests for TextBlock model."""

    @pytest.fixture
    def sample_lines(self):
        """Create sample text lines for testing."""
        element1 = TextElement(
            polygon=[[0, 0], [50, 0], [50, 30], [0, 30]],
            bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=30),
            text="Line1",
            confidence=0.9,
        )
        element2 = TextElement(
            polygon=[[0, 40], [50, 40], [50, 70], [0, 70]],
            bbox=BoundingBox(x_min=0, y_min=40, x_max=50, y_max=70),
            text="Line2",
            confidence=0.85,
        )

        line1 = TextLine(
            elements=[element1],
            reading_order=0,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=30),
        )
        line2 = TextLine(
            elements=[element2],
            reading_order=1,
            bbox=BoundingBox(x_min=0, y_min=40, x_max=50, y_max=70),
        )

        return [line1, line2]

    def test_valid_block_creation(self, sample_lines):
        """Valid block should be accepted."""
        bbox = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=70)

        block = TextBlock(
            block_type="paragraph",
            lines=sample_lines,
            bbox=bbox,
        )

        assert block.block_type == "paragraph"
        assert len(block.lines) == 2

    def test_text_property(self, sample_lines):
        """Text should concatenate line texts with newlines."""
        bbox = BoundingBox(x_min=0, y_min=0, x_max=50, y_max=70)

        block = TextBlock(
            block_type="paragraph",
            lines=sample_lines,
            bbox=bbox,
        )

        assert block.text == "Line1\nLine2"


class TestLayoutResult:
    """Tests for LayoutResult model."""

    def test_empty_result(self):
        """Empty result should be valid."""
        result = LayoutResult()

        assert result.blocks == []
        assert result.reading_order == []
        assert result.num_elements == 0
        assert result.num_lines == 0
        assert result.text == ""

    def test_with_blocks(self):
        """Result with blocks should compute properties."""
        element = TextElement(
            polygon=[[0, 0], [50, 0], [50, 30], [0, 30]],
            bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=30),
            text="Test",
            confidence=0.9,
        )
        line = TextLine(
            elements=[element],
            reading_order=0,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=30),
        )
        block = TextBlock(
            block_type="paragraph",
            lines=[line],
            bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=30),
        )

        result = LayoutResult(
            blocks=[block],
            reading_order=[block.id],
        )

        assert result.num_elements == 1
        assert result.num_lines == 1
        assert result.text == "Test"

    def test_get_block_by_id(self):
        """Should be able to find block by ID."""
        line = TextLine(
            elements=[],
            reading_order=0,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=30),
        )
        block = TextBlock(
            id="test123",
            block_type="paragraph",
            lines=[line],
            bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=30),
        )

        result = LayoutResult(blocks=[block])

        found = result.get_block_by_id("test123")
        assert found is block

        not_found = result.get_block_by_id("nonexistent")
        assert not_found is None
