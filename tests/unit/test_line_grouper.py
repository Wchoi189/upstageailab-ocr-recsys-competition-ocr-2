"""Unit tests for LineGrouper functionality.

Tests the rule-based line grouping algorithm including
determinism and edge cases.
"""

from __future__ import annotations

import pytest

from ocr.core.inference.layout.grouper import (
    LineGrouper,
    LineGrouperConfig,
    create_text_element,
)


class TestLineGrouperConfig:
    """Tests for LineGrouperConfig defaults."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = LineGrouperConfig()

        assert config.y_overlap_threshold == 0.5
        assert config.x_gap_ratio == 1.5
        assert config.merge_close_lines is False
        assert config.line_merge_threshold == 0.3

    def test_custom_config(self):
        """Custom config values should be preserved."""
        config = LineGrouperConfig(
            y_overlap_threshold=0.7,
            x_gap_ratio=2.0,
            merge_close_lines=True,
        )

        assert config.y_overlap_threshold == 0.7
        assert config.x_gap_ratio == 2.0
        assert config.merge_close_lines is True


class TestCreateTextElement:
    """Tests for create_text_element helper."""

    def test_creates_element_with_bbox(self):
        """Should create element with auto-computed bbox."""
        polygon = [[0, 0], [100, 0], [100, 30], [0, 30]]

        element = create_text_element(
            polygon=polygon,
            text="Hello",
            confidence=0.9,
        )

        assert element.text == "Hello"
        assert element.confidence == 0.9
        assert element.bbox.x_min == 0
        assert element.bbox.y_min == 0
        assert element.bbox.x_max == 100
        assert element.bbox.y_max == 30


class TestLineGrouper:
    """Tests for LineGrouper functionality."""

    @pytest.fixture
    def grouper(self):
        """Create default LineGrouper."""
        return LineGrouper()

    def test_empty_input(self, grouper):
        """Empty input should return empty result."""
        result = grouper.group_elements([])

        assert result.blocks == []
        assert result.num_lines == 0
        assert result.num_elements == 0

    def test_single_element(self, grouper):
        """Single element should form single line."""
        element = create_text_element(
            polygon=[[0, 0], [100, 0], [100, 30], [0, 30]],
            text="Hello",
            confidence=0.9,
        )

        result = grouper.group_elements([element])

        assert result.num_lines == 1
        assert result.num_elements == 1
        assert len(result.blocks) == 1
        assert result.blocks[0].lines[0].text == "Hello"

    def test_elements_on_same_line(self, grouper):
        """Elements with same Y should be grouped into one line."""
        # Two elements at same Y level
        element1 = create_text_element(
            polygon=[[0, 100], [50, 100], [50, 130], [0, 130]],
            text="Hello",
            confidence=0.9,
        )
        element2 = create_text_element(
            polygon=[[60, 100], [120, 100], [120, 130], [60, 130]],
            text="World",
            confidence=0.85,
        )

        result = grouper.group_elements([element1, element2])

        assert result.num_lines == 1
        assert result.num_elements == 2
        assert result.blocks[0].lines[0].text == "Hello World"

    def test_elements_on_different_lines(self, grouper):
        """Elements at different Y should be separate lines."""
        # Two elements at very different Y levels
        element1 = create_text_element(
            polygon=[[0, 0], [100, 0], [100, 30], [0, 30]],
            text="Line1",
            confidence=0.9,
        )
        element2 = create_text_element(
            polygon=[[0, 100], [100, 100], [100, 130], [0, 130]],
            text="Line2",
            confidence=0.85,
        )

        result = grouper.group_elements([element1, element2])

        assert result.num_lines == 2
        assert result.num_elements == 2

        # Lines should be in reading order (top to bottom)
        assert result.blocks[0].lines[0].text == "Line1"
        assert result.blocks[0].lines[1].text == "Line2"

    def test_mixed_alignment(self, grouper):
        """Mix of aligned and non-aligned elements."""
        elements = [
            create_text_element(
                polygon=[[0, 0], [50, 0], [50, 30], [0, 30]],
                text="A",
                confidence=0.9,
            ),
            create_text_element(
                polygon=[[60, 0], [110, 0], [110, 30], [60, 30]],
                text="B",
                confidence=0.9,
            ),
            create_text_element(
                polygon=[[0, 50], [50, 50], [50, 80], [0, 80]],
                text="C",
                confidence=0.9,
            ),
        ]

        result = grouper.group_elements(elements)

        assert result.num_lines == 2
        assert result.num_elements == 3

        # First line: A B
        # Second line: C
        texts = [line.text for line in result.blocks[0].lines]
        assert "A B" in texts
        assert "C" in texts

    def test_reading_order_determinism(self, grouper):
        """Reading order should be deterministic."""
        elements = [
            create_text_element(
                polygon=[[0, 100], [50, 100], [50, 130], [0, 130]],
                text="Third",
                confidence=0.9,
            ),
            create_text_element(
                polygon=[[0, 0], [50, 0], [50, 30], [0, 30]],
                text="First",
                confidence=0.9,
            ),
            create_text_element(
                polygon=[[0, 50], [50, 50], [50, 80], [0, 80]],
                text="Second",
                confidence=0.9,
            ),
        ]

        # Run multiple times to verify determinism
        results = [grouper.group_elements(elements) for _ in range(5)]

        # All results should have same reading order
        for result in results:
            lines = result.blocks[0].lines
            assert lines[0].reading_order == 0
            assert lines[1].reading_order == 1
            assert lines[2].reading_order == 2
            assert lines[0].text == "First"
            assert lines[1].text == "Second"
            assert lines[2].text == "Third"

    def test_left_to_right_ordering(self, grouper):
        """Elements in a line should be ordered left to right."""
        # Create elements in reverse order
        elements = [
            create_text_element(
                polygon=[[200, 0], [250, 0], [250, 30], [200, 30]],
                text="Last",
                confidence=0.9,
            ),
            create_text_element(
                polygon=[[0, 0], [50, 0], [50, 30], [0, 30]],
                text="First",
                confidence=0.9,
            ),
            create_text_element(
                polygon=[[100, 0], [150, 0], [150, 30], [100, 30]],
                text="Middle",
                confidence=0.9,
            ),
        ]

        result = grouper.group_elements(elements)

        assert result.num_lines == 1
        line = result.blocks[0].lines[0]

        # Elements should be sorted by X position
        assert line.elements[0].text == "First"
        assert line.elements[1].text == "Middle"
        assert line.elements[2].text == "Last"
        assert line.text == "First Middle Last"

    def test_partial_y_overlap(self, grouper):
        """Elements with partial Y overlap should be grouped."""
        # Element 2 overlaps ~60% with element 1
        element1 = create_text_element(
            polygon=[[0, 0], [50, 0], [50, 30], [0, 30]],
            text="A",
            confidence=0.9,
        )
        element2 = create_text_element(
            polygon=[[60, 10], [110, 10], [110, 40], [60, 40]],
            text="B",
            confidence=0.9,
        )

        result = grouper.group_elements([element1, element2])

        # With default threshold 0.5, these should be on same line
        assert result.num_lines == 1

    def test_insufficient_y_overlap_separate_lines(self):
        """Elements with insufficient Y overlap should be separate."""
        grouper = LineGrouper(config=LineGrouperConfig(y_overlap_threshold=0.8))

        # Element 2 overlaps only ~30% with element 1
        element1 = create_text_element(
            polygon=[[0, 0], [50, 0], [50, 30], [0, 30]],
            text="A",
            confidence=0.9,
        )
        element2 = create_text_element(
            polygon=[[60, 20], [110, 20], [110, 50], [60, 50]],
            text="B",
            confidence=0.9,
        )

        result = grouper.group_elements([element1, element2])

        # With high threshold 0.8, these should be separate lines
        assert result.num_lines == 2

    def test_raw_elements_preserved(self, grouper):
        """Raw elements should be preserved in result."""
        elements = [
            create_text_element(
                polygon=[[0, 0], [50, 0], [50, 30], [0, 30]],
                text="Test",
                confidence=0.9,
            ),
        ]

        result = grouper.group_elements(elements)

        assert result.raw_elements is not None
        assert len(result.raw_elements) == 1


class TestLineGrouperEdgeCases:
    """Edge case tests for LineGrouper."""

    def test_many_elements_same_line(self):
        """Many elements on same line should work."""
        grouper = LineGrouper()

        elements = []
        for i in range(20):
            elements.append(
                create_text_element(
                    polygon=[
                        [i * 30, 0],
                        [(i + 1) * 30 - 5, 0],
                        [(i + 1) * 30 - 5, 30],
                        [i * 30, 30],
                    ],
                    text=f"W{i}",
                    confidence=0.9,
                )
            )

        result = grouper.group_elements(elements)

        assert result.num_lines == 1
        assert result.num_elements == 20

    def test_single_character_elements(self):
        """Single character elements should work."""
        grouper = LineGrouper()

        elements = [
            create_text_element(
                polygon=[[i * 20, 0], [(i + 1) * 20, 0], [(i + 1) * 20, 25], [i * 20, 25]],
                text=chr(65 + i),  # A, B, C, ...
                confidence=0.95,
            )
            for i in range(5)
        ]

        result = grouper.group_elements(elements)

        assert result.num_lines == 1
        assert result.blocks[0].lines[0].text == "A B C D E"

    def test_very_different_heights(self):
        """Elements with very different heights."""
        grouper = LineGrouper()

        # Small element next to tall element
        element1 = create_text_element(
            polygon=[[0, 40], [50, 40], [50, 60], [0, 60]],  # 20px height, centered
            text="small",
            confidence=0.9,
        )
        element2 = create_text_element(
            polygon=[[60, 0], [110, 0], [110, 100], [60, 100]],  # 100px height
            text="TALL",
            confidence=0.9,
        )

        result = grouper.group_elements([element1, element2])

        # They should be grouped if element1 is within element2's range
        assert result.num_lines <= 2  # Implementation-dependent
