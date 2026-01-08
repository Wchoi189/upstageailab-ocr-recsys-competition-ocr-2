# Layout Detection Feature

## Overview

The **layout** feature provides layout analysis capabilities for the OCR pipeline. It groups text elements detected by OCR into logical structures (lines and blocks) based on spatial relationships.

## Purpose

Layout detection is a **feature** (not core infrastructure) with specialized algorithms for:
- Grouping text elements into coherent lines based on Y-overlap
- Organizing lines into blocks for reading order
- Handling edge cases (rotated text, different heights, etc.)

## Components

### `inference/grouper.py`
- **`LineGrouper`**: Rule-based algorithm for grouping text elements into lines
- **`LineGrouperConfig`**: Configuration for grouping thresholds
- **`create_text_element`**: Helper function to create TextElement from polygon

### `inference/contracts.py`
- **`TextElement`**: Represents a single detected text element with bounding box
- **`TextLine`**: Represents a line of grouped text elements
- **`TextBlock`**: Represents a block containing multiple lines
- **`LayoutResult`**: Complete layout analysis result
- **`BBox`**: Bounding box representation (x_min, y_min, x_max, y_max)

## Usage

```python
from ocr.features.layout import LineGrouper, create_text_element

# Create grouper
grouper = LineGrouper()

# Create text elements
elements = [
    create_text_element(
        polygon=[[0, 0], [100, 0], [100, 30], [0, 30]],
        text="Hello",
        confidence=0.9
    ),
    # ... more elements
]

# Group elements into lines
result = grouper.group_elements(elements)

# Access results
for line in result.blocks[0].lines:
    print(f"Line {line.reading_order}: {line.text}")
```

## Design Principles

### Why Layout Is a Feature
1. **Specialized Algorithms**: Uses geometric analysis and heuristics specific to layout detection
2. **Domain-Specific**: Focused on spatial relationships between text elements
3. **Pluggable**: Could support alternative implementations (rule-based vs. ML-based)
4. **Self-Contained**: Has its own data models and configuration

### Relationship to Core
- **Core provides**: Generic base classes, orchestration framework
- **Layout feature provides**: Specific implementation of layout analysis
- **Interface**: Layout could implement a `LayoutDetector` interface defined in `ocr/core/interfaces/` (future work)

## Testing

Tests are located in:
- `tests/unit/test_line_grouper.py` - Tests for LineGrouper algorithm
- `tests/unit/test_layout_contracts.py` - Tests for data models

Run tests:
```bash
uv run pytest tests/unit/test_line_grouper.py tests/unit/test_layout_contracts.py -v
```

## Future Enhancements
- ML-based layout detection as alternative to rule-based grouper
- Support for column detection and complex layouts
- Document-level layout analysis (page segmentation)
