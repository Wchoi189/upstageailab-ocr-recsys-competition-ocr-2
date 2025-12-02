# Coding Standards for VLM Module

## Overview

This document defines coding standards and best practices for the VLM module, designed to maintain code quality and prepare for potential future independence as a standalone project.

## Style Guide

- Follow PEP 8 Python style guide
- Use `ruff` for linting and formatting
- Maximum line length: 100 characters
- Use type hints for all public functions and classes
- Use Google-style docstrings

## Type Hints

All public functions and classes must have complete type hints:

```python
def analyze_image(
    self,
    image_data: ProcessedImage,
    prompt: str,
    mode: AnalysisMode,
) -> str:
    """Analyze an image."""
    ...
```

## Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
    """
    ...
```

## Error Handling

- Use custom exception classes from `core.interfaces`
- Provide clear error messages
- Include context in error messages
- Use exception chaining (`raise ... from ...`)

## Logging

Use structured logging:

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Processing image", extra={"image_path": str(image_path)})
```

## Testing

- Minimum 80% code coverage
- Use pytest for testing
- Mock external dependencies (APIs, file system)
- Test both success and failure cases

## Import Organization

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
import json
import os
from pathlib import Path

from pydantic import BaseModel

from AgentQMS.vlm.core.contracts import ImageData
```

## Naming Conventions

- **Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: Prefix with `_`

## Code Organization

- One class per file (when possible)
- Group related functionality
- Keep functions focused and small
- Use composition over inheritance

## Independence Considerations

- Minimize external dependencies
- Use abstract interfaces for external systems
- Provide clear public API
- Document all public interfaces
- Version management for future releases
