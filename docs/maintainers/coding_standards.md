# Coding Standards

This document defines coding standards for the OCR playground project, covering both Python (backend) and TypeScript/React (frontend) codebases.

## Table of Contents

- [General Principles](#general-principles)
- [Python Standards](#python-standards)
- [TypeScript/React Standards](#typescriptreact-standards)
- [Code Length Guidelines](#code-length-guidelines)
- [File Organization](#file-organization)
- [Naming Conventions](#naming-conventions)
- [Documentation](#documentation)
- [Testing Standards](#testing-standards)

---

## General Principles

1. **Readability First**: Code should be self-documenting with clear naming
2. **Consistency**: Follow existing patterns and conventions
3. **Type Safety**: Use type hints (Python) and TypeScript types everywhere
4. **DRY**: Don't Repeat Yourself, but prioritize clarity over abstraction
5. **Performance**: Optimize for the playground's real-time requirements

---

## Python Standards

### Line Length

- **Maximum line length: 140 characters** (configured in `pyproject.toml`)
- Soft limit: 120 characters (prefer breaking at 120 when possible)
- Hard limit: 140 characters (enforced by Ruff)

**Breaking long lines:**
```python
# Good: Break function calls with aligned arguments
result = some_function(
    argument_one="value",
    argument_two="another value",
    argument_three="yet another value",
)

# Good: Break long conditions
if (
    condition_one
    and condition_two
    and condition_three
):
    do_something()
```

### Function Length

- **Target: < 50 lines per function**
- **Maximum: 100 lines** (consider refactoring if exceeded)
- Break complex functions into smaller, focused functions

```python
# Good: Focused, single-responsibility function
def process_image(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply preprocessing pipeline to image."""
    if params.get("autocontrast"):
        image = apply_autocontrast(image)
    if params.get("blur"):
        image = apply_blur(image, params["blur_radius"])
    return image

# Bad: Too long, multiple responsibilities
def process_everything(image, params, metadata, cache, ...):  # 150+ lines
    # ... too much logic ...
```

### Class Length

- **Target: < 200 lines per class**
- **Maximum: 400 lines** (consider splitting into multiple classes)
- Use composition over inheritance when classes grow large

### File Length

- **Target: < 500 lines per file**
- **Maximum: 1000 lines** (split into modules if exceeded)
- Group related functionality into separate modules

### Type Hints

- **Required** for all function signatures
- Use `from __future__ import annotations` for forward references
- Prefer `dict[str, Any]` over `Dict[str, Any]` (Python 3.9+)

```python
from __future__ import annotations

def build_command(
    schema_id: str,
    values: dict[str, Any],
    append_suffix: bool = True,
) -> str:
    """Build CLI command from schema values."""
    ...
```

### Import Organization

- Standard library imports
- Third-party imports
- Local application imports
- Use `isort` (via Ruff) to auto-format

```python
# Standard library
from pathlib import Path
from typing import Any

# Third-party
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

# Local - Path utilities (import from central utility)
from ocr.utils.path_utils import PROJECT_ROOT  # ✅ Preferred: stable, works from any location
# OR
from services.playground_api.utils.paths import PROJECT_ROOT  # ✅ Also valid (re-exports from ocr.utils.path_utils)
```

### Path Management

**CRITICAL**: Use centralized path utilities to avoid brittle path calculations.

**✅ DO**:
```python
# Import PROJECT_ROOT from central utility (stable, works from any location)
from ocr.utils.path_utils import PROJECT_ROOT

# Use path resolver for configurable paths
from ocr.utils.path_utils import get_path_resolver

resolver = get_path_resolver()
output_dir = resolver.config.output_dir
config_dir = resolver.config.config_dir
```

**❌ DON'T**:
```python
# ❌ Brittle - breaks if file moved
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ❌ Hardcoded - breaks in different environments
output_dir = Path("outputs")
config_dir = PROJECT_ROOT / "configs"
```

**Environment Variables**:
- Paths support environment variable overrides (`OCR_PROJECT_ROOT`, `OCR_OUTPUT_DIR`, etc.)
- See `docs/maintainers/environment-variables.md` for complete list
- FastAPI and Streamlit apps automatically read env vars on startup

### Error Handling

- Use specific exception types
- Include context in error messages
- Use `raise ... from` for exception chaining

```python
# Good
try:
    result = process_data(data)
except ValueError as e:
    raise ProcessingError(f"Failed to process {data}: {e}") from e

# Bad
try:
    result = process_data(data)
except:  # Too broad
    raise  # No context
```

---

## TypeScript/React Standards

### Line Length

- **Maximum line length: 100 characters** (stricter than Python for readability)
- Use Prettier for automatic formatting
- Break long lines at logical points

```typescript
// Good: Break long function calls
const result = await fetch(
  `/api/pipelines/preview`,
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  }
);

// Good: Break long type definitions
type WorkerTask<TParams = Record<string, unknown>> = {
  taskId: string;
  type: WorkerTaskType;
  params: TParams;
  imageBuffer: ArrayBuffer;
  metadata?: Record<string, unknown>;
};
```

### Function Length

- **Target: < 40 lines per function**
- **Maximum: 80 lines** (consider extracting logic)
- React components can be longer but should be broken into smaller components

```typescript
// Good: Focused function
async function handleTask(task: WorkerTask): Promise<WorkerResult> {
  const start = performance.now();
  const handler = registry[task.type];
  if (!handler) {
    return createErrorResult(task.taskId, `Unknown task type: ${task.type}`);
  }
  try {
    const bitmap = await handler(task, ctx);
    return createSuccessResult(task.taskId, bitmap, start);
  } catch (error) {
    return createErrorResult(task.taskId, error);
  }
}

// Bad: Too long, multiple responsibilities
async function doEverything(task, context, cache, ...): Promise<Result> {
  // 150+ lines of mixed logic
}
```

### Component Length

- **Target: < 150 lines per React component**
- **Maximum: 300 lines** (extract sub-components or hooks)
- Extract custom hooks for complex logic

```typescript
// Good: Component with extracted hook
function PreprocessingCanvas({ image, params }: Props) {
  const { result, loading, error } = useWorkerTask(image, params);
  return <CanvasView result={result} loading={loading} error={error} />;
}

// Bad: Monolithic component
function PreprocessingCanvas({ image, params }: Props) {
  // 400+ lines of state, effects, handlers, and JSX
}
```

### File Length

- **Target: < 300 lines per file**
- **Maximum: 500 lines** (split into multiple files)
- One component/hook per file (except closely related utilities)

### Type Safety

- **Required**: All functions must have explicit return types
- Use `interface` for object shapes, `type` for unions/intersections
- Avoid `any`; use `unknown` when type is truly unknown

```typescript
// Good: Explicit types
interface WorkerResult<TPayload = Record<string, unknown>> {
  taskId: string;
  status: "success" | "error";
  payload?: TPayload;
  error?: string;
  durationMs: number;
}

function processTask(task: WorkerTask): Promise<WorkerResult> {
  // ...
}

// Bad: Missing types
function processTask(task) {  // No types
  // ...
}
```

### Import Organization

- React imports first
- Third-party imports
- Local imports (absolute paths preferred)
- Type-only imports with `import type`

```typescript
// React
import { useState, useEffect } from "react";

// Third-party
import { z } from "zod";
import { useQuery } from "@tanstack/react-query";

// Local (absolute paths)
import type { WorkerTask, WorkerResult } from "@/workers/types";
import { useWorkerTask } from "@/hooks/useWorkerTask";
```

### React Patterns

- Use functional components with hooks
- Extract custom hooks for reusable logic
- Use `useMemo` and `useCallback` judiciously (not everywhere)
- Prefer composition over prop drilling

```typescript
// Good: Custom hook extraction
function useImageProcessing(image: ImageData, params: Params) {
  const [result, setResult] = useState<ImageData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Processing logic
  }, [image, params]);

  return { result, loading };
}

// Good: Component using hook
function ImageViewer({ image }: Props) {
  const { result, loading } = useImageProcessing(image, params);
  return loading ? <Spinner /> : <Canvas image={result} />;
}
```

---

## Code Length Guidelines

### Summary Table

| Language | Line Length | Function | Class/Component | File |
|----------|-------------|----------|-----------------|------|
| **Python** | 140 (hard) | 50 (target)<br>100 (max) | 200 (target)<br>400 (max) | 500 (target)<br>1000 (max) |
| **TypeScript** | 100 (hard) | 40 (target)<br>80 (max) | 150 (target)<br>300 (max) | 300 (target)<br>500 (max) |

### When to Refactor

Refactor when:
- Functions exceed maximum length
- Functions have > 3 levels of nesting
- Functions have > 5 parameters (use dataclass/Pydantic model)
- Cyclomatic complexity > 10
- Files exceed maximum length

### Refactoring Strategies

1. **Extract functions**: Break large functions into smaller, focused ones
2. **Extract classes/components**: Split large classes into smaller, composable units
3. **Extract modules**: Move related functionality to separate files
4. **Use composition**: Prefer composition over inheritance
5. **Create utilities**: Extract common patterns into utility functions

---

## File Organization

### Python Structure

```
services/
  playground_api/
    __init__.py          # Package exports
    app.py               # FastAPI app factory
    routers/
      __init__.py
      command_builder.py # Route handlers
      inference.py
    utils/
      __init__.py
      paths.py           # Utility functions
```

### TypeScript Structure

```
frontend/
  src/
    components/          # React components
      ui/               # Reusable UI components
      features/         # Feature-specific components
    hooks/              # Custom React hooks
    workers/            # Web Worker code
    api/                # API client code
    types/              # TypeScript type definitions
    utils/              # Utility functions
```

### Naming Files

- **Python**: `snake_case.py` (e.g., `command_builder.py`)
- **TypeScript**: `camelCase.tsx` for components, `camelCase.ts` for utilities
- **React Components**: `PascalCase.tsx` (e.g., `PreprocessingCanvas.tsx`)

---

## Naming Conventions

### Python

- **Functions/Methods**: `snake_case` (e.g., `build_command`, `process_image`)
- **Classes**: `PascalCase` (e.g., `CommandBuilder`, `InferenceService`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- **Private**: Prefix with `_` (e.g., `_internal_helper`)

### TypeScript

- **Functions/Variables**: `camelCase` (e.g., `buildCommand`, `processImage`)
- **Components**: `PascalCase` (e.g., `PreprocessingCanvas`, `CommandBuilder`)
- **Types/Interfaces**: `PascalCase` (e.g., `WorkerTask`, `ApiResponse`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- **Private**: Prefix with `_` (e.g., `_internalHelper`)

---

## Documentation

### Python Docstrings

- Use Google-style docstrings
- Include type information in docstrings if not in type hints
- Document parameters, return values, and exceptions

```python
def build_command(
    schema_id: str,
    values: dict[str, Any],
    append_suffix: bool = True,
) -> str:
    """Build CLI command from schema values.

    Args:
        schema_id: Identifier of the schema to use (train, test, predict).
        values: Form values keyed by schema element key.
        append_suffix: Whether to append architecture/backbone to experiment name.

    Returns:
        Complete CLI command string ready for execution.

    Raises:
        ValueError: If schema_id is invalid or required values are missing.
    """
    ...
```

### TypeScript JSDoc

- Use JSDoc for complex functions
- Document parameters and return types
- Include examples for public APIs

```typescript
/**
 * Processes a worker task and returns the result.
 *
 * @param task - The worker task to process
 * @param ctx - Transform context with canvas and rendering context
 * @returns Promise resolving to worker result with bitmap or error
 * @throws {Error} If task type is unknown or processing fails
 */
async function handleTask(
  task: WorkerTask,
  ctx: TransformContext
): Promise<WorkerResult> {
  // ...
}
```

### Inline Comments

- Explain **why**, not **what** (code should be self-explanatory)
- Use comments for complex algorithms or business logic
- Keep comments up-to-date with code changes

```python
# Good: Explains why
# Use MD5 for cache keys (fast, non-cryptographic hash is sufficient)
cache_key = hashlib.md5(image_bytes).hexdigest()

# Bad: Explains what (obvious from code)
# Calculate MD5 hash of image bytes
cache_key = hashlib.md5(image_bytes).hexdigest()
```

---

## Testing Standards

### Python Tests

- **File naming**: `test_*.py` or `*_test.py`
- **Function naming**: `test_<functionality>`
- **Target coverage**: > 80% for critical paths
- Use pytest fixtures for setup/teardown

```python
def test_build_command_with_valid_schema():
    """Test command building with valid schema and values."""
    builder = CommandBuilder()
    result = builder.build_command_from_overrides(
        script="train.py",
        overrides=["model.encoder=resnet18", "data.batch_size=16"],
    )
    assert "train.py" in result
    assert "resnet18" in result
```

### TypeScript Tests

- **File naming**: `*.test.ts` or `*.test.tsx`
- **Function naming**: `describe` blocks with `it` or `test`
- Use Vitest or Jest
- Test user interactions, not implementation details

```typescript
describe("useWorkerTask", () => {
  it("should process task and return result", async () => {
    const task = createMockTask();
    const { result } = renderHook(() => useWorkerTask(task));
    await waitFor(() => expect(result.current.status).toBe("success"));
  });
});
```

---

## Tools & Automation

### Python

- **Linting**: Ruff (configured in `pyproject.toml`)
- **Formatting**: Ruff formatter
- **Type Checking**: mypy
- **Pre-commit**: Hooks configured in `.pre-commit-config.yaml`

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type check
uv run mypy services/
```

### TypeScript

- **Linting**: ESLint (to be configured)
- **Formatting**: Prettier (to be configured)
- **Type Checking**: TypeScript compiler
- **Pre-commit**: Add ESLint/Prettier hooks

```bash
# Format code
npm run format

# Lint code
npm run lint

# Type check
npm run type-check
```

---

## Exceptions

These standards are guidelines, not strict rules. Exceptions are acceptable when:

1. **Performance**: Optimizations that require longer functions/files
2. **Third-party APIs**: Matching external API patterns
3. **Legacy Code**: Gradual migration of existing code
4. **Complex Algorithms**: Well-documented, complex logic that benefits from being in one place

When making exceptions, document the reason in a comment:

```python
# Exception: This function is intentionally long (120 lines) to keep the
# image processing pipeline in one place for performance optimization.
# Breaking it up would add function call overhead in the hot path.
def process_image_pipeline(image: np.ndarray, ...) -> np.ndarray:
    ...
```

---

## References

- [PEP 8](https://pep8.org/) - Python style guide
- [TypeScript Style Guide](https://google.github.io/styleguide/tsguide.html)
- [React Best Practices](https://react.dev/learn/thinking-in-react)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Prettier Documentation](https://prettier.io/docs/en/)

---

**Last Updated**: 2025-11-18
**Maintained By**: Development Team

