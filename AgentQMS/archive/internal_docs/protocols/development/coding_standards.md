# Coding Standards & Naming Conventions

**Document ID**: `PROTO-DEV-001`
**Status**: ACTIVE
**Type**: Development Protocol

---

## TL;DR

- Format with Ruff (line length 88).
- Use framework path utilities, never manipulate `sys.path`.
- Follow PEP8 naming: snake_case for modules/functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants.

---

## Code Formatting

- Use Ruff Formatter (line length 88)
- Auto-format: `uv run ruff check . --fix && uv run ruff format .`

---

## Import Organization

- Three groups: standard library, third-party, local
- One import per line, alphabetical within groups

---

## Path Management

- **ALWAYS** use framework path utilities for path setup
- **NEVER** manually manipulate `sys.path`

```python
# ✅ CORRECT
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path
ensure_project_root_on_sys_path()

# ❌ WRONG
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
```

See: [Import Handling Protocol](import_handling_protocol.md)

---

## Pydantic V2 Patterns

```python
from pydantic import BaseModel, Field, field_validator, model_validator

class Model(BaseModel):
    field: str = Field(..., constraints...)

    @field_validator('field')
    @classmethod
    def validate_field(cls, v): return v

    @model_validator(mode='after')
    def validate_cross_fields(self): return self
```

### Serialization

- `model.model_dump()` for dict
- `model.model_dump_json()` for JSON
- `ModelClass.model_validate(data)` for dict
- `ModelClass.model_validate_json(json_data)` for JSON

---

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules/Packages | snake_case | `artifact_workflow` |
| Classes | PascalCase | `ArtifactValidator` |
| Functions/Methods | snake_case | `validate_artifacts` |
| Variables | snake_case | `artifact_root` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_CONFIG` |

---

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict .
```

