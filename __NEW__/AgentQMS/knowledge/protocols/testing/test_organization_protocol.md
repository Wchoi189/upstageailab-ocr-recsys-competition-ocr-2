# Test Organization Protocol

**Status**: ACTIVE
**Type**: Testing Protocol

---

## TL;DR

- All tests go in `tests/` directory, never in project root.
- Use subdirectories: `unit/`, `integration/`, `components/`, `data/`.
- Name files: `test_<module_name>.py`.
- Name functions: `test_<function>_<scenario>`.

---

## Critical Rules

### ❌ Never Place Tests In

- Project root directory
- `scripts/` folder
- `src/` or module folders

### ✅ Always Place Tests In

- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests
- `tests/components/` - Component tests
- `tests/data/` - Data validation tests
- `tests/manual/` - Manual/exploratory tests

---

## Directory Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── unit/
│   ├── test_data_loaders.py
│   └── test_validators.py
├── integration/
│   ├── test_full_pipeline.py
│   └── test_api_integration.py
├── components/
│   └── test_ui_elements.py
└── data/
    └── test_data_validation.py
```

---

## Naming Conventions

### Files

- Unit: `test_<module_name>.py`
- Integration: `test_<feature>_integration.py`
- Component: `test_<component_name>.py`

### Functions

- Unit: `test_<function_name>_<scenario>`
- Integration: `test_<feature>_<outcome>`

```python
# tests/unit/test_data_loaders.py
def test_load_csv_file_success():
def test_load_csv_file_invalid_path():

# tests/integration/test_api_integration.py
def test_full_prompt_submission_flow():
```

---

## Checklist

- [ ] Test type identified (unit/integration/component/data)
- [ ] Correct directory chosen within `tests/`
- [ ] File named according to convention
- [ ] Test follows AAA pattern (Arrange-Act-Assert)
- [ ] Test is isolated

