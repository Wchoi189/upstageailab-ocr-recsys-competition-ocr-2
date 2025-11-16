---
title: "{{ title }}"
author: "{{ author }}"
timestamp: "{{ timestamp }}"
branch: "{{ branch }}"
status: "draft"
tags: ["data-contract"]
---

## 1. Overview

**Purpose:** Define the data contract for {{ area_name }}.

**Scope:** {{ scope_description }}

**Key Components:**
- Input data structures and validation
- Output data structures and validation
- Field constraints and relationships
- Error handling expectations

---

## 2. Input Contract

### 2.1 Data Structure

**Primary Input Type:** `{{ input_type }}`

```python
# Example input structure
{{ input_example }}
```

### 2.2 Field Definitions

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `field1` | `type1` | Yes/No | `constraints` | Description |
| `field2` | `type2` | Yes/No | `constraints` | Description |

### 2.3 Validation Rules

- **Field-level validation:**
  - Rule 1: Description
  - Rule 2: Description

- **Cross-field validation:**
  - Rule 1: Description
  - Rule 2: Description

---

## 3. Output Contract

### 3.1 Data Structure

**Primary Output Type:** `{{ output_type }}`

```python
# Example output structure
{{ output_example }}
```

### 3.2 Field Definitions

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `field1` | `type1` | Yes/No | `constraints` | Description |
| `field2` | `type2` | Yes/No | `constraints` | Description |

### 3.3 Validation Rules

- **Field-level validation:**
  - Rule 1: Description
  - Rule 2: Description

- **Cross-field validation:**
  - Rule 1: Description
  - Rule 2: Description

---

## 4. Validation Rules

### 4.1 Input Validation

**Pre-processing checks:**
- Check 1: Description
- Check 2: Description

**Type validation:**
- Type check 1: Description
- Type check 2: Description

### 4.2 Output Validation

**Post-processing checks:**
- Check 1: Description
- Check 2: Description

**Consistency checks:**
- Consistency check 1: Description
- Consistency check 2: Description

### 4.3 Error Handling

**Validation failures:**
- Error type 1: Description and handling
- Error type 2: Description and handling

**Edge cases:**
- Edge case 1: Description and handling
- Edge case 2: Description and handling

---

## 5. Examples

### 5.1 Valid Examples

**Example 1: Standard Case**
```python
# Input
input_data = {
    # ... example input
}

# Expected Output
output_data = {
    # ... example output
}
```

**Example 2: Edge Case**
```python
# Input
input_data = {
    # ... example input
}

# Expected Output
output_data = {
    # ... example output
}
```

### 5.2 Invalid Examples

**Example 1: Missing Required Field**
```python
# Invalid Input
input_data = {
    # ... missing required field
}

# Expected Error
ValidationError: "Field 'field_name' is required"
```

**Example 2: Type Mismatch**
```python
# Invalid Input
input_data = {
    "field_name": "wrong_type"  # Expected: int, Got: str
}

# Expected Error
ValidationError: "Field 'field_name' must be of type int, got str"
```

---

## 6. Implementation Notes

### 6.1 Pydantic Models (if applicable)

```python
from pydantic import BaseModel, Field, field_validator

class InputContract(BaseModel):
    """Input contract model."""
    # ... field definitions

class OutputContract(BaseModel):
    """Output contract model."""
    # ... field definitions
```

### 6.2 Testing Requirements

- Unit tests for valid inputs
- Unit tests for invalid inputs
- Integration tests for end-to-end validation
- Performance tests for large inputs

### 6.3 Migration Considerations

- Backward compatibility requirements
- Deprecation timeline (if applicable)
- Migration path for existing code

---

## 7. References

- Related data contracts: [links]
- Implementation code: [links]
- Test files: [links]
- Documentation: [links]

