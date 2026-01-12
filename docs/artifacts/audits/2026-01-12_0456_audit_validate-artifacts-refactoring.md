---
ads_version: "1.0"
type: "audit"
category: "code_quality"
status: "draft"
version: "1.0"
tags: "None"
title: "Code Review: validate_artifacts.py Refactoring Opportunities"
date: "2026-01-12 04:56 (KST)"
branch: "main"
description: "Code review of validate_artifacts.py identifying refactoring opportunities"
---

# Code Review: validate_artifacts.py

## Overview

| Metric                  | Value                        |
| ----------------------- | ---------------------------- |
| Total lines             | 1244                         |
| ArtifactValidator class | Lines 128-1123 (~1000 lines) |
| Methods                 | 20+                          |
| Cognitive complexity    | **High**                     |

## Key Issues

### 1. Monolithic Class (Critical)
`ArtifactValidator` is a ~1000-line god class with too many responsibilities:
- Configuration loading
- Plugin extensions
- 7+ validation methods
- Report generation
- Fix suggestions
- Bundle validation

**Recommendation:** Split into focused modules:
```
validate_artifacts/
├── __init__.py           # Public API
├── validator.py          # Core ArtifactValidator (slimmed)
├── rules_loader.py       # Load from artifact_rules.yaml
├── validators/
│   ├── naming.py         # validate_naming_convention
│   ├── frontmatter.py    # validate_frontmatter
│   ├── directory.py      # validate_directory_placement
│   └── consistency.py    # validate_type_consistency
├── reporting.py          # generate_report, fix_suggestions
└── bundles.py            # validate_bundles (100 lines)
```

### 2. Hardcoded Value Lists (High)
Lines 130-180 contain `_BUILTIN_ARTIFACT_TYPES`, `_BUILTIN_TYPES`, etc. as class attributes.

**Current:**
```python
_BUILTIN_TYPES: list[str] = [
    "implementation_plan",
    "assessment",
    ...
]
```

**Better:** Load from `artifact_rules.yaml` (already exists!) or `frontmatter-master.yaml`:
```python
# In rules_loader.py
def load_valid_types() -> list[str]:
    rules = load_artifact_rules()
    return list(rules.get("artifact_types", {}).keys())
```

### 3. Long Methods
- `validate_bundles`: 100+ lines — extract to separate module
- `generate_report`: 70+ lines — extract to `reporting.py`
- `__init__`: 70+ lines — too much setup logic

### 4. Redundant Data
Value lists in code duplicate what's in:
- `artifact_rules.yaml` (the canonical source)
- `frontmatter-master.yaml` (we just created this!)

## Refactoring Priority

| Priority | Change                               | Effort | Impact |
| -------- | ------------------------------------ | ------ | ------ |
| 1        | Externalize builtin lists to YAML    | 1h     | High   |
| 2        | Extract `validate_bundles` to module | 30m    | Medium |
| 3        | Extract `reporting.py`               | 30m    | Medium |
| 4        | Split validators into separate files | 2h     | High   |

## Quick Win: Externalize Builtins

Replace hardcoded `_BUILTIN_*` with dynamic loading:

```python
# Load from existing artifact_rules.yaml
rules = load_artifact_rules()
self.valid_types = list(rules.get("artifact_types", {}).keys())
self.valid_statuses = rules.get("frontmatter", {}).get("valid_statuses", [])
self.valid_categories = rules.get("frontmatter", {}).get("valid_categories", [])
```

This already happens partially (lines 186-221) but the `_BUILTIN_*` fallbacks are never needed if `artifact_rules.yaml` is present.

## Recommendation

Start with **Priority 1**: Remove hardcoded `_BUILTIN_*` class attributes and rely solely on `artifact_rules.yaml`. The fallbacks add complexity without benefit since the YAML file is always present.
