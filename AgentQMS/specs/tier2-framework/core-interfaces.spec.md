---
ads_version: '2.0'
id: 'FW-CORE-INTERFACES'
type: 'rule_set'
tier: 2
priority: 'high'
spec_version: '1.0.0'
updated: '2026-02-09'
description: 'Core/Interfaces layer for cross-domain data contracts'
---

# Core Interfaces Pattern

> Defines shared validation models and data contracts used across domain boundaries.

## Purpose

**Location**: `ocr/core/interfaces/`

The interfaces layer contains shared Pydantic models that serve as data contracts between:
- Core infrastructure modules (`ocr/core/`)
- Domain-specific modules (`ocr/domains/detection/`, `ocr/domains/recognition/`, etc.)

## Architecture Rules

```yaml
rule: CORE_INTERFACES_PATTERN
enforcement: architecture-guardian pre-commit hook

allowed_patterns:
  - "Core modules import from core/interfaces"
  - "Domain modules import from core/interfaces"
  - "Interfaces may reference domain concepts (polygons, maps) as data contracts"

prohibited_patterns:
  - "Core modules importing from domains"
  - "Cross-domain imports without interfaces"
```

## Shared Validation Models

**File**: `ocr/core/interfaces/validation_models.py`

### DataItem
Validated dataset sample from OCR pipeline. Used by:
- `ocr.core.utils.cache_manager` (caching layer)
- `ocr.domains.detection.validation` (re-exported for backward compatibility)

Fields: `image`, `polygons`, `metadata`, `prob_map`, `thresh_map`, `inverse_matrix`

### MapData
Cached probability/threshold maps. Used by:
- `ocr.core.utils.cache_manager` (map caching)
- Detection domain datasets

Fields: `prob_map`, `thresh_map`

### MetricConfig
CLEval metric configuration. Used by:
- `ocr.core.lightning.utils.config_utils` (metric extraction)
- Detection domain evaluation

Fields: metric thresholds, scale bins, polygon limits

## Migration Pattern

When moving models to interfaces:

1. Create model in `ocr/core/interfaces/validation_models.py`
2. Update core module imports: `from ocr.core.interfaces.validation_models import X`
3. Add re-export in domain: `from ocr.core.interfaces.validation_models import X`
4. Remove duplicate definition from domain

## Why This Matters

**Problem**: Core was importing from detection domain (CORE_PURITY violation)

**Solution**: Shared models moved to interfaces layer

**Benefit**: Clean dependency graph, domains can be swapped without affecting core
