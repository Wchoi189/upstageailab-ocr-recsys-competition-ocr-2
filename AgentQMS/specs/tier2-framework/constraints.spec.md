---
ads_version: '2.0'
id: 'FW-CONSTRAINTS.SPEC'
type: 'rule_set'
tier: 2
priority: 'high'
spec_version: '1.0.0'
updated: '2026-02-03'
description: 'Constraints Specification for framework tier'
---

# Constraints Specification

**Tier**: 2 (Framework)
**Scope**: Performance, Testing, and Validation Limits.

## 1. Performance SLAs
*   **Inference Latency**: < 200ms (p95) for Single Page OCR.
*   **Startup Time**: < 3s (Cold Start).
*   **Memory**: < 4GB VRAM per worker.

## 2. Testing Standards
*   **Unit Tests**: Standard `pytest`. Must run in < 5s total.
*   **Integration Tests**: Test full pipeline flow.
*   **Coverage**: Target > 80% for Core Infra.

### Pydantic Validation
*   **Usage**: Use Pydantic V2 for all data inputs (API requests).
*   **Strict**: Enable `strict=True` for type coercion prevention where possible.

## 3. Bloat Detection
*   **File Size**: Warn if > 300 lines (Python) or > 600 tokens (Markdown Spec).
*   **Complexity**: McCobe < 15 per function.
