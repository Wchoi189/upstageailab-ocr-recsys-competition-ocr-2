---
type: "audit"
category: "compliance"
status: "draft"
version: "1.0"
ads_version: "v1.0"
related_artifacts: []
generated_artifacts: []
tags: "architecture, audit, ocr, refactoring, separation-of-concerns"
title: "OCR Pipeline Architecture Audit: Post-Refactor Boundary Analysis"
date: "2026-01-09 02:17 (KST)"
branch: "main"
description: "Comprehensive audit of the OCR pipeline architecture evaluating the logical organization and separation of concerns between detection, recognition, and KIE features after the refactor execution."
---

# OCR Pipeline Architecture Audit: Post-Refactor Boundary Analysis

## Executive Summary

This audit evaluates the logical organization of the `ocr/` directory following the "Source Code Refactoring 2.0" execution. While the refactor successfully moved domain modules to `ocr/features/`, **significant boundary violations remain** that undermine the feature-first architecture goal.

> [!CAUTION]
> **Critical Finding**: The `ocr/core/` directory contains **domain-specific code** that violates separation of concerns. The distinction between "core" (shared infrastructure) and "features" (domain-specific logic) is unclear and inconsistently applied.

### Key Issues Identified

1. **KIE-specific code in `ocr/core/`** (Receipt extraction, KIE validation schemas, KIE-specific callbacks)
2. **Layout detection as infrastructure** (Should be a standalone feature)
3. **Ambiguous "core" definition** (Infrastructure vs. original detection pipeline)
4. **Feature-specific references scattered in shared modules**

---

## Background Context

### Previous Refactor Execution

From `session_handover_20260109_013004.md`, the refactor aimed to:
- **Phase 1**: Delete duplicates (`ocr/lightning_modules`, `ocr/datasets`, `ocr/metrics`)
- **Phase 2**: Move domains to `ocr/features/` (`detection`, `recognition`, `kie`)
- **Phase 3**: Split `ocr/inference` into core and feature-specific components

### Current Directory Structure

```
ocr/
├── core/                    # 🚨 AMBIGUOUS PURPOSE
│   ├── inference/
│   │   ├── extraction/      # ❌ KIE-specific (receipt extraction)
│   │   ├── layout/          # ❌ Should be a feature
│   │   └── *.py             # ✅ Core engines (orchestrator, engine)
│   ├── lightning/
│   │   ├── callbacks/
│   │   │   └── kie_wandb_image_logging.py  # ❌ KIE-specific
│   │   └── ocr_pl.py        # ⚠️  Mixed detection/recognition references
│   ├── kie_validation.py    # ❌ KIE-specific
│   ├── validation.py        # ✅ Truly shared
│   ├── metrics/             # ⚠️  Detection-heavy (CLEval)
│   ├── models/              # ✅ Shared architecture components
│   └── utils/               # ⚠️  Mixed concerns
├── features/
│   ├── detection/           # ✅ Clean feature boundary
│   ├── recognition/         # ✅ Clean feature boundary
│   └── kie/                 # ✅ Clean feature boundary
└── data/                    # ✅ Shared data handling
```

---

## Detailed Findings

### 🚨 **Critical Issue 1: KIE-Specific Code in `ocr/core/`**

#### Evidence

| File/Directory                                            | Purpose                                                             | Current Location | Correct Location                         |
| --------------------------------------------------------- | ------------------------------------------------------------------- | ---------------- | ---------------------------------------- |
| `ocr/core/inference/extraction/`                          | Receipt field extraction (regex patterns, line item parsing)        | `ocr/core/`      | `ocr/features/kie/inference/extraction/` |
| `ocr/core/kie_validation.py`                              | KIE data validation schemas (`KIEDataItem`, token-level validation) | `ocr/core/`      | `ocr/features/kie/validation.py`         |
| `ocr/core/lightning/callbacks/kie_wandb_image_logging.py` | KIE-specific WandB logging                                          | `ocr/core/`      | `ocr/features/kie/lightning/callbacks/`  |

#### Analysis

**`ocr/core/inference/extraction/`** contains:
- `field_extractor.py`: **425 lines** of receipt-specific regex patterns and heuristics
- `receipt_schema.py`: Domain models (`LineItem`, `ReceiptData`, `ReceiptMetadata`)
- `normalizers.py`: Date/phone/time normalization for KIE
- `vlm_extractor.py`: VLM-based extraction (KIE use case)

This is **NOT** infrastructure—it's a **KIE-specific feature**. The extraction module:
- Has zero usage outside KIE pipelines
- Contains business logic specific to receipt processing
- Depends on `layout/` (another misplaced feature)

**Recommendation**: Move entire `extraction/` to `ocr/features/kie/inference/extraction/`

---

### 🚨 **Critical Issue 2: Layout Detection Misclassified as Infrastructure**

#### Evidence

`ocr/core/inference/layout/` contains:
- `grouper.py`: Line grouping algorithms (7498 bytes)
- `contracts.py`: Layout-specific data models (`TextElement`, `TextLine`, `LayoutResult`)
- Purpose: "Layout detection module for OCR pipeline"

#### Analysis

**Layout detection is a FEATURE, not core infrastructure**:
- It's a distinct processing stage with its own algorithms
- Has specialized requirements (geometric analysis, line grouping)
- May have unique configuration needs
- Could have alternative implementations (rule-based vs ML-based)

**Current Problem**:
- Layout is **tightly coupled** to inference orchestration
- No clear boundary for layout-specific vs. general orchestration logic
- Prevents modular development/testing of layout algorithms

**Recommendation**:
1. Create `ocr/features/layout/` as a standalone feature
2. Define clear interfaces for layout detection in `ocr/core/`
3. Move `ocr/core/inference/layout/` → `ocr/features/layout/inference/`
4. Update orchestrator to use layout as a pluggable component

---

### ⚠️ **Issue 3: Ambiguous "Core" Definition**

#### The Problem

The term **"core"** is used inconsistently:

**Interpretation A: Shared Infrastructure** (Intended Goal)
- Base classes (`BaseEncoder`, `BaseDecoder`, etc.)
- Registry system
- Common validation schemas
- Inference orchestration engine

**Interpretation B: Detection Pipeline Legacy** (Actual State)
- Detection-heavy metrics (`CLEval` for text detection)
- Detection-specific postprocessing
- Hybrid inference engines that assume detection+recognition

#### Evidence of Confusion

From `ocr/core/metrics/`:
```python
# CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks
```
- CLEval is **detection-centric** (IoU matching, box evaluation)
- Used primarily for detection tasks
- Not truly "shared" across all features (KIE doesn't use CLEval)

From `ocr/core/lightning/ocr_pl.py`:
```python
"""OCR PyTorch Lightning Module for text detection and recognition."""
```
- Module is **detection+recognition specific**
- Not applicable to KIE tasks (which use `KIEPLModule` in `features/kie/`)

#### Recommendation

**Clarify the purpose of `ocr/core/`**:
1. **`ocr/core/` = Shared Infrastructure ONLY**
   - Base classes and interfaces
   - Component registry
   - Truly domain-agnostic utilities
   - Generic validation schemas

2. **Move domain-specific "core" to features**:
   - `ocr/core/lightning/ocr_pl.py` → `ocr/features/detection/lightning/` or `ocr/features/recognition/lightning/`
   - `ocr/core/metrics/` → `ocr/features/detection/metrics/` (since it's detection-focused)

3. **Document the definition**:
   - Add `ocr/core/README.md` with strict inclusion criteria
   - Use pre-commit hooks to enforce boundaries

---

### ⚠️ **Issue 4: Scattered Feature References in Shared Modules**

#### Evidence

Search results show feature-specific terminology in shared code:

**Detection references in `ocr/core/`**:
- `min_detection_size` parameters scattered across inference modules
- Detection-specific postprocessing in shared pipelines
- Contour-based detection fallbacks

**Recognition references in `ocr/core/`**:
- Crop extraction assumes recognition workflow
- Recognizer-specific orchestration logic

#### Analysis

**The orchestrator is too opinionated**:
- `ocr/core/inference/orchestrator.py` (24243 bytes) contains:
  - Detection-specific thresholds
  - Recognition-specific crop extraction
  - Hardcoded pipeline assumptions (detect → crop → recognize)

**This violates the Open/Closed Principle**:
- Adding new features (e.g., table extraction, formula recognition) requires modifying core
- Can't independently test/deploy features

#### Recommendation

1. **Extract detection logic** from orchestrator to `ocr/features/detection/inference/`
2. **Extract recognition logic** from crop_extractor to `ocr/features/recognition/inference/`
3. **Refactor orchestrator** to use a **plugin/registry pattern**:
   ```python
   # Pseudocode
   orchestrator.register_stage("detection", detection_feature)
   orchestrator.register_stage("recognition", recognition_feature)
   orchestrator.run(["detection", "recognition"])
   ```

---

## Architectural Assessment

### What "Core" Should Be

**✅ Correctly Classified as Core**:
- `ocr/core/base_classes.py`: Abstract interfaces
- `ocr/core/registry.py`: Component registration system
- `ocr/core/validation.py`: Generic data validation (not `kie_validation.py`)
- `ocr/core/models/`: Shared encoder/decoder/head architectures
- `ocr/core/inference/engine.py`: Model loading and inference orchestration
- `ocr/core/utils/config.py`, `path_utils.py`: Truly shared utilities

**❌ Incorrectly Classified as Core** (Should Be Features):
- `ocr/core/inference/extraction/`: **KIE-specific**
- `ocr/core/inference/layout/`: **Layout feature**
- `ocr/core/kie_validation.py`: **KIE-specific**
- `ocr/core/lightning/callbacks/kie_wandb_image_logging.py`: **KIE-specific**
- `ocr/core/metrics/`: **Detection-heavy** (should be in detection feature)
- `ocr/core/lightning/ocr_pl.py`: **Detection+Recognition** (not applicable to KIE)

### Feature Boundary Principles

To resolve the ambiguity, enforce these rules:

> **Rule 1: Core Must Be Domain-Agnostic**
> If code mentions "detection", "recognition", "KIE", "receipt", "layout" in its logic (not just as examples), it belongs in a feature.

> **Rule 2: Features Must Be Self-Contained**
> Each feature should have its own `models/`, `inference/`, `data/`, `lightning/`, `metrics/` as needed.

> **Rule 3: Zero Cross-Feature Dependencies**
> Features communicate only through core interfaces. No `from ocr.features.detection import ...` in recognition code.

> **Rule 4: Core Has No Business Logic**
> Core provides **mechanisms** (registries, base classes, orchestration), features provide **policies** (specific algorithms, models, metrics).

---

## Recommendations

### 🎯 **Immediate Actions** (High Priority)

1. **Move KIE-specific code to `ocr/features/kie/`**
   ```
   ocr/core/inference/extraction/     → ocr/features/kie/inference/extraction/
   ocr/core/kie_validation.py         → ocr/features/kie/validation.py
   ocr/core/lightning/callbacks/kie_* → ocr/features/kie/lightning/callbacks/
   ```

2. **Extract layout as a standalone feature**
   ```
   ocr/core/inference/layout/ → ocr/features/layout/inference/
   ```

3. **Document `ocr/core/` purpose**
   - Create `ocr/core/README.md` with strict inclusion criteria
   - Add architecture decision record (ADR) for feature boundaries

### 🔧 **Medium-Term Refactoring** (Medium Priority)

4. **Refactor orchestrator for plugin architecture**
   - Define `FeatureStage` interface in `ocr/core/`
   - Move detection/recognition orchestration to respective features
   - Use registry pattern for feature discovery

5. **Relocate detection-specific metrics**
   ```
   ocr/core/metrics/ → ocr/features/detection/metrics/
   ```

6. **Split hybrid Lightning modules**
   - Keep generic `BasePLModule` in core
   - Move `ocr_pl.py` to `ocr/features/detection/` or shared by detection/recognition

### 📋 **Long-Term Improvements** (Lower Priority)

7. **Add architectural guardrails**
   - Pre-commit hook to enforce feature boundaries
   - Import analyzer to detect cross-feature dependencies
   - CI checks for "core" purity (no domain-specific keywords)

8. **Standardize feature structure**
   ```
   ocr/features/{feature_name}/
   ├── __init__.py
   ├── models/           # Feature-specific models
   ├── data/             # Feature-specific datasets
   ├── inference/        # Feature-specific inference logic
   ├── lightning/        # Feature-specific trainers
   ├── metrics/          # Feature-specific evaluation
   └── README.md         # Feature documentation
   ```

9. **Create feature contracts**
   - Define standard interfaces each feature must implement
   - Document feature lifecycle (registration, initialization, execution)

---

## Migration Impact Analysis

### Files to Move

| Source                                                    | Destination                              | Impact                                     | LOC    |
| --------------------------------------------------------- | ---------------------------------------- | ------------------------------------------ | ------ |
| `ocr/core/inference/extraction/`                          | `ocr/features/kie/inference/extraction/` | **High** - Update all KIE imports          | ~1,500 |
| `ocr/core/inference/layout/`                              | `ocr/features/layout/inference/`         | **High** - Update orchestrator, extraction | ~800   |
| `ocr/core/kie_validation.py`                              | `ocr/features/kie/validation.py`         | **Low** - Few external references          | ~70    |
| `ocr/core/lightning/callbacks/kie_wandb_image_logging.py` | `ocr/features/kie/lightning/callbacks/`  | **Low** - Isolated callback                | ~180   |
| `ocr/core/metrics/`                                       | `ocr/features/detection/metrics/`        | **Medium** - Used in training/eval         | ~2,000 |

### Import Update Scope

**Estimated affected files**: 15-25 files
**Tooling**: Use AST-based refactoring tools (e.g., `bowler`, `rope`) for safe import updates

### Testing Requirements

1. **Unit tests**: Update imports, verify all tests pass
2. **Integration tests**: Verify end-to-end pipelines (detection+recognition, KIE)
3. **Configuration tests**: Ensure Hydra configs resolve correctly after moves

---

## Conclusion

The "Source Code Refactoring 2.0" successfully **containerized features** but **failed to clarify the core/feature boundary**. The `ocr/core/` directory contains:
- ✅ True shared infrastructure (base classes, registry)
- ❌ KIE-specific code (extraction, validation, callbacks)
- ❌ Layout detection (a misclassified feature)
- ⚠️  Detection-heavy code (metrics, Lightning modules)

### The Root Cause

**Lack of explicit definition**: The refactor moved directories without defining **what qualifies as "core"**. The original codebase treated detection as the default pipeline, so detection-adjacent code was never explicitly labeled.

### The Path Forward

1. **Define "core"** strictly as domain-agnostic infrastructure
2. **Move all feature-specific code** to `ocr/features/`
3. **Enforce boundaries** with tooling and documentation
4. **Refactor orchestration** to support pluggable features

### Success Criteria

After implementing recommendations, verify:
- [ ] `grep -r "receipt\|kie\|detection\|recognition" ocr/core/` returns only generic references (no logic)
- [ ] Each feature is independently testable
- [ ] CI enforces feature isolation (no cross-feature imports)
- [ ] New features can be added without modifying `ocr/core/`

---

## Appendix: File Classification Matrix

### `ocr/core/` File Audit

| File/Directory              | Classification          | Reasoning                            | Action                            |
| --------------------------- | ----------------------- | ------------------------------------ | --------------------------------- |
| `base_classes.py`           | ✅ Core                  | Abstract interfaces for all features | Keep                              |
| `registry.py`               | ✅ Core                  | Component registration system        | Keep                              |
| `validation.py`             | ✅ Core                  | Generic data validation              | Keep                              |
| `kie_validation.py`         | ❌ KIE Feature           | KIE-specific schemas                 | **Move to `features/kie/`**       |
| `inference/engine.py`       | ✅ Core                  | Model loading orchestration          | Keep (refactor for plugins)       |
| `inference/orchestrator.py` | ⚠️  Hybrid               | Mixed detection/recognition logic    | **Refactor to plugin pattern**    |
| `inference/extraction/`     | ❌ KIE Feature           | Receipt extraction logic             | **Move to `features/kie/`**       |
| `inference/layout/`         | ❌ Layout Feature        | Layout detection algorithms          | **Move to `features/layout/`**    |
| `lightning/ocr_pl.py`       | ❌ Detection+Recognition | Not applicable to KIE                | **Move to feature or split**      |
| `lightning/callbacks/kie_*` | ❌ KIE Feature           | KIE-specific logging                 | **Move to `features/kie/`**       |
| `metrics/`                  | ❌ Detection Feature     | CLEval is detection-focused          | **Move to `features/detection/`** |
| `models/`                   | ✅ Core                  | Shared architecture components       | Keep                              |
| `utils/config.py`           | ✅ Core                  | Configuration utilities              | Keep                              |
| `utils/wandb_utils.py`      | ✅ Core                  | Generic logging utilities            | Keep                              |

---

**Audit Date**: 2026-01-09
**Auditor**: Antigravity AI Agent
**Session**: OCR Architecture Post-Refactor Assessment
