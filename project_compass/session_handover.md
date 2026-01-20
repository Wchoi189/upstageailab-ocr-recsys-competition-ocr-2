# Session Handover: Surgical Core Audit 2 (The Witch Hunt Continues)

**Date**: 2026-01-21
**Previous Session**: Surgical Audit of `ocr/core` (Completed)
**Next Session Objective**: Surgical Audit of `ocr/data` & Architecture Decoupling

## 1. Executive Summary
The `ocr/core` directory has been surgically audited. Massive amounts of detection-specific logic (Losses, Metrics, Inference, Validation Schemas) were forcibly relocated to `ocr/domains/detection`. Generic data schemas were extracted to `ocr/core/data/schemas.py`.

**However, the war is not won.** The codebase suffers from **Critical Architecture Failure**:
1.  **Eager Import Hell**: Importing one model triggers a cascade that imports *every* model and architecture in the system.
2.  **Dataset Contamination**: `ocr/data/datasets` is hiding "criminals"—domain-specific preprocessing, heavy dependencies, and legacy logic that slows down startup and entangles domains.
3.  **Registry Anti-Pattern**: The current registry system likely requires eager importation of modules to "register" them, defeating the purpose of lazy loading.

## 2. Critical Architecture Feedback

### A. The "Import One, Import All" Registry Problem
**Diagnosis**: The current `ComponentRegistry` likely relies on Python decorators (`@registry.register`) which only run when the file is imported. To ensure components are available, `__init__.py` files (like `ocr/models/__init__.py`) act as "factories" that import *everything*.
**Impact**: Circular dependencies, 20s+ startup times, broken isolation.
**Solution**:
-   **Kill the Registry**: Deprecate `ocr.core.registry`. Use Hydra's `_target_` instantiation natively. If a registry is needed for string-based lookup, it MUST be configuration-based, not import-based.
-   **Lazy Loading**: Do not import `dbnet` or `parseq` until `instantiate(cfg)` is called.

### B. The `ocr/data` Quagmire
**Diagnosis**: `ocr/data/datasets` contains files like `advanced_detector.py` and `advanced_preprocessor.py`. These sound like **Detection Domain** logic, not generic dataset loading.
**Impact**: `ocr.data` depends on `scipy`, `cv2`, `numpy`, and likely `ocr.domains.detection`. This makes the data layer "heavy" and domain-coupled.
**Solution**:
-   **Strict Purity**: `ocr/data` should *only* contain:
    -   `LMDBLoader` (Generic)
    -   `ImageReader` (Generic)
    -   `BaseDataset` (Abstract)
-   **Move the Rest**:
    -   `advanced_detector.py` -> `ocr/domains/detection/data/preprocessing/`
    -   `augments.py` -> `ocr/domains/detection/data/transforms/`

### C. Eager Evaluation & Validation
**Diagnosis**: `ValidatedOCRDataset` likely validates strict Pydantic schemas *during `__init__`* or even at import time via global schema definitions.
**Impact**: Slow dataset initialization. If one schema import fails, the entire pipeline crashes.
**Solution**:
-   **Lazy Validation**: Validate only when `__getitem__` is called, or use a "Debug Mode" flag to disable expensive checks in production.
-   **Decoupled Schemas**: We started this with `ocr/core/data/schemas.py`. Continue separating "Transport Schemas" (simple dicts/dataclasses) from "Validation Logic" (Pydantic).

## 3. Road Map: Surgical Audit 2

### Phase 1: The Data Purge
- [ ] **Audit `ocr/data/datasets`**: Identify every file that imports `ocr.domains.*` or contains domain-specific keywords (Box, Polygon, Text Recognition).
- [ ] **Relocate**: Move these files to `ocr/domains/{domain}/data/`.
- [ ] **Refactor**: Rewrite `ocr/data/datasets/__init__.py` to export *nothing* by default.

### Phase 2: Registry Demolition
- [ ] **Audit `ocr/core/registry.py`**: Determine exact usage.
- [ ] **Trace Roots**: Find `ocr/models/__init__.py` or `ocr/core/__init__.py` files that eagerly import submodules.
- [ ] **Switch to Hydra**: Replace `@registry.register` with direct `_target_` paths in `.yaml` files.
- [ ] **Delete Registry**: Remove the registry system entirely if possible.

### Phase 3: The Firewall
- [ ] **Circular Dependency Check**: Run a script that attempts to import `ocr.domains.detection` without importing `ocr.domains.recognition`. If it fails, fix the leak.
- [ ] **Performance Profiling**: Measure import time. strict target: < 2s.

## 4. Immediate Next Steps for User
1.  **Review the new `ocr/core/data/schemas.py`**.
2.  **Approve the "Data Purge" plan**.
3.  **Prepare for breaking changes** as we rip out the Registry system.
