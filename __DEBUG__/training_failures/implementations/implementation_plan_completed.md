# Refactoring Implementation Plan - OCR Pipeline

> **Status:** ✅ **Phase 1 Completed** (Refactoring & Compliance)
> **Active Focus:** 🔴 Critical Zero-Accuracy Issue (See `session_handover.md`)

## Goal Description
Address "High Cognitive Complexity" and "Production Code Clutter" in the OCR recognition pipeline.

**Completed Targets:**
1.  ✅ **Orchestrator**: Refactored `_inject_vocab_size` strategy.
2.  ✅ **Module**: Removed fragile `getattr` usage and debug prints.
3.  ✅ **Compliance**: Fixed all pre-commit violations.

**Pending / Future Work:**
4.  ⏳ **Architecture**: Remove hardcoded defaults (Deferred).
5.  ⏳ **Inference Strategy**: Refactor `generate()` (Deferred).

## User Review Required
> [!IMPORTANT]
> **Breaking Change Potential**: Defaults are being removed from `PARSeqDecoder` to force explicit configuration matching `parseq.yaml`. If custom scripts rely on these defaults being implicitly set, they may break. The standard `hydra` configs will be updated to match if needed (they should already match).

## Proposed Changes

### 1. Orchestrator Strategy Pattern
**File**: `ocr/pipelines/orchestrator.py`
-   Extract `_inject_vocab_size` into a dedicated `RecognitionConfigStrategy` helper.
-   Simplify `setup_modules` by delegating domain-specific setup.

### 2. Typed Configuration Wrapper
**File**: `ocr/domains/recognition/module.py`, `__DEBUG__/training_failures/scripts/inspect_predictions.py`
-   Create `RecognitionModelConfig` (Pydantic or simple class) to wrap `self.config`.
-   **Fix**: `cfg.global` is a SyntaxError in Python. Change to `cfg["global"]`.
-   Replace `getattr(getattr(self.config, "global", None), "debug", False)` with `self.cfg.debug` (requires wrapper).
-   Replace `print()` with `logger.debug()`.

### 5. Pre-commit & Compliance Fixes
-   **Artifacts**: Add strict frontmatter (YAML) to `docs/artifacts/bug_reports/2026-02-07_0419_bug_20260207_vocab-injection.md`.
-   **Scripts**: Remove `sys.path.append` from `scripts/debug_lmdb.py` (use `uv run` context instead).
-   **Hydra**: Investigate `configs/trainer` warning. It might need to be moved to `configs/training` or listed in allowed tiers.

## Verification Plan
**File**: `ocr/domains/recognition/models/decoder.py`, `architecture.py`
-   Remove default values for `d_model`, `nhead`, `num_layers` in `PARSeqDecoder.__init__`.
-   Update `PARSeq` to ensure `cfg` is strictly validated.

### 4. Inference Refactoring
**File**: `ocr/domains/recognition/models/architecture.py`
-   Separate `inference` logic from `forward`.
-   Introduce `generate(images, method="greedy" | "beam")`.
-   Move greedy loop out of `forward` (or keep `forward` for training only).

## Verification Plan

### Automated Verification
1.  **Configuration Test**:
    Create `__DEBUG__/training_failures/scripts/verify_instantiation.py` to instantiate components without training.
    ```bash
    python __DEBUG__/training_failures/scripts/verify_instantiation.py
    ```

2.  **Regression Training**:
    Run a short training loop to ensure data flow and device movement still works.
    ```bash
    uv run python scripts/runners/train.py \
      experiment=rec_baseline_v1 \
      trainer.limit_train_batches=10 \
      trainer.limit_val_batches=2 \
      trainer.max_epochs=1 \
      +hardware.accelerator=cpu
    ```
    *(Use CPU or GPU depending on availability, assuming GPU since user mentioned RTX 3090)*

### Manual Verification
-   Inspect logs to ensure "debug" prints are gone or properly logged.
-   Verify "Domain Mismatch" warnings are resolved (should have been fixed, but orchestrator refactor touches this).
