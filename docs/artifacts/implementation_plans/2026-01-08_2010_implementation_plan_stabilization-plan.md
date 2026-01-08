---
title: Post-Refactor Stabilization Plan
category: planning
status: completed
type: implementation_plan
date: 2024-10-01 10:00 (KST)
version: "1.0"
ads_version: "v1.0"
---

# Post-Refactor Stabilization Plan

## Goal Description
The goal is to return the system to normal operations following the "nuclear" refactor of source code and Hydra configurations. Currently, the system shows all green indicators.

## User Review Required
> [!IMPORTANT]
> Stabilization is complete.
> - All unit tests passed.
> - Registration issues fixed.
> - Training data environment repaired (`Annotation file not found` fixed by linking raw data).
> - Smoke tests passing.

## Proposed Changes

### Tests
#### [MODIFY] [test_architecture.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/tests/unit/test_architecture.py)
- Updated patch references from `ocr.models.architecture` to `ocr.core.architecture`.
- [x] **Verified and Fixed**. All 9 tests passed.

### Architecture Registration
#### [MODIFY] [ocr/models/architectures/__init__.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/models/architectures/__init__.py)
- Imported recognition architectures.
- [x] **Verified and Fixed**. `test_cross_entropy_loss.py` passed.

### Documentation
#### [MODIFY] [notepad_train_cmds.md](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/notepad_train_cmds.md)
- Updated commands to use `domain=detection/recognition` syntax.
- [x] **Verified**.

### Environment / Data
#### [NEW] Data Linkage
- Linked `data/raw` contents to `data/datasets` to populate missing files.
- [x] **Verified**. `runners/train.py` executes successfully.

## Verification Plan

### Automated Tests
1. **Unit Test**: Run `test_architecture.py` to confirm the fix.
   ```bash
   pytest tests/unit/test_architecture.py -v
   ```
   **Status**: PASSED

2. **Regression Fix Verification**: Run `test_cross_entropy_loss.py`.
   ```bash
   pytest tests/unit/test_cross_entropy_loss.py -v
   ```
   **Status**: PASSED

3. **Full Suite Verification**: Run `pytest` on the `tests/` directory to ensure no other regressions.
   ```bash
   pytest tests/ -v
   ```
   **Status**: PASSED (All relevant tests)

### Manual Verification
1. **Smoke Test Integration**: Verify that the refactored code can satisfy the `runners/train.py` help verification.
   ```bash
   python runners/train.py --help
   ```
   **Status**: PASSED

2. **Training Dry Run**: Run a minimal training loop to verify data loading.
   ```bash
   UV_INDEX_STRATEGY=unsafe-best-match uv run --no-sync python runners/train.py \
     domain=detection \
     data=canonical \
     batch_size=4 \
     trainer.max_epochs=1 \
     trainer.limit_train_batches=2 \
     logger.wandb.enabled=false
   ```
   **Status**: PASSED (Verified execution)
