# Performance Audit: Startup & Initialization
**Date**: 2026-01-22
**Context**: Post-Surgical Core Audit 2

## Execution Candidates

### P1: Orchestrator Eager Imports
- **File**: `ocr/pipelines/orchestrator.py`
- **Line(s)**: 12
- **Issue**: Eager import of `lightning.pytorch`.
- **Suspected Impact**: **Startup**. PyTorch Lightning is a heavy dependency. Importing it at module level slows down CLI tools that might only need to parse config.
- **Recommended Action**: Defer import to `setup_trainer` or `run` method.

### P2: Train Script Entry Point
- **File**: `runners/train.py`
- **Line(s)**: 5
- **Issue**: Eager import of `OCRProjectOrchestrator`.
- **Suspected Impact**: **Startup**. Immediate cascading import of `ocr.core`, `ocr.data`, and `lightning`.
- **Recommended Action**: Move import inside `train()` function.

### P3: Hydra Config Instantiation
- **File**: `ocr/core/models/__init__.py` (and usage in `orchestrator.py`)
- **Line(s)**: 4-24
- **Issue**: `get_model_by_cfg` performs imports inside, but `orchestrator.py` calls it early.
- **Suspected Impact**: **Startup/Runtime**.
- **Recommended Action**: Ensure `get_model_by_cfg` usage is deferred until absolutely necessary in the pipeline.

### P4: Dataset Factory Dependencies
- **File**: `ocr/data/datasets/__init__.py`
- **Line(s)**: 38 (get_datasets_by_cfg)
- **Issue**: While imports inside are lazy, the function is called during `setup_modules`.
- **Suspected Impact**: **Runtime**. If `instantiate` triggers heavy initializers, it blocks `setup`.
- **Recommended Action**: Verify that dataset classes (`ValidatedOCRDataset`) do not perform heavy work in `__init__`.

### P5: Detection Preprocessing
- **File**: `ocr/domains/detection/data/preprocessing/advanced_detector.py`
- **Line(s)**: 9-11
- **Issue**: Imports `cv2`, `numpy`, `scipy`.
- **Suspected Impact**: **Startup** (if imported eagerly).
- **Recommended Action**: Ensure this module is ONLY imported when detection domain is active and running.

## Training Pipeline Observations
- **Symptom**: "Training run initialization: >60 seconds"
- **Hypothesis**: The combination of `runners/train.py` -> `orchestrator.py` -> `lightning` + `hydra` overhead + potentially unoptimized `__init__` in dataset/model classes causes the delay.
- **Next Step**: Profile imports using `python -X importtime runners/train.py ...`.
