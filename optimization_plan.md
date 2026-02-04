# Optimization Plan: Import Latency & Environment Hangs

## Goal Description
Address severe startup timeouts (>60s) and import hangs encountered during Phase 3 verification. The primary objective is to reduce import time by implementing lazy loading for heavy dependencies and pruning unnecessary top-level imports.

## Diagnostic Steps (Session Start)
1.  **Generate Import Profile**:
    ```bash
    uv run python -X importtime runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True ~train.logger.wandb_logger 2> import.log
    ```
2.  **Visualize with Tuna**:
    ```bash
    uv pip install tuna
    tuna import.log
    ```
    *Deliverable*: Screenshot or summary of the Icicle chart identifying the exact bottlenecks (confirming `torchmetrics` and `lightning` hypotheses).

## Proposed Optimizations

### 1. Fix `torchmetrics` Bottleneck
**Hypothesis**: `ocr.core` imports `torchmetrics`, which cascades into `transformers` (BERT) and `tokenizers` C++ compilation.
-   **Action**: Audit `ocr/core/interfaces/` and `ocr/core/metrics/`.
-   **Refactor**: Move heavy imports specific to text metrics (e.g., BREDS, PER, WER if they use heavy backends) inside the metric calculation functions or wrapped in `TYPE_CHECKING` blocks.

### 2. Optimize Lightning Imports
**Hypothesis**: `lightning.pytorch` imports massive strategy suites (DeepSpeed, XLA) by default.
-   **Action**: Check `ocr/pipelines/orchestrator.py` and `runners/train.py`.
-   **Refactor**: Ensure we are not importing `lightning` components at the module level unless necessary. Use `setup_trainer` methods to import specific callbacks/loggers only when needed.

### 3. Wildcard Import Audit
**Hypothesis**: `ocr/core/__init__.py` or similar files might be using `from .module import *`, triggering recursive loading.
-   **Action**: Scan all `__init__.py` files in `ocr/`.
-   **Refactor**: Replace `*` imports with explicit `__all__` definitions or remove them if simpler usage is preferred.

### 4. Scientific Stack Deferral
**Hypothesis**: `scipy`, `sklearn`, `matplotlib` are loading early.
-   **Action**: Identify where these are imported (likely in dataset visualization or metric utilities).
-   **Refactor**: Move to local scope (e.g., inside `visualize_batch` functions).

## Verification Success Criteria
-   `fast_dev_run` initiates within <15 seconds.
-   No `importtime` log shows any single module taking >5s to load.
