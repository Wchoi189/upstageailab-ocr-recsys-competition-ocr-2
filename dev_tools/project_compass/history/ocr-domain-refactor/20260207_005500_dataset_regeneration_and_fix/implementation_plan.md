# Implementation Plan - Dataset Regeneration

## Goal Description
Regenerate the corrupted `aihub_lmdb_validation` dataset from the raw source files located in `data/raw/external`. This involves extracting the `.zip.tar` archives and running the existing ETL pipeline to create a fresh LMDB database.

## User Review Required
> [!WARNING]
> **Data Overwrite:**
> This process will Replace the existing (corrupt) dataset at `/workspaces/data/processed/recognition/aihub_lmdb_validation`. The corrupt version will be moved to `_corrupt` suffix as a backup just in case.

> [!IMPORTANT]
> **Disk Space:**
> The operation requires extracting ~8GB of raw data and generating an LMDB of similar size. Ensure sufficient disk space is available (approx 20GB buffer recommended).

## Proposed Changes

### 1. Data Extraction
*   **Source:** `/workspaces/data/raw/external/`
    *   `[라벨]validation_32mb.zip.tar`
    *   `[원천]validation_7.47gb.zip.tar`
*   **Destination:** `/workspaces/data/interim/recognition/aihub_validation_raw`
*   **Action:** Extract tar archives and then unzip the inner contents if nested.

### 2. ETL Script Fixes
#### [MODIFY] [cli.py](file:///workspaces/scripts/data/etl/cli.py)
*   Fix the commented-out import of `LMDBConverter`.
*   Ensure the script is runnable via `uv run`.

### 3. LMDB Generation
*   **Command:**
    ```bash
    uv run python scripts/data/etl/cli.py convert \
        --input-dir /workspaces/data/interim/recognition/aihub_validation_raw \
        --output-dir /workspaces/data/processed/recognition/aihub_lmdb_validation \
        --num-workers 4
    ```

## Verification Plan

### Automated Tests
#### 1. Inspect Generated LMDB
Use the `inspect` command in the CLI to verify read access and sample counts:
```bash
uv run python scripts/data/etl/cli.py inspect --lmdb-path /workspaces/data/processed/recognition/aihub_lmdb_validation
```
*   **Success:** Prints total samples and displays valid metadata for the first 5 samples without crashing.

#### 2. Re-run Phase 3 Verification
Execute the training script again with `num_workers=0` (safety) then `num_workers=4`:
```bash
uv run python scripts/runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True
```
*   **Success:** Training loop starts and completes `fast_dev_run` without SIGBUS.
