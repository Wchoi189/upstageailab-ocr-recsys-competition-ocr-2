# Implementation Plan: Recognition Pipeline Repair

## Goal
Restore the Recognition pipeline to a functional state by implementing missing V5 atomic configurations for `head` and `loss`, and fixing V5 compliance gaps.

---

## User Review Required
> [!IMPORTANT]
> **Decision Needed:** Which specific class to use for `head`?
> [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) uses a simple linear projection `nn.Linear(d_model, vocab_size + 1)`.
> We need to either:
> A. Create a `LinearHead` class in `ocr/core/models/head` (or domain specific)
> B. Use `torch.nn.Linear` directly as the target? (Hydra allows this)
>
> **Decision Needed:** Which class for `loss`?
> `CrossEntropyLoss` is standard. `PARSeq` legacy might have used a custom wrapper.
> We will assume `torch.nn.CrossEntropyLoss` for atomic V5 compliance unless a specific `PARSeqLoss` is found.

---

## Proposed Changes

### 1. Configuration Fixes

#### [MODIFY] `configs/domain/recognition.yaml`
- **Fix:** Ensure it exports a proper dictionary structure, matching detection.
- **Action:** Verify `_group_` behavior and fix logic if needed.

#### [MODIFY] `configs/experiment/rec_baseline_v1.yaml`
- **Fix:** Remove `domain: recognition` string override which destroys the domain struct.
- **Action:** Delete line `domain: recognition`.

#### [MODIFY] `configs/model/architectures/parseq.yaml`
- **Fix:** Add missing `head` and `loss` components.
- **Structure:**
  ```yaml
  head:
    _target_: torch.nn.Linear
    in_features: 512
    out_features: ${model.vocab_size} # +1 for EOS? Need to check PARSeq logic
  loss:
    _target_: torch.nn.CrossEntropyLoss
    ignore_index: 0 # Check padding token id
  ```

### 2. Code Adjustments (If needed)

#### [CHECK] `ocr/domains/recognition/models/architecture.py`
- Verify if `PARSeq` expects `vocab_size` to include special tokens.
- Verify if `nn.Linear` is sufficient (signatures must match).

---

## Verification Plan

### Automated Verification
1.  **Dry Run Re-Test:**
    - Run `scripts/audit/dry_run_parseq.py`.
    - **Expectation:** Forward pass SUCCEEDS.
2.  **Fast Dev Run:**
    - Run `uv run python runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True`.
    - **Expectation:** Training loop starts (no optimizer error, no NoneType error).

### Manual Verification
- Inspect WandB/logs for loss values (should not be NaN).
