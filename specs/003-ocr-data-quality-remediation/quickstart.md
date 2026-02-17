# Quickstart: OCR Data-Quality Remediation (Planning-First)

## Session Policy

- This feature is currently in planning mode only.
- Do not execute remediation training/filtering runs in this session.
- Use this runbook to prepare the next execution session.

## 1) Emergency Actions (Any Session)

If active training is running with suspected corrupted GT:

1. Stop immediately (`Ctrl+C` in runner terminal).
2. Preserve latest checkpoint in `outputs/checkpoints/emergency_backup/`.
3. Record incident note in ETK experiment `.metadata/00-status/`.
4. Continue only after remediation review and gate reset.

## 2) Validate Planning Artifacts

```bash
uv run python AgentQMS/tools/compliance/validate_artifacts.py --all
```

## 3) Run Non-Mutating Diagnostics (Planning Allowed)

```bash
python scripts/audit/analyze_defect_distribution.py \
  --run_id <run_id> \
  --output data/audit/defect_prevalence.json

python scripts/audit/compute_loss_distribution.py \
  --manifest data/processed/recognition/train_manifest.jsonl \
  --output data/audit/loss_percentiles.json

python scripts/audit/analyze_sequence_lengths.py \
  --manifest data/processed/recognition/train_manifest.jsonl \
  --tokenizer_max_len 25 \
  --output data/audit/truncation_analysis.json
```

## 4) Initialize Experiment Workspace (done in planning session)

```bash
cd dev_tools/experiment_manager
uv run etk init ocr_data_quality_remediation --type custom --intention "Plan-first OCR label-noise remediation and controlled testing"
```

## 5) Deferred Execution Commands (next session only)

```bash
# Build filtered manifest (example)
uv run python scripts/data/build_filtered_manifest.py \
  --input data/processed/recognition/train_manifest.jsonl \
  --output data/processed/recognition/train_manifest.filtered.jsonl \
  --drop-script-mismatch --drop-clipping-risk --drop-illegible --max-loss-percentile 95

# Controlled continuation training (example)
uv run python scripts/runners/train.py \
  mode=train \
  experiment=parseq_flash_plateau_images \
  +checkpoint_path=outputs/checkpoints/best-acc-0.8372_v2.ckpt \
  data.sequence.tokenizer_max_len=32 \
  recognition.max_label_length=32 \
  trainer.val_check_interval=1.0 \
  +train.logger.wandb.high_loss_audit.enabled=true
```

## 6) Gate Metrics for Execution Phase

- Clean holdout CER/WER (mandatory)
- Annotation consistency rate
- Truncation rate (`len(gt) >= max_len-2`)
- High-loss defect purity in top-k audited samples

## 7) Golden Validation Dataset Strategy (Execution Planning)

Tiered validation workflow for 200-500 clean holdout candidates:

1. Tier 1: model-confidence triage (auto-accept obvious clean matches)
2. Tier 2: local validator (`PaddleOCR`) for low-cost filtering
3. Tier 3: Upstage OCR API verification for ambiguous/high-risk samples
4. Manual review only for unresolved/low-confidence disagreements

Guidelines:
- Keep validation patch-level (not full-document) for GT attribution.
- Use confidence thresholds to reduce API cost.
- Version and freeze the final clean holdout manifest after verification.

Suggested confidence policy:
- `>= 0.95` + GT match: auto-accept
- `>= 0.98` + GT mismatch: auto-correct candidate
- `< 0.95`: manual review queue

Suggested holdout size policy:
- Start with 200-500 samples
- Require minimum 200 verified samples for gate-eligible clean holdout
