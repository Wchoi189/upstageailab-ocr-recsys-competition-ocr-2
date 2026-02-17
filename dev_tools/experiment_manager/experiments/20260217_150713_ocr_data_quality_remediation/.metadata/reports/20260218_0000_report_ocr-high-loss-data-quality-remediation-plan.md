# OCR High-Loss Data-Quality Remediation — Technical Planning Report

**Date**: 2026-02-18
**Experiment**: `20260217_150713_ocr_data_quality_remediation`
**Feature Spec**: `specs/003-ocr-data-quality-remediation/spec.md`
**Status**: Planning Complete (Execution Deferred)

---

## 1. Problem Statement

Observed high-loss OCR samples show systematic annotation defects (not only difficult-but-valid samples), causing supervision contamination. The critical risk is incorrect ground truth propagation; reduced short-term performance is acceptable during remediation, but incorrect labels are unacceptable.

## 2. Audit Findings Summary

Primary defect classes confirmed in high-loss examples:
- Script mismatch (`605` image vs GT `6악`)
- Missing characters due to clipping
- Hallucinated GT characters (`번지의` image vs GT `번지dh`)
- Unreadable/noisy scrawl samples
- Truncation and sequence misalignment near max token length

## 3. Defect Taxonomy and Mitigation Mapping

| Defect | Technical Term | Mitigation |
|---|---|---|
| Wrong-script GT | Cross-script label corruption | Rule-based exclusion + relabel queue |
| GT includes unseen chars | Crop-label sequence misalignment | Re-crop or GT trim; otherwise drop |
| GT hallucinated chars | Annotation drift / hallucinated supervision | Human-verified correction queue |
| Unreadable handwriting/noise | Low-SNR supervision | Exclude from primary training set |
| Full GT with clipped patch | Truncation-label inconsistency | Clip-aware filtering and partial-label policy |

## 4. Data Strategy Decisions

### 4.1 Filtering Strategy
- Three-lane triage: `drop` (deterministic defects), `review` (uncertain/high-loss), `keep` (clean).
- Combine deterministic rules and loss percentile; avoid loss-only filtering.

### 4.2 Semi-Supervised GT Correction
- Use model consensus + rule checks to prioritize likely GT errors.
- Mandatory human verification before applying corrected labels.

### 4.3 Synthetic Data Augmentation
- Hybrid approach selected: high-confidence real data as anchor + synthetic augmentation for coverage.
- Candidate tools: SynthDoG, TRDG.

## 5. Multi-Phase Plan and Gates

### Phase 0 (Completed in planning)
- Baseline diagnostics, taxonomy, governance.
- Gate: report and metric definitions finalized.

### Phase 1 (Completed in planning)
- Data model + control contract + quickstart runbook.
- Gate: measurable policies for filtering/correction/synthetic tracks.

### Phase 2 (Next session)
- Execution task graph and controlled run sequence.
- Gate metrics:
  - Clean holdout CER/WER
  - Annotation consistency rate
  - Truncation rate
  - High-loss defect purity

## 6. IDE Language Hygiene Note

- Preferred IDE/report language for this initiative: **English**.
- No workspace-level VS Code language override was found in this repository.
- OCR config `charset: korean` is model vocabulary configuration and not IDE UI language.

## 7. Deferred Execution Runbook (Not Executed Here)

```bash
# Build filtered manifest (example)
uv run python scripts/data/build_filtered_manifest.py \
  --input data/processed/recognition/train_manifest.jsonl \
  --output data/processed/recognition/train_manifest.filtered.jsonl \
  --drop-script-mismatch --drop-clipping-risk --drop-illegible --max-loss-percentile 95

# Controlled training continuation (example)
uv run python scripts/runners/train.py \
  mode=train \
  experiment=parseq_flash_plateau_images \
  +checkpoint_path=outputs/checkpoints/best-acc-0.8372_v2.ckpt \
  data.sequence.tokenizer_max_len=25 \
  recognition.max_label_length=25 \
  trainer.val_check_interval=1.0 \
  +train.logger.wandb.high_loss_audit.enabled=true
```

## 8. Final Planning Outcome

- Spec-Kit planning artifacts generated and validated for feature `003-ocr-data-quality-remediation`.
- Experiment workspace initialized for controlled follow-up execution.
- Technical report persisted for reuse and cross-session traceability.
