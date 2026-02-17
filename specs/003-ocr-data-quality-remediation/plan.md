# Implementation Plan: OCR High-Loss Data-Quality Remediation

**Branch**: `[003-ocr-data-quality-remediation]` | **Date**: 2026-02-18 | **Spec**: `/specs/003-ocr-data-quality-remediation/spec.md`
**Input**: Feature specification from `/specs/003-ocr-data-quality-remediation/spec.md`

## Summary

Create a planning-first, execution-deferred remediation blueprint for OCR data-quality defects that contaminate supervision (incorrect GT, clipping misalignment, unreadable samples, script mismatch). The approach defines a persistent audit report, a controlled Experiment Manager workspace, and a multi-phase gated workflow focused on data filtering, semi-supervised correction queueing, and synthetic augmentation strategy.

## Technical Context

**Language/Version**: Python 3.11 (repo standard via `uv run`)
**Primary Dependencies**: PyTorch/Lightning OCR stack, Hydra/OmegaConf configs, Weights & Biases audit logging, ETK (`experiment_manager`)
**Storage**: Filesystem artifacts (`docs/reports`, `specs/*`, `dev_tools/experiment_manager/experiments/*`), JSON/CSV manifests, W&B media/table artifacts
**Testing**: `pytest` (unit/integration), contract-style checks on manifests and metrics definitions
**Target Platform**: Linux + CUDA training environment, VS Code workspace workflows
**Project Type**: Single ML repository with structured experiment tracking and spec-first planning
**Performance Goals**: Preserve or improve clean-holdout CER/WER while reducing defective-label exposure in training batches; keep truncation and annotation inconsistency rates explicitly monitored
**Constraints**: Planning-first session (no remediation training execution in this session), maintain reproducibility and artifact traceability, avoid architecture rewrites
**Scale/Scope**: Remediation workflow for recognition datasets at production-like scale (tens of thousands to hundreds of thousands of text patches)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The constitution file at `.specify/memory/constitution.md` is currently template-only placeholders. Interim gates are derived from repository non-negotiables (`AGENTS.md`, `copilot-instructions.md`) and the feature spec constraints.

| Gate | Source | Status | Notes |
|------|--------|--------|-------|
| Spec-driven workflow | Spec-Kit + AGENTS rules | PASS | Work anchored to `spec.md` and generated planning artifacts. |
| No ad hoc artifact sprawl | AgentQMS/AGENTS constraints | PASS | Artifacts limited to feature spec directory, docs report, and ETK experiment metadata. |
| Planning-first (no execution) | Feature constraints FR-009 | PASS | Commands documented as deferred runbook only. |
| Context-minimal, targeted analysis | AgentQMS/AGENTS constraints | PASS | Only relevant OCR/training/ETK files referenced. |

### Post-Design Re-check (after Phase 1)

All gates remain PASS. Design artifacts include explicit deferred execution language, measurable gates, and no implementation run commitments.

## Project Structure

### Documentation (this feature)

```text
specs/003-ocr-data-quality-remediation/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── remediation-control.openapi.yaml
└── tasks.md                     # to be generated in separate session via /speckit.tasks
```

### Source Code / Runtime Areas (repository)

```text
configs/
├── domain/
├── data/
└── model/

ocr/
└── domains/recognition/

docs/
└── reports/

dev_tools/experiment_manager/
├── src/etk/
└── experiments/

scripts/
└── runners/
```

**Structure Decision**: Keep existing monorepo structure; this phase adds planning and control artifacts only, with no runtime code-path modifications.

## Multi-Phase Plan (Planning-first, Execution Deferred)

### Phase 0 — Diagnostic Baseline and Governance
- Deliverables: `research.md`, persistent technical report in `docs/reports`, initialized ETK experiment metadata.
- Gate 0 (entry): Defect taxonomy and severity model agreed.
- Gate 0 (exit): Baseline report includes defect evidence, metrics definitions, and deferred runbook.

#### Phase 0 Required Non-Mutating Diagnostics

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

- These diagnostics are allowed in planning because they are read-only and required for threshold calibration.

### Phase 1 — Data Policy and Control Design
- Deliverables: `data-model.md`, `contracts/remediation-control.openapi.yaml`, `quickstart.md`.
- Gate 1 (entry): Phase 0 artifacts reviewed.
- Gate 1 (exit): Filtering policy, correction queue policy, synthetic augmentation policy have measurable acceptance criteria.

#### Phase 1 Target Policy Baselines
- Initial loss filter boundary: `p95` of observed loss distribution (dataset-specific).
- Synthetic starter mix for robustness run: `70% synthetic : 30% verified real`.
- Golden holdout minimum size: 200 samples, immutable after versioning.

### Phase 2 — Execution Blueprint (Next Session)
- Deliverables (next session): `tasks.md`, command-by-command run sequence, validation checkpoints.
- Gate 2 (entry): Approved Phase 1 design and experiment workspace ready.
- Gate 2 (exit): Execution-ready task graph with ownership and rollback criteria.

### Phase 3 — Controlled Execution and Evaluation (Future)
- Deliverables (future): filtered dataset manifests, correction queue outputs, training/evaluation logs.
- Mandatory metrics: clean-holdout CER/WER, truncation rate, annotation consistency, high-loss defect purity.
- Gate 3 (exit): Go/no-go decision for broader adoption.

## Emergency Actions (Execution Sessions)

If active training is suspected to be learning from corrupted GT:
1. Stop run immediately (`Ctrl+C` or orchestrator/runner stop command)
2. Preserve latest checkpoint into `outputs/checkpoints/emergency_backup/`
3. Log incident note in experiment metadata (`.metadata/00-status/`)
4. Mark run for remediation review before any continuation

## Rollback Triggers

### Filtering Stage
- If >40% dataset flagged for removal: **ABORT** and recalibrate thresholds
- If clean-holdout CER worsens >20% after filtering: **ROLLBACK** to prior manifest
- If any key label class is fully removed: **REVIEW** for distribution bias

### Correction Queue Stage
- If inter-annotator kappa <0.70: **PAUSE** and retrain annotation protocol
- If correction ratio >30% of reviewed queue: **ESCALATE** root-cause analysis

### Synthetic Augmentation Stage
- If synthetic/real discriminator >70% accuracy: **REJECT** synthetic set and regenerate
- If synthetic-only validation CER >10%: **REJECT** synthetic set quality

## Gate Definition Source

- Canonical phase gates and thresholds should be maintained in `specs/003-ocr-data-quality-remediation/planning/ocr-data-quality-phase-gates.md`.

## Complexity Tracking

No constitution violations requiring exceptions in this planning session.
