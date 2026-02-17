# Research: OCR High-Loss Data-Quality Remediation

## Scope

Resolve planning unknowns and choose design decisions for:
- Defect triage policy
- Real vs synthetic data strategy
- Semi-supervised correction flow
- Experiment tracking and deferred execution governance

## Decisions

### Decision 1: Treat label corruption as hard-blocking data risk
- Rationale: Incorrect supervision causes harmful gradient signals and unstable sequence learning; temporary accuracy stagnation is acceptable, corrupted supervision is not.
- Alternatives considered:
  - Keep all samples with robust loss tricks only (rejected: does not remove toxic labels)
  - End-to-end model redesign first (rejected: bypasses root data issue)

### Decision 2: Use a three-lane triage policy (drop / review / keep)
- Rationale: Deterministic defects (script mismatch, clipping, unreadable patches) should be removed quickly; ambiguous cases should enter review queue; clean samples remain.
- Alternatives considered:
  - Pure loss-threshold filtering only (rejected: high loss includes both hard-valid and mislabeled samples)
  - Fully manual curation (rejected: too slow and expensive)

### Decision 3: Adopt hybrid real+synthetic strategy
- Rationale: Real data provides domain realism; synthetic data improves clean coverage and balancing for sparse patterns.
- Alternatives considered:
  - Real-only with noisy labels (rejected: quality ceiling and contamination risk)
  - Synthetic-only pretraining for this phase (rejected: domain gap risk)

### Decision 4: Implement semi-supervised GT correction queue (human-in-the-loop)
- Rationale: Model consensus plus rule checks can prioritize likely GT errors for human relabel, reducing manual load while preserving quality control.
- Alternatives considered:
  - Automatic relabel without review (rejected: unsafe for governance)
  - No correction queue (rejected: unresolved defects accumulate)

### Decision 5: Sequence budget should be expanded in execution phase
- Rationale: Current max length (25) risks truncation for long strings and can confound clipping analysis.
- Alternatives considered:
  - Keep max length fixed at 25 (rejected: elevated truncation bias)
  - Large jump to 48+ immediately (rejected: unnecessary memory increase before measurement)

### Decision 6: Run non-mutating diagnostics during planning to calibrate thresholds
- Rationale: Gate thresholds (loss boundaries, truncation alerts) are not defensible without baseline distributions.
- Required diagnostics:
  - defect prevalence distribution
  - loss percentile distribution
  - truncation statistics at current tokenizer max length
- Alternatives considered:
  - Full deferral to execution (rejected: weak threshold rationale)
  - Fixed hardcoded thresholds (rejected: poor cross-dataset generalization)

### Decision 7: Initialize dedicated ETK experiment now, defer all mutating training runs
- Rationale: Enables traceable governance and artifact linkage while honoring planning-only scope.
- Alternatives considered:
  - Delay experiment creation until implementation (rejected: weak traceability)
  - Run pilot training in same session (rejected: violates planning-first constraint)

### Decision 8: Calibrate loss threshold per dataset using percentiles
- Rationale: Absolute thresholds (e.g., fixed loss cutoff) do not transfer reliably.
- Initial policy: Use `p95` as starting filter boundary, then adjust after defect purity review.
- Alternatives considered:
  - fixed `loss > 3.0` threshold (rejected as universal default)
  - no thresholding (rejected: review queue overload)

### Decision 9: Adopt explicit synthetic composition target for execution baseline
- Target starting mix: `70% synthetic : 30% verified real` for robustness-focused phase, with later rebalancing.
- Synthetic spec baseline:
  - target volume: 50,000 samples
  - include Hangul, digits, and frequent symbols/patterns (addresses, registration formats)
  - degradation profile includes blur/noise/perspective/font diversity

### Decision 10: Tiered golden-set verification using local triage + Upstage OCR
- Tier 1: Model confidence/prediction agreement triage.
- Tier 2: Local OCR check (PaddleOCR) and optional local VLM triage.
- Tier 3: Upstage Document OCR API verification for ambiguous/high-risk cases.
- Rationale: balances cost, throughput, and validation quality.
- Alternatives considered:
  - Upstage-only validation on all samples (rejected: unnecessary cost)
  - local-only validation (rejected: lower reliability on edge cases)

## Best-Practice Notes by Technology

### OCR Data Curation
- Combine deterministic rule filters with uncertainty-aware review queues.
- Track defect-class prevalence and per-class CER impact across iterations.

### W&B High-Loss Audits
- Keep per-epoch top-k high-loss sampling enabled for regression visibility.
- Tag each sampled defect with taxonomy labels for trend monitoring.

### Experiment Manager (ETK)
- Use `etk init` for experiment creation and avoid manual directory creation.
- Maintain artifact lineage between spec outputs and experiment metadata.

### Golden Validation via OCR APIs
- Use patch-level validation, not full-document parsing, to preserve GT-to-patch attribution.
- For small patches, allow optional upscaling prior to API call when needed.
- Apply confidence-based triage to reduce paid API calls.

## Resolved Clarifications

- English letters (ASCII) are supported by tokenizer charset.
