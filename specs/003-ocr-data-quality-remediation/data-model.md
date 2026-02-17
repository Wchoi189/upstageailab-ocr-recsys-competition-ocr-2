# Data Model: OCR High-Loss Data-Quality Remediation

## Canonical Defect Taxonomy
- Source of truth: `contracts/remediation-control.openapi.yaml#/components/schemas/DefectClass`
- All taxonomy references in spec, tasks, and scripts must mirror this enum exactly.

## Entity: HighLossAuditReport
- Purpose: Persistent audit summary for stakeholder communication and governance.
- Fields:
  - `report_id` (string, unique)
  - `created_at` (datetime)
  - `run_id` (string)
  - `model_variant` (string)
  - `defect_summary` (object: class -> counts/rates)
  - `metrics_snapshot` (object: train_loss, val_loss, val_acc, val_cer)
  - `recommendations` (list[string])
  - `status` (enum: draft, reviewed, approved)

## Entity: DefectClassRecord
- Purpose: Attach one or more taxonomy labels to a sample.
- Fields:
  - `sample_id` (string)
  - `image_path` (string)
  - `gt_text` (string)
  - `pred_text` (string)
  - `loss` (float)
  - `defect_classes` (list[enum])
    - `script_mismatch`
    - `missing_char_due_to_clipping`
    - `hallucinated_gt_chars`
    - `unreadable_sample`
    - `truncation_misalignment`
  - `severity` (enum: low, medium, high, critical)
  - `review_status` (enum: pending, verified, corrected, dropped)

## Entity: RemediationPhasePlan
- Purpose: Structured control of multi-phase work and gate decisions.
- Fields:
  - `phase_id` (string)
  - `phase_name` (string)
  - `entry_gates` (list[string])
  - `exit_gates` (list[string])
  - `required_metrics` (list[string])
  - `deferred_commands` (list[string])
  - `owner` (string)
  - `state` (enum: planned, active, blocked, done)

## Entity: ExperimentWorkspaceRecord
- Purpose: Track dedicated ETK experiment context for this initiative.
- Fields:
  - `experiment_id` (string)
  - `name` (string)
  - `type` (string)
  - `intention` (string)
  - `linked_spec` (path)
  - `linked_report` (path)
  - `status` (enum: initialized, planning, execution_ready, running, completed)

## Entity: DataStrategyDecisionLog
- Purpose: Preserve rationale and tradeoffs for filtering/correction/synthetic decisions.
- Fields:
  - `decision_id` (string)
  - `timestamp` (datetime)
  - `track` (enum: filtering, correction_queue, synthetic_augmentation)
  - `decision` (string)
  - `rationale` (string)
  - `alternatives` (list[string])
  - `expected_effect` (string)
  - `risk_level` (enum: low, medium, high)

## Entity: CleanHoldoutDefinition
- Purpose: Authoritative validation split free of known defects.
- Fields:
  - `holdout_id` (string)
  - `source_manifest` (path)
  - `sampling_strategy` (enum: random, stratified_by_length, stratified_by_script)
  - `sample_size` (int, minimum 200)
  - `verification_method` (enum: double_blind_manual, model_consensus_plus_manual_review)
  - `defect_exclusion_criteria` (list[enum])
  - `created_at` (datetime)
  - `version` (string)

## Relationships
- `HighLossAuditReport` 1:N `DefectClassRecord`
- `RemediationPhasePlan` 1:N `DataStrategyDecisionLog`
- `ExperimentWorkspaceRecord` 1:N `HighLossAuditReport`
- `ExperimentWorkspaceRecord` 1:N `RemediationPhasePlan`
- `ExperimentWorkspaceRecord` 1:N `CleanHoldoutDefinition`

## Validation Rules
- `defect_classes` must be non-empty for records entering manual review queue.
- `severity=critical` requires `review_status` in `{verified, corrected, dropped}` before phase exit.
- `exit_gates` cannot be marked passed unless all `required_metrics` are present.
- `linked_spec` and `linked_report` must resolve to existing paths before execution phase starts.
- `sample_size >= 200` for clean holdout definitions.
- `CleanHoldoutDefinition.version` must be immutable once used for gate evaluation.

## Metrics

### Metric: Truncation Rate
- Formula: `count(samples where len(gt_text) >= tokenizer_max_len - 2) / total_samples`
- Interpretation:
  - `< 0.05`: acceptable
  - `0.05-0.10`: review required
  - `> 0.10`: blocks phase exit

### Metric: Annotation Consistency
- Definition: Agreement between GT text and visual evidence in image patch.
- Measurement protocol:
  - Sample 100 random instances per checkpoint
  - Manual verification question: “Does GT match visible text?” (yes/no)
  - Consistency rate = `correct_annotations / total_reviewed`
- Threshold: `>= 0.95` required for phase exit.

## Annotation Quality Assurance
- Double-blind review on 20% of manual correction queue.
- Cohen's kappa targets:
  - defect class assignment: `kappa >= 0.75`
  - severity assignment: `kappa >= 0.70`
  - corrected GT exact-match agreement: `>= 0.85`
- Disagreement handling:
  - `kappa < 0.60`: escalate to senior annotator and root-cause review
  - `0.60 <= kappa < 0.75`: discussion and re-annotation
  - `kappa >= 0.75`: accept with majority vote or senior tiebreak

## State Transitions

### DefectClassRecord.review_status
- `pending -> verified`
- `pending -> corrected`
- `pending -> dropped`
- `verified -> corrected` (allowed when correction evidence is added)

### ExperimentWorkspaceRecord.status
- `initialized -> planning -> execution_ready -> running -> completed`
- `running -> execution_ready` allowed for rollback/rework
