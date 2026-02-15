---
ads_version: 1.0
type: implementation_plan
category: development
status: active
version: 1.0
tags: recognition,parseq,performance,hydra,traceability
title: Recognition Dynamic Sequence Trim Plan
date: 2026-02-15 02:22 (KST)
branch: main
---

# Implementation Plan - Recognition Dynamic Sequence Trim Plan

## Goal
Improve recognition training throughput by reducing unnecessary PAD-token compute through safe, batch-level dynamic sequence trimming, while preserving model/checkpoint compatibility by keeping global model `max_len` unchanged.

## Constraints and Guardrails
- Do not reduce architecture/tokenizer checkpoint shape anchors (`max_len`, `max_label_length`) in this change.
- Preserve existing experiment behavior by default (feature-flagged rollout).
- Keep naming and config behavior Hydra-compatible and discoverable.
- Changes must be traceable through artifact + config keys + tests.

## Scope
- In scope:
	- Recognition collate enhancement for optional right-trim to batch max non-PAD length.
	- Config externalization for trim behavior under `data` / `collate_fn`.
	- Lightweight observability for sequence-length and PAD ratio.
	- Validation for train/val parity and resume compatibility.
- Out of scope:
	- Reducing model `max_len` or tokenizer fixed max length.
	- Truncated BPTT and auxiliary length-prediction heads.

## Baseline Findings (Context Bundles + Code Inspection)
- Current tokenizer pads to fixed length in `ocr/domains/recognition/data/tokenizer.py`.
- Current collate stacks pre-padded tensors in `ocr/domains/recognition/data/collate.py`.
- Decoder already computes with runtime `T = targets.shape[1]`, making dynamic trim feasible without decoder rewrite.
- Max-length constants are spread across `configs/data`, `configs/domain`, and model architecture configs; lowering these can break checkpoint compatibility.

## Proposed Changes

### Configuration
- [ ] Add explicit feature flags in recognition dataset config:
	- `data.sequence.trim_pad_to_batch_max: false` (default off)
	- `data.sequence.min_keep_tokens: 4` (BOS/EOS safety margin)
	- `data.sequence.log_length_stats: true`
- [ ] Extend collate config declaration (Hydra) to pass trim flags into recognition collate callable.
- [ ] Document config ownership and precedence in one location (recognition data config README note or inline schema comments).

### Code
- [ ] Update recognition collate to support optional dynamic right-trim:
	- Determine per-batch max non-PAD position.
	- Clamp with `min_keep_tokens`.
	- Slice `text_tokens[:, :trim_len]` before batch return.
- [ ] Add runtime telemetry (first N steps/epoch):
	- batch max token length
	- mean non-PAD length
	- PAD ratio
- [ ] Keep default behavior unchanged when feature flag is `false`.
- [ ] Ensure no API break in `OCRDataPLModule` hydra instantiation path.

## Externalization Plan (Hard-coded Discovery Mitigation)
- Canonicalize sequence behavior under config (not code constants):
	- `configs/data/datasets/recognition.yaml` owns sequence trim runtime knobs.
	- Domain/model files retain architectural max-length anchors for compatibility.
- Add a short “length knobs” section to recognition config docs to centralize discoverability.

## Rollout Plan (Low-Risk)
1. **Phase A (No Behavior Change)**
	 - Add config keys + collate code path behind default-off flag.
	 - Add tests for exact output parity when flag is off.
2. **Phase B (Controlled Enablement)**
	 - Enable trim flag in a single experiment config (e.g., `parseq_flash_fast`) only.
	 - Compare throughput, val/acc, val/cer for 1–3 short runs.
3. **Phase C (Broader Adoption)**
	 - If stable, propagate flag to other recognition experiments.
	 - Keep fallback toggle for immediate rollback.

## Risk Assessment
- **Compatibility risk (medium):** unexpected assumptions about fixed token length in downstream code.
	- Mitigation: default-off rollout + focused tests around batch shapes.
- **Metric drift risk (medium):** trimming mistakes could drop EOS/PAD semantics.
	- Mitigation: enforce minimum keep tokens and verify token invariants.
- **Operational risk (low):** Hydra wiring mistakes.
	- Mitigation: startup config validation and one-command smoke run.

## Verification Plan

### Automated Tests
- [ ] Unit: collate default mode returns identical tensor shapes/values.
- [ ] Unit: trim mode returns reduced `T` but preserves BOS/EOS and non-PAD tokens.
- [ ] Unit: min token guard prevents over-trimming.
- [ ] Integration: one recognition dataloader batch builds successfully with both modes.

### Manual Verification
- [ ] Run short train (1–2 epochs) with trim disabled and enabled.
- [ ] Confirm checkpoint resume still works unchanged.
- [ ] Confirm logs show expected sequence stats and improved iter/sec when enabled.

## Traceability
- Artifact: this implementation plan.
- Planned touchpoints:
	- `ocr/domains/recognition/data/collate.py`
	- `configs/data/datasets/recognition.yaml`
	- selected experiment config(s) for opt-in enablement
- Validation hooks:
	- `aqms artifact validate --all`
	- targeted pytest selection for recognition data/collate.

## Execution Log
- 2026-02-15: Phase A implemented (default-off behavior)
	- Added configurable dynamic right-trim path in recognition collate.
	- Externalized sequence knobs under `data.sequence` in recognition dataset config.
	- Added focused tests: `tests/ocr/datasets/test_recognition_collate.py`.
	- Verified with `uv run python -m pytest -q tests/ocr/datasets/test_recognition_collate.py` (4 passed).
- 2026-02-15: Phase B started (single-experiment opt-in)
	- Enabled `data.sequence.trim_pad_to_batch_max: true` in `configs/experiment/parseq_flash_fast.yaml` only.
	- Kept global defaults and other experiments unchanged for safe rollout.
- 2026-02-15: Smoke validation with no-checkpoint guarantee
	- Executed short train with runtime overrides:
		- `trainer.enable_checkpointing=false`
		- `train.callbacks=null`
		- `checkpoint_path=null`
		- `train.logger=null`
		- tiny limits for train/val batches
	- Checkpoint verification before/after run:
		- `find outputs/checkpoints -name '*.ckpt' | wc -l` remained `20`.
		- Latest checkpoint mtimes/paths unchanged.
- 2026-02-15: Phase B A/B smoke benchmark (no-checkpoint mode)
	- Shared runtime overrides for both runs:
		- `trainer.enable_checkpointing=false`
		- `train.callbacks=null`
		- `train.logger=null`
		- `checkpoint_path=null`
		- `trainer.limit_train_batches=10`
		- `trainer.limit_val_batches=0`
		- `trainer.val_check_interval=null`
	- A/B settings:
		- **trim OFF:** `data.sequence.trim_pad_to_batch_max=false`
			- End-of-epoch throughput: `5.53 it/s`
			- Wall time: `ELAPSED_SEC=12.616`
		- **trim ON:** `data.sequence.trim_pad_to_batch_max=true`
			- End-of-epoch throughput: `5.84 it/s`
			- Wall time: `ELAPSED_SEC=13.981`
	- Interpretation:
		- Throughput indicator improved in this micro-run, but wall-clock did not (startup/noise dominates at small batch count).
		- Requires longer controlled run for stable conclusion.
	- Checkpoint verification after A/B:
		- `CKPT_BEFORE=20`, `CKPT_AFTER=20` (no new checkpoint artifacts).

## Spec Kit Fit
Using GitHub Spec Kit is appropriate for this task because:
- The change seems small but has cross-cutting effects (data pipeline, config, training behavior).
- It benefits from explicit spec/plan/tasks traceability before implementation.
- Recommended flow: `speckit.specify` → `speckit.plan` → `speckit.tasks` before code edits.
