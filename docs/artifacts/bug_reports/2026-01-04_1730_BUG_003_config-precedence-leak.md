---
ads_version: "1.0"
type: "bug_report"
category: "architecture"
status: "completed"
severity: "critical"
version: "1.0"
tags: ['configuration', 'hydra', 'legacy', 'precedence']
title: "Legacy Config Leaks Override Architecture Components"
date: "2026-01-04 17:30 (KST)"
resolved_date: "2026-01-04 19:30 (KST)"
branch: "main"
summary: "The PARSeq training configuration incorrectly instantiates `FPNDecoder` instead of `PARSeqDecoder` because legacy defaults (from `train_v2` → `_base/model` → `dbnet`) take precedence over the architecture's specific component definitions during the configuration merge process."
---

# Details

## Symptoms
When running `fast_dev_run` for PARSeq:
```text
ValueError: FPNDecoder requires at least two feature maps from the encoder.
```
This confirms `FPNDecoder` is being used, despite `parseq.yaml` specifying `parseq_decoder`.

## Root Cause
The `train_parseq.yaml` experiment config inherits from `defaults: - train_v2` (legacy) AND `- model/architectures/parseq`. The `_base/model.yaml` includes `/model/architectures: dbnet` as a default.
The `OCRModel._prepare_component_configs` method merged:
1. `top_level_overrides` (from cfg) AFTER `direct_overrides` (from architecture), allowing legacy to win
2. `cfg.component_overrides` at highest priority, containing legacy dbnet components

## Fix Implementation (BUG_003)
Updated `ocr/models/architecture.py`:
1. Added `_filter_architecture_conflicts()` method to remove legacy components when they conflict with architecture
2. Reordered merges: `filtered_top_level` → `direct_overrides` → `filtered_user_overrides`
3. Both `top_level_overrides` and `cfg.component_overrides` are now filtered

## Verification
```
INFO ocr.models.architecture - BUG_003: Filtering legacy decoder (fpn_decoder) in favor of architecture decoder (parseq_decoder)
INFO ocr.models.architecture - BUG_003: Filtering legacy head (db_head) in favor of architecture head (parseq_head)
INFO ocr.models.architecture - BUG_003: Filtering legacy loss (db_loss) in favor of architecture loss (cross_entropy)
```
