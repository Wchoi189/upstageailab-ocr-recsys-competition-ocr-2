---
ads_version: 1.0
type: implementation_plan
category: development
status: active
version: 1.0
tags:
  - implementation
  - logging
  - cleanup
title: WandB Logging Consolidation and Fallback Removal
date: 2026-02-14 04:23 (KST)
branch: 001-mcp-tooling-refactor
---

# Implementation Plan - WandB Logging Consolidation and Fallback Removal

## Goal
Consolidate recognition WandB image logging into a single, explicit path and remove silent fallbacks so validation visuals reflect real GT/pred decoding.

## Proposed Changes

### Configuration
- [x] Remove legacy callback wiring from experiment defaults and train callback presets.
- [ ] Ensure only the module-based recognition logger is enabled for validation images.

### Code
- [x] Deprecate or delete `RecognitionWandbImageLogger` if unused after config cleanup.
- [x] Add explicit warnings or hard failures when GT/pred decoding is unavailable in logging.
- [ ] Document required batch keys (`images`, `text_tokens` or `label`) and inference outputs (`tokens`).

## Verification Plan

### Automated Tests
- [ ] Add a small unit test that exercises the module-based logger with missing keys and asserts a warning or error.

### Manual Verification
- [ ] Run a 1-epoch validation and confirm WandB shows `val_recognition_samples` with real GT/Pred strings.
- [ ] Confirm the legacy `validation/recognition_samples` panel no longer appears.
