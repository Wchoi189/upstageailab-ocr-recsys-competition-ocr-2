---
ads_version: 1.0
type: assessment
category: evaluation
status: active
version: 1.0
tags:
  - assessment
  - observability
  - logging
title: WandB Logging Fallbacks Caused Misleading Validation Images
date: 2026-02-14 04:22 (KST)
branch: 001-mcp-tooling-refactor
---

# Assessment - WandB Logging Fallbacks Caused Misleading Validation Images

## Purpose
Document the validation image logging confusion caused by WandB fallback paths, capture the root cause, and define the direction to simplify callbacks and remove silent fallbacks.

## Findings

### Key Observations
1. Validation images were logged by a legacy callback that could not decode GT or predictions, producing placeholders like "?" and "(no pred)" and a dark canvas.
2. A newer module-based logger already exists and correctly decodes `tokens` and `text_tokens`, but multiple logging paths made it unclear which one was active.
3. Fallback behavior in the legacy callback hid missing data contracts instead of surfacing a clear error or warning.

## Analysis
- **Root cause**: Multiple WandB image logging paths (module-based logger + legacy callback) and permissive fallbacks masked data contract mismatches.
- **Why it was hard to debug**: The fallback produced plausible-looking images without indicating which logger was active or that GT/pred decoding failed.
- **System impact**: Reduced trust in validation visuals and increased time to diagnose training issues.

## Recommendations
1. **Single source of truth for recognition image logging** (Priority: High). Keep the module-based logger and remove or deprecate the legacy callback.
2. **No silent fallbacks in logging** (Priority: High). Replace placeholders with explicit warnings or hard failures when GT/pred decoding is unavailable.
3. **Document the data contract** for recognition logging (Priority: Medium). Specify required batch keys and inference outputs.

## Implementation Plan
- [ ] Consolidate WandB logging to the module-based logger and remove legacy callback wiring in configs.
- [ ] Add a minimal spec rule that forbids silent fallbacks in observability paths.
- [ ] Create a short developer note describing the single logging path and required keys.
