# Session Handover: Recognition Training Failure Debugging

**Date:** 2026-02-07 23:05
**Status:** 🟡 Planning Complete → Ready for Diagnostic Execution
**Context:** `__DEBUG__/training_failures` | Model collapse to unigram (0% acc, 100% CER)

---

## Executive Summary

This session performed **root cause analysis** of the PARSeq decoder's failure to learn, building on previous debugging work (Runs 22-41). External architectural insights revealed **three structural defects** in the cross-attention mechanism. We've created:

1. **Comprehensive technical analysis** ([cross_attention_analysis.md](file:///home/vscode/.gemini/antigravity/brain/5a68977d-fe39-47f2-bea0-95f735f9a519/cross_attention_analysis.md))
2. **Phased implementation plan** ([implementation_plan_cross_attention_fix.md](file:///home/vscode/.gemini/antigravity/brain/5a68977d-fe39-47f2-bea0-95f735f9a519/implementation_plan_cross_attention_fix.md))
3. **Diagnostic scripts** (Phase 1 - ready to run)

**No code changes have been made yet** - this was a planning/analysis session.

---

## Key Findings

### Root Causes Identified

| Issue | Severity | Location | Impact |
|-------|----------|----------|--------|
| **Missing `memory_key_padding_mask`** | 🔴 CRITICAL | [`decoder.py:136`](file:///workspaces/ocr/domains/recognition/models/decoder.py#136) | Decoder attends to padded visual features → gradient dilution → unigram collapse |
| **BOS/EOS Token Handling** | ⚠️ HIGH | [`architecture.py:126`](file:///workspaces/ocr/domains/recognition/models/architecture.py#126) | Potential token misalignment → immediate loss explosion → vanishing gradients |
| **Visual-Text Scale Disparity** | ⚠️ MEDIUM | [`decoder.py:120-126`](file:///workspaces/ocr/domains/recognition/models/decoder.py#120-126) | 3:1 magnitude ratio (28 vs 9.5) biases attention toward text-only pathways |

**Primary Hypothesis:** Missing memory mask is the main blocker. Run 41's failure on synthetic data proves this is architectural, not data-related.

---

## Next Steps (Immediate Actions)

### Phase 1: Run Diagnostics
```bash
# BOS/EOS verification
cd /workspaces
uv run python __DEBUG__/training_failures/scripts/inspect_tokens.py

# Visual padding analysis
uv run python __DEBUG__/training_failures/scripts/check_visual_padding.py
```

### Phase 2: Apply Decoder Fix
Modify [`decoder.py`](file:///workspaces/ocr/domains/recognition/models/decoder.py):
1. Add `memory_key_padding_mask` parameter
2. Generate default mask if not provided
3. Pass to `self.decoder()` call

See [implementation plan](file:///home/vscode/.gemini/antigravity/brain/5a68977d-fe39-47f2-bea0-95f735f9a519/implementation_plan_cross_attention_fix.md) for exact changes.

### Phase 2 Verification (Run 42)
```bash
uv run python scripts/runners/train.py \
  experiment=rec_baseline_v1 \
  trainer.max_epochs=3 \
  trainer.limit_train_batches=20 \
  +run_name="run42_memory_mask_fix"
```

**Success:** `val/acc` > 0.0, `val/cer` < 1.0, gradients stay non-zero

---

## Continuation Prompt

```
Continue debugging the recognition pipeline. Phase 0 (analysis) complete.
Phase 1 diagnostic scripts ready. Run them, then implement Phase 2 decoder fix.
See task.md and implementation_plan_cross_attention_fix.md for details.
```
