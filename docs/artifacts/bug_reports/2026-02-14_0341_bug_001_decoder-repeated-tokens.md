---
ads_version: 1.0
type: bug_report
category: troubleshooting
status: completed
version: 1.0
severity: high
description: Decoder outputs identical logits across decoding steps, producing repeated token predictions during inference.
tags:
  - bug
  - issue
  - troubleshooting
title: Fix PARSeq Decoder Inference Bug (Repeated Token Predictions)
date: 2026-02-14 03:41 (KST)
branch: 001-mcp-tooling-refactor
---

# Bug Report - Fix PARSeq Decoder Inference Bug (Repeated Token Predictions)
Bug ID: BUG-001

## Summary
During refactors to add PLM + Flash Attention optimizations, the PARSeq decoder now emits identical logits at each decoding step, producing repeated token predictions during autoregressive inference. The issue reproduces with standard `nn.TransformerDecoderLayer`, so Flash Attention is not the cause. Root cause is a regression in the decoder architecture, primarily double-scaling positional encodings, which overwhelms token embeddings and prevents conditioning on previous tokens.

## Environment
- **OS/Env**: Linux
- **Dependencies**: PyTorch (version unspecified), PARSeq decoder, Flash Attention optional (issue reproduces without Flash Attention)

## Reproduction
1. Run inference with the current PARSeq decoder implementation.
2. Observe logits across decoding steps and predicted tokens.
3. Compare behavior to the original working implementation in __DEBUG__/training_failures_2026-02-07/implementations/baudm_parseq/.

Diagnostic scripts:
- `uv run python dev_tools/experiment_manager/experiments/20260212_191401_flashattentionlearningfailu/scripts/check_model_learning.py`
- `uv run python dev_tools/experiment_manager/experiments/20260212_191401_flashattentionlearningfailu/scripts/test_inference_fix.py`

## Comparison
**Expected**: Logits change across decoding steps and predicted tokens vary (diverse sequences across samples). Training should show non-zero validation accuracy after short runs.
**Actual**: Logits are identical across decoding steps, producing repeated token predictions. Validation accuracy remains 0.000 during short training.

## Root Cause

**UPDATE (2026-02-14 04:00)**: After analyzing the test results, the root cause is **information leakage in the two-stream attention mechanism**, not positional encoding scaling.

### The Actual Bug

In `ocr/domains/recognition/models/decoder.py:156-180`, the two-stream decoder constructs content and query streams as follows:

For targets = `[BOS, tok1, tok2]` (T=3):
- **Content stream**: `[embed(BOS), pos_queries[0] + embed(tok1), pos_queries[1] + embed(tok2)]`
- **Query stream**: `pos_queries[:3]` (positions [0, 1, 2])
- **Causal mask**: `torch.triu(torch.ones((T, T)), 1)` - upper triangular excluding diagonal

The causal mask allows position `i` to attend to positions `[0, 1, ..., i]` **including itself**. This means:
- Query at position 2 can see content at position 2, which contains `embed(tok2)`
- **The model sees the token it's supposed to predict** → information leakage!

### Why This Causes Repeated Tokens

1. During autoregressive decoding, when predicting token at position `i`, the query should only see tokens `[0, 1, ..., i-1]`
2. With the current implementation, query[i] sees content[i], which includes the embedding of token[i]
3. The model doesn't learn to use previous context properly because it's "cheating" by seeing the target
4. At inference (with random initialization), without access to the actual token embeddings, the model produces flat/uncertain logits
5. Result: Repeated predictions of the most common token (388 in this case)

### Evidence

From test results:
- Training works (val/acc: 0.722) because teacher forcing provides correct embeddings
- Inference breaks because generated tokens don't match what the model learned to "cheat" with
- Issue persists with standard attention (not Flash-specific)
- Logits are flat/uncertain (~0.6% probabilities) indicating the model isn't using context

### Previous Misdiagnosis

1. ~~Double scaling positional encodings~~ - Not the issue; positional encoding helps distinguish positions
2. ~~Architecture drift from two-stream~~ - Actually using two-stream, but incorrectly
3. ~~Missing position-specific query mask~~ - Mask exists but allows self-attention incorrectly

## Proposed Fix

### Option 1: Fix Content Stream Alignment (Recommended)

Modify the content stream construction in `decoder.py` to prevent information leakage:

**Current (Broken)**:
```python
# Content contains token embeddings at positions [0, 1, ..., T-1]
tgt_emb = [embed(BOS), pos_queries[0] + embed(tok1), pos_queries[1] + embed(tok2)]
# Query at position i can attend to content[i] (sees its own token!)
```

**Fixed**:
```python
# Shift content stream by one position - content[i] should only have info from tokens [0..i-1]
# Option A: Don't include current token in content
null_ctx = scaled_emb[:, :1]
if T > 1:
    # Shift: content[i] = pos_queries[i-1] + embed(token[i-1])
    tgt_emb_rest = self.pos_queries[:, 1:T] + scaled_emb[:, :-1]
    tgt_emb = torch.cat([null_ctx, tgt_emb_rest], dim=1)

# Option B: Adjust causal mask to prevent position i from attending to content[i]
# Change mask from triu(..., 1) to triu(..., 0) - block diagonal attendance
causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), 0)  # diagonal=True
```

### Option 2: Use Standard Transformer Decoder

Replace two-stream decoder with standard transformer decoder that properly handles autoregressive constraints:
- Use standard `nn.TransformerDecoderLayer` with proper causal masking
- Ensure position embeddings are added correctly without information leakage
- Simpler architecture, easier to maintain

### Verification Steps

Phase 1: Implement fix and verify with test script
```bash
uv run python dev_tools/experiment_manager/experiments/.../scripts/test_inference_fix.py
```

Phase 2: Check for diverse predictions
- Unique tokens > 1 per sequence
- Logits should vary across steps
- Top-5 probabilities should be more confident (>5%)

Phase 3: Short training sanity check
- Val accuracy > 0 after 1000 steps
- CER decreasing over time

## Logs

### Training (Works)
```
Epoch 0/0  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1000/1000 0:02:54
val/acc: 0.722 val/cer: 0.208
✅ Training complete!
```

### Inference (Broken)
```
TEST 1: Current Implementation (BROKEN)
Predicted tokens shape: torch.Size([4, 26])
  Sample 0: [388, 388, 388, 388, 388, 388, 388] → Unique tokens: 1/7
    ❌ FAILED: All tokens are identical!

STEP-BY-STEP ANALYSIS:
  Step 1: Current length=1
    Top 5 predictions: [388, 429, 496, 590, 164]
    Top 5 probs: ['0.0086', '0.0043', '0.0037', '0.0036', '0.0035']
    Selected: 388
    Logits stats: mean=0.0317, std=0.5788

  [Steps 2-10: Same token 388 predicted every time]

  Final sequence: [1, 388, 388, 388, 388, 388, 388, 388, 388, 388, 388]
  Unique tokens (excluding BOS): 1/10

TEST 3: Fix 2 - Standard Attention for Inference
Creating model with standard attention...
  Sample 0: [388, 388, 388, 388, 388, 388, 388] → Unique tokens: 1/7
    ❌ FAILED: Still repeating with standard attention!
  ❌ FAILED: Issue persists even without Flash Attention
```

**Key Observations**:
- Training works because teacher forcing provides correct token embeddings
- Inference produces flat distributions (top prob ~0.8%, should be >50% for confident predictions)
- Issue is NOT Flash Attention-specific - persists with standard attention
- Model always predicts token 388 (likely most common token in vocab)

## Impact
Critical regression in inference and training: the decoder does not condition on prior tokens, leading to repeated predictions and stalled learning. This blocks Flash Attention and PLM optimization work until resolved.

## Resolution

**Date**: 2026-02-14 04:53 (KST)
**Fix**: Implemented Option 2 (Content Stream Shift)

### Changes Made

Modified `ocr/domains/recognition/models/decoder.py` line 161-169 to shift the content stream:

```python
# BUGFIX (BUG-001): Shift content stream to prevent information leakage
# Original: content[i] = pos_queries[i-1] + embed(token[i])
# Fixed: content[i] = pos_queries[i] + embed(token[i-1])
if T > 1:
    # Shift embeddings: use tokens [0:T-1] instead of [1:T]
    # Use positions [1:T] to maintain proper positional encoding alignment
    tgt_emb_rest = self.pos_queries[:, 1:T] + scaled_emb[:, :T-1]
    tgt_emb = torch.cat([null_ctx, tgt_emb_rest], dim=1)
```

### Verification Results

Test script: `scripts/test_decoder_fix.py`

**Before Fix:**
- Token 388 repeated identically (1 unique token)
- Top probability: ~0.8% (very uncertain)
- Flat logit distribution

**After Fix:** ✅
- 5 unique tokens out of 7 predictions
- Logits vary properly across steps (mean diffs: 0.02-0.24)
- Diverse token predictions

### Next Steps

1. ✅ Fix implemented and verified with unit test
2. ⏳ Run full training with fixed decoder to validate convergence
3. ⏳ Test Flash Attention performance with corrected architecture
4. ⏳ Re-enable PLM training mode once AR decoding is stable
