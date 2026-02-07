# Session Handover: Cross-Attention Debugging Results

**Date:** 2026-02-08 00:32
**Status:** 🔴 All Hypotheses Failed → Requires New Approach
**Context:** Debugging PARSeq recognition pipeline (0% acc, 100% CER)

---

## Executive Summary

This session implemented and tested **three architectural fixes** based on external cross-attention analysis. **All three failed** to resolve the training collapse:

| Run | Fix | Visual Mean | Pos Mean | Result |
|-----|-----|-------------|----------|--------|
| 42  | Memory masking | ~1.0 | ~8.0 | ❌ Degraded (empty strings) |
| 43  | + LayerNorm | ~0.0 | ~8.0 | ❌ Still failing (empty strings) |
| 44  | + 0.1× pos scale | ~0.0 | ~0.8 | ❌ Still failing (empty strings) |

**All runs:** 0% val/acc, 100% val/cer, gradients vanish by step 10-20

**Conclusion:** The problem is **deeper than signal balancing**. The external hypothesis about cross-attention issues was partially correct (magnitude imbalance existed), but fixes alone don't resolve the core failure.

---

## What We Learned

### 1. Signal Imbalance Was Real (But Not Root Cause)
- Visual features (~1.0) vs Positional (~8.0) = 8× disparity ✓ Confirmed
- LayerNorm + scaling fixed the imbalance (both ~0.8-1.0) ✓ Achieved
- Model still can't learn ❌ Imbalance was symptom, not cause

### 2. Memory Masking May Be Harmful
- Run 42 (with masking) degraded model from repeating chars → empty strings
- Suggests masking isn't needed OR was implemented incorrectly

### 3. Code Changes Made
Files modified with fixes that should be **reconsidered or reverted**:

**[`decoder.py`](file:///workspaces/ocr/domains/recognition/models/decoder.py)**
- Added `memory_key_padding_mask` parameter
- Passes default all-False mask to transformer decoder

**[`architecture.py`](file:///workspaces/ocr/domains/recognition/models/architecture.py)**
- Added `self.visual_norm = nn.LayerNorm(256)`
- Normalizes visual features before pos encoding
- Scaled pos encoding by 0.1× (now ~0.8 instead of ~8.0)

**Recommendation:** These changes balance signals but don't fix training. Consider:
- **Keep LayerNorm** (good practice for ViT-style architectures)
- **Remove pos scaling** or make learnable
- **Remove memory masking** (no clear benefit, possible harm)

---

## Remaining Hypotheses

Since signal balancing didn't work, the problem is likely:

### Hypothesis A: BOS/EOS Token Misalignment (High Priority)
External analysis flagged this but we couldn't verify due to diagnostic script complexity.

**Evidence:**
- Model predicts empty strings (BOS+EOS immediately)
- Suggests decoder learns to "give up" instantly
- Could be dataset not adding BOS/EOS, or adding them incorrectly

**Next Step:** Manually inspect a batch to verify token structure

### Hypothesis B: Fundamental Decoder Architecture Issue
PARSeq's autoregressive decoder may have bugs or incompatibilities:
-Use of learned vs sinusoidal pos encoding
- Cross-attention mechanism itself
- Loss calculation on shifted tokens

**Next Step:** Try simpler decoder (CRNN-style) to isolate issue

### Hypothesis C: Need Pretrained Weights
The analysis hinted at pretraining. From scratch training may fail due to:
- Insufficient data
- Optimization landscape too difficult
- Initialization issues

**Next Step:** Try TrOCR or pretrained PARSeq weights

---

## Recommended Next Actions (Priority Order)

### 1. **Manual Token Inspection** (Quickest)
```python
# In Python console or Jupyter
from ocr.core.data import ...
batch = next(iter(val_loader))
print(batch["text_tokens"][:5])  # First 5 samples
# Check: First token == BOS (1)? Last non-pad == EOS (2)?
```

### 2. **Try CRNN Architecture** (Medium effort)
- Simpler RNN-based decoder
- Proven to work for OCR
- If CRNN works → PARSeq decoder is broken
- If CRNN fails → Data/encoder issue

### 3. **Use Pretrained Weights** (If available)
- PARSeq paper has pretrained checkpoints
- TrOCR from HuggingFace (different architecture but proven)

### 4. **Deep Decoder Audit** (High effort, last resort)
- Line-by-line review of decoder implementation
- Compare with official PARSeq repository
- Check for PyTorch version incompatibilities

---

## Files & Artifacts

### Code Modified
- [`ocr/domains/recognition/models/decoder.py`](file:///workspaces/ocr/domains/recognition/models/decoder.py) - Added memory masking
- [`ocr/domains/recognition/models/architecture.py`](file:///workspaces/ocr/domains/recognition/models/architecture.py) - Added LayerNorm + pos scaling

### Analysis Documents
- [`cross_attention_analysis.md`](file:///home/vscode/.gemini/antigravity/brain/5a68977d-fe39-47f2-bea0-95f735f9a519/cross_attention_analysis.md) - Initial analysis from external insights
- [`run42_analysis.md`](file:///home/vscode/.gemini/antigravity/brain/5a68977d-fe39-47f2-bea0-95f735f9a519/run42_analysis.md) - Memory masking failure
- [`run43_analysis.md`](file:///home/vscode/.gemini/antigravity/brain/5a68977d-fe39-47f2-bea0-95f735f9a519/run43_analysis.md) - LayerNorm insufficient

### Training Logs
- `__DEBUG__/training_failures/logs/run42.log` - Memory masking test
- `__DEBUG__/training_failures/logs/run43.log` - LayerNorm test
- `__DEBUG__/training_failures/logs/run44.log` - Pos scaling test

### Task Tracking
- [`__DEBUG__/training_failures/task.md`](file:///workspaces/__DEBUG__/training_failures/task.md) - Updated through Phase 3

---

## Continuation Prompt

```
The cross-attention debugging approach has been exhausted. All signal balancing fixes (memory masking, LayerNorm, positional scaling) failed to resolve 0% accuracy.

Priority investigation:
1. Manually inspect batch tokens to verify BOS/EOS structure
2. If tokens OK, try CRNN architecture as control experiment
3. If CRNN fails, investigate encoder/data
4. If CRNN works, deep audit PARSeq decoder vs official implementation

Current hypothesis: BOS/EOS token misalignment or fundamental decoder bug, not cross-attention signal issues.
```

---

## Token Budget

**Current:** ~43% used (86k/200k)
**Remaining:** ~114k tokens - sufficient for next debugging phase

---

## Key Insight

**Signal magnitude balancing is necessary but not sufficient.** The PARSeq decoder has a deeper issue that prevents learning even with perfectly balanced inputs. This likely requires either:
- Switching architectures (CRNN, TrOCR)
- Using pretrained weights
- Finding and fixing a critical decoder bug

The external cross-attention analysis was valuable for identifying the signal imbalance, but the proposed fixes don't address the root cause of the training failure.
