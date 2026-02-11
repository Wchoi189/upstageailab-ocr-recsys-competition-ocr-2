# Run 42 Analysis: Memory Mask Fix Failed

**Date:** 2026-02-08 00:15  
**Run:** 42 (memory_key_padding_mask implementation)  
**Result:** ❌ FAILED - Degraded further

---

## Summary

Implementing `memory_key_padding_mask` in the decoder **did NOT fix the training failure**. The model degraded from:
- **Before (baseline/Run 41):** Repeating single character ('과과과과...')
- **After (Run 42):** Empty string (BOS+EOS only: `[1, 2]`)

**Final metrics remain unchanged:**
- `val/acc: 0.000`
- `val/cer: 1.000`

---

## Key Observations

### 1. Gradients Still Vanishing
```
Step 0:  Grad Norm = 0.0053-0.0193  (non-zero, learning starts)
Step 10: Grad Norm = 0.0000         (vanished)
Step 20: Grad Norm = 0.0000         (vanished)
```

###  2. **CRITICAL: 8:1 Scale Disparity**
```
Visual Features Mean: ~1.0 (range 0.74-1.26)
Pos Embed Mean:       ~8.06 (constant at 8.0625 ± 7.94 std)
```

**This is an 8× magnitude difference!** The positional embedding is drowning out the visual signal in cross-attention.

### 3. Model Behavior Worsened
- Pre-fix: At least generating characters (unigram prior)
- Post-fix: Generating nothing (immediate EOS after BOS)

---

## Revised Hypothesis

The external analysis may have been partially incorrect. The **primary** issue is likely:

**Positional Embedding Dominance** (not memory masking)

### Evidence:
1. **8:1 ratio** between pos_embed and visual features
2. Gradients vanish **after** model processes first few batches (not immediately)
3. Visual features (~1.0) are numerically insignificant compared to pos encoding (~8.0)

### What Actually Happens:
1. Visual encoder outputs reasonable features (mean ~1.0)
2. Positional encoding **adds** ~8.0 magnitude signal
3. Combined signal = dominated by position, not content
4. Decoder cross-attention learns to **ignore visual content**, only attends to positional structure
5. Without visual grounding, decoder collapses to language model prior (unigram/BOS-EOS)

---

## Next Steps (Revised)

### Option A: Normalize **Before** Adding Positional Encoding
```python
# in architecture.py, before pos encoding
visual_feat = self.visual_norm(visual_feat)  # LayerNorm
visual_feat = self.pos_encoding(visual_feat)
```

### Option B: Scale Down Positional Encoding
```python
# Multiply pos encoding by 0.1 or learnable scale
pos_embed = self.pos_encoding(x) * 0.1
```

### Option C: Use Learned Positional Embeddings (not sinusoidal)
- Switch from `PositionalEncoding2D` to learned `nn.Parameter`
- Learnable params will auto-scale during training

---

## Recommendation

**Pri 1:** Try Option A (LayerNorm before pos encoding)  
- Most conservative, standard practice in ViT/BERT
- Normalizes visual features to unit scale before adding positional info

**Pri 2:** If that fails, try Option B (scale down pos encoding 10×)  
- Quick experiment to validate hypothesis

**Pri 3:** Option C requires more changes, save for last

---

## Files to Modify

### Option A (LayerNorm):
- [ocr/domains/recognition/models/architecture.py](file:///workspaces/ocr/domains/recognition/models/architecture.py)
  - Add `self.visual_norm = nn.LayerNorm(...)` in [__init__](file:///workspaces/ocr/domains/recognition/models/architecture.py#16-58)
  - Apply before `self.pos_encoding`

### Run 43 Config:
```bash
uv run python scripts/runners/train.py \
  experiment=rec_baseline_v1 \
  trainer.max_epochs=3 \
  trainer.limit_train_batches=20 \
  +run_name="run43_visual_layernorm"
```

---

## Conclusion

**Memory masking was a red herring.** The decoder can't learn from visual features because they're numerically insignificant compared to positional embeddings. Fix the scale disparity first, then reassess if memory masking is still needed.
