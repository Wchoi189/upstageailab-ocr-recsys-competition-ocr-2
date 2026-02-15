# Flash Attention Learning Failure - Root Cause Analysis

**Date**: 2026-02-13
**Experiment ID**: 20260212_191401_flashattentionlearningfailu
**Status**: ROOT CAUSE IDENTIFIED

## Executive Summary

The model produces repeated single tokens (e.g., "김김김김...") during validation NOT because it failed to learn, but because of a critical bug in the **greedy decoding inference loop** when using Flash Attention.

## Key Findings

### 1. Model Initialization is Correct
- ✅ Head weights: Xavier initialization (mean=0.000078, std=0.029482)
- ✅ Token embeddings: All unique (1027/1027)
- ✅ Positional encodings: Distinct per position (diff=0.070)
- ✅ Training forward pass: Diverse logits (std=0.554)

### 2. Inference Immediately Collapses
- ❌ **With a FRESHLY INITIALIZED model**, inference produces repeated token 836 for ALL samples
- ❌ This happens even without any training
- ❌ The issue is NOT about learning - it's about the inference algorithm itself

### 3. Training vs Inference Behavior

| Mode | Behavior | Logits Diversity |
|------|----------|------------------|
| Training (teacher forcing) | ✅ Works correctly | High (std=0.554) |
| Inference (greedy decoding) | ❌ Collapses to single token | Unknown |

## Root Cause: Greedy Decoding Loop Issue

**Location**: `ocr/domains/recognition/models/architecture.py:181-206`

### The Problematic Loop

```python
for i in range(max_len):
    # Decode one step
    step_logits = self._decode_step(visual_memory, tgt_tokens)  # [B, 1, V]

    # Greedy selection
    next_token = step_logits.argmax(dim=-1)  # [B, 1]

    # Append
    tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
```

### Hypothesis: Attention Mask Issue During Incremental Decoding

When `_decode_step` calls the decoder with growing `tgt_tokens`:
- Step 1: `tgt_tokens` = [BOS] → Length 1
- Step 2: `tgt_tokens` = [BOS, token1] → Length 2
- Step 3: `tgt_tokens` = [BOS, token1, token2] → Length 3

Inside the decoder (architecture.py:188-190):
```python
if tgt_mask is None:
    # Standard causal mask for AR decoding
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
```

**Potential Issues**:
1. The causal mask is regenerated each step with growing size
2. Flash Attention's handling of this growing mask might cause attention to collapse
3. The last position might be attending to a degenerate subset of previous tokens

## Evidence from Training Logs

From user's training output:
```
Pred IDs: [1, 261, 261, 261, 261, 261, 261, 261, ...]
Pred: '김김김김김김김김김김김김...' | GT: '"변화와'
```

- Different batches predict different repeated tokens (382, 832, 988, 769, 261, 836)
- This is consistent with the freshly initialized model predicting token 836
- The specific token depends on random initialization, but the REPETITION pattern is systematic

## Next Steps

### Immediate Fix Options

1. **Disable Flash Attention for Inference Only**
   - Keep Flash Attention for training (speed benefit)
   - Use standard attention for inference (correctness)
   - Test if standard attention fixes the repetition issue

2. **Fix the Mask Handling**
   - Investigate why causal mask + Flash Attention causes collapse
   - Check if using `is_causal=True` instead of explicit mask helps
   - Verify mask dtype/device compatibility during incremental decoding

3. **Alternative: Use `is_causal=True` Flag**
   - Instead of passing explicit causal mask, use the `is_causal` parameter
   - Flash Attention has optimized path for causal attention
   - May avoid the mask conversion issues

### Testing Plan

1. Create minimal reproduction script with just decoder + inference loop
2. Test with standard attention (no Flash) to confirm it works
3. Test with Flash Attention + `is_causal=True` flag
4. Test with Flash Attention + explicit mask (current broken state)
5. Compare attention patterns between working and broken configs

## Diagnostic Scripts

- `scripts/diagnose_attention_collapse.py`: Tests Flash Attention primitives ✅ All passed
- `scripts/check_model_learning.py`: Tests model initialization and inference ❌ Found the bug

## Related Code Locations

- Inference loop: `ocr/domains/recognition/models/architecture.py:160-206`
- Decoder forward: `ocr/domains/recognition/models/decoder.py:127-203`
- Flash Attention: `ocr/domains/recognition/models/flash_attention.py:198-290`
- Training wrapper: `ocr/domains/recognition/module.py:68-121`

## CRITICAL UPDATE: Decoder Produces Identical Logits Regardless of Input!

### Step-by-Step Analysis Reveals the True Bug

Step-by-step decoding analysis shows **IDENTICAL logits across ALL decoding steps**:

```
Step 1:  length=1  → logits: mean=-0.0032, std=0.5795, top=[1013, 135, 623, 262, 393]
Step 2:  length=2  → logits: mean=-0.0032, std=0.5795, top=[1013, 135, 623, 262, 393]
Step 3:  length=3  → logits: mean=-0.0032, std=0.5795, top=[1013, 135, 623, 262, 393]
...
Step 10: length=10 → logits: mean=-0.0032, std=0.5795, top=[1013, 135, 623, 262, 393]
```

**The decoder is NOT conditioning on the input sequence at all!**

### NOT a Flash Attention Bug

Testing with standard attention shows **IDENTICAL ISSUE**:
- ❌ Standard Attention: Repeated tokens (943)
- ❌ Flash Attention: Repeated tokens (1013)

**This has NOTHING to do with Flash Attention!** It's a fundamental decoder bug.

### Root Cause: Positional Encoding or Attention Mechanism

**Leading Theory**: The decoder is broken - either:
1. Positional embeddings not applied/scaled correctly
2. Causal mask malformed causing all positions to attend to position 0 only
3. Token embeddings not being used properly

The decoder.py:177-184 scales embeddings by `sqrt(d_model)` = ~19.6, which might cause issues.

## Immediate Action Required

1. Add debug logging to decoder to trace where inputs become identical
2. Test without positional encoding scaling
3. Test with no causal mask (full attention)
4. Compare with working PARSeq implementation

## Conclusion

**The decoder is fundamentally broken - NOT a Flash Attention issue**. The model cannot learn because inference doesn't work at all. The decoder produces the same output regardless of input sequence, making autoregressive generation impossible.

**Priority**: Fix the decoder BEFORE any training experiments.
