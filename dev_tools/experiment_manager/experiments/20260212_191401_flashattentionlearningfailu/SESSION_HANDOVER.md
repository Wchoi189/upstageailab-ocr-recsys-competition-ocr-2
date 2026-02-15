---
type: session_handover
experiment_id: 20260212_191401_flashattentionlearningfailu
date: 2026-02-14
status: decoder_fixed_ready_for_training
---

# Session Handover: Flash Attention Decoder Fix

## Problem
- Model predicted repeated tokens: `"김김김김..."` instead of `"변화와"`
- Validation accuracy: 0.000 (no learning)
- Occurred after PLM+Flash refactor from working 82% baseline

## Root Cause
**Architecture mismatch**: Refactor replaced original two-stream decoder with single-stream `nn.TransformerDecoder`

Original PARSeq uses:
- **Two streams**: query (positional) + content (tokens)
- **Learned positional queries** (std=0.02)
- **Special BOS handling**: null context for first token

Current (broken) used:
- Single-stream `nn.TransformerDecoder`
- All positions treated identically
- Result: identical logits across all decoding steps

## Solution Implemented ✅

Created `/workspaces/ocr/domains/recognition/models/two_stream_decoder.py`:
```python
class TwoStreamDecoderLayer(nn.Module):
    # Pre-LN with separate query/content streams
    def forward(self, query, content, memory, ...):
        query = self.forward_stream(query, query_norm, content_norm, memory, ...)
        content = self.forward_stream(content, content_norm, content_norm, memory, ...)
        return query, content
```

Updated `/workspaces/ocr/domains/recognition/models/decoder.py`:
- Line 4: Import TwoStreamDecoder
- Line 48-55: Use TwoStreamDecoder instead of nn.TransformerDecoder
- Line 55: Changed `pos_encoder` → `pos_queries` (learned, not sinusoidal)
- Line 77-80: Init with `trunc_normal_(std=0.02)`
- Line 154-190: Two-stream forward pass with query/content separation

## Verification Results

**Before Fix:**
```
Step 1-10: logits mean=-0.0032, std=0.5795 (IDENTICAL)
Predictions: [836, 836, 836, 836, ...]
```

**After Fix:**
```
Step 1: logits mean=-0.0276, std=0.5797
Step 2: logits mean=-0.0238, std=0.5910
Predictions: [736, 347, 406, 982, 175, 424] (DIVERSE)
```

## Next Steps

### 1. Baseline Training (No Flash, No PLM)
**Goal**: Restore 82% accuracy with fixed two-stream decoder

```bash
uv run python scripts/runners/train.py \
  experiment=rec_baseline_official \
  trainer.max_epochs=30 \
  data.batch_size=128 \
  +train/logger=wandb \
  +train/callbacks=recognition_wandb
```

**Success Criteria**: `val/acc >= 0.80` after 30 epochs

### 2. Add Flash Attention (No PLM)
**Goal**: Verify Flash Attention works with two-stream decoder

```bash
uv run python scripts/runners/train.py \
  experiment=parseq_flash \
  trainer.max_epochs=30 \
  data.batch_size=256 \
  +train/logger=wandb
```

**Expected**: 2-4x speedup, 400-640 img/sec, 95% GPU util

### 3. Add PLM Training
**Goal**: Full optimization with PLM + Flash

```bash
uv run python scripts/runners/train.py \
  experiment=parseq_plm_flash \
  trainer.max_epochs=30 \
  data.batch_size=160 \
  ++data.num_workers=12 \
  +train/logger=wandb
```

## Critical Files

**Modified:**
- `/workspaces/ocr/domains/recognition/models/decoder.py` (two-stream integration)
- `/workspaces/ocr/domains/recognition/models/two_stream_decoder.py` (NEW)

**Reference (original working):**
- `__DEBUG__/training_failures_2026-02-07/implementations/baudm_parseq/strhub/models/parseq/modules.py`
- `__DEBUG__/training_failures_2026-02-07/implementations/baudm_parseq/strhub/models/parseq/model.py`

**Diagnostic Scripts:**
- `experiments/20260212_191401_flashattentionlearningfailu/scripts/check_model_learning.py`
- `experiments/20260212_191401_flashattentionlearningfailu/scripts/test_inference_fix.py`

**Findings:**
- `experiments/20260212_191401_flashattentionlearningfailu/FINDINGS.md`

## Configuration Notes

### parseq_flash_fast.yaml
```yaml
domain: recognition_flash
model:
  architectures: parseq_flash
  decoder:
    use_flash_attention: true  # Currently ignored, uses two-stream
    plm_config: null
trainer:
  precision: "16-mixed"
  max_epochs: 20
data:
  batch_size: 512
```

### Current Limitations
- `use_flash_attention` flag in decoder config is **bypassed**
- Two-stream decoder doesn't integrate Flash Attention yet
- Need to add Flash support to `TwoStreamDecoderLayer` (future work)

## Flash Attention Integration (TODO)

To add Flash Attention to two-stream decoder:

1. Modify `TwoStreamDecoderLayer.__init__`:
```python
from ocr.domains.recognition.models.flash_attention import FlashMultiheadAttention

self.self_attn = FlashMultiheadAttention(d_model, nhead, dropout, batch_first=True)
self.cross_attn = FlashMultiheadAttention(d_model, nhead, dropout, batch_first=True)
```

2. Wrap forward in `enable_flash_attention_kernel()` context

3. Test with `use_flash_attention=True`

## Continuation Prompt

```
The PARSeq decoder has been fixed with two-stream architecture. Model now produces diverse predictions instead of repeated tokens.

Current status:
- ✅ Two-stream decoder implemented
- ✅ Inference working (logits change per step)
- ⏳ Training needed to restore 82% baseline accuracy
- ⏳ Flash Attention integration pending

Next: Run baseline training (30 epochs) to verify model learns correctly, then incrementally add Flash Attention and PLM optimizations.

Reference materials:
- Original working implementation: __DEBUG__/training_failures_2026-02-07/implementations/baudm_parseq/
- Experiment workspace: dev_tools/experiment_manager/experiments/20260212_191401_flashattentionlearningfailu/
- Modified decoder: ocr/domains/recognition/models/decoder.py
```

## Key Insights

1. **Not a Flash Attention bug**: Standard attention had same issue
2. **Architecture matters**: Single-stream decoder fundamentally incompatible with PARSeq design
3. **Refactor carefully**: Working baseline (82%) proves architecture can work - compare before changing
4. **Learned PE > Sinusoidal**: Original uses learned positional queries (std=0.02), not scaled sinusoidal

## Debug Commands

**Quick test (5 min):**
```bash
uv run python scripts/runners/train.py \
  experiment=parseq_flash_fast \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=100 \
  trainer.limit_val_batches=50
```

**Check predictions:**
```bash
uv run python dev_tools/experiment_manager/experiments/20260212_191401_flashattentionlearningfailu/scripts/check_model_learning.py
```

**Step-by-step analysis:**
```bash
uv run python dev_tools/experiment_manager/experiments/20260212_191401_flashattentionlearningfailu/scripts/test_inference_fix.py
```

## Contact Points

- Bug report: `docs/artifacts/bug_reports/2026-02-13_0134_bug_001_attention-plm-cuda-error.md`
- Scratchpad: `__DEBUG__/scratchpad.md`
- Plan file: `.claude/plans/keen-weaving-crystal.md`
