# Flash Attention Decoder Fix - Complete ✅

**Status**: Decoder fixed, ready for training
**Date**: 2026-02-14
**Experiment**: `dev_tools/experiment_manager/experiments/20260212_191401_flashattentionlearningfailu/`

## Quick Summary

**Problem**: Repeated token predictions `"김김김..."` (val_acc=0.000)
**Cause**: Single-stream decoder replaced original two-stream architecture
**Fix**: Implemented `TwoStreamDecoder` from original PARSeq
**Result**: ✅ Diverse predictions, logits change per step

## Key Files

- `ocr/domains/recognition/models/two_stream_decoder.py` - NEW two-stream decoder
- `ocr/domains/recognition/models/decoder.py` - Updated to use two-stream
- `SESSION_HANDOVER.md` - Full handover doc in experiment directory

## Next Steps

1. **Train baseline** (30 epochs) → restore 82% accuracy
2. **Add Flash Attention** to TwoStreamDecoderLayer
3. **Enable PLM training** with Flash

## Train Commands

```bash
# Baseline (verify learning)
uv run python scripts/runners/train.py \
  experiment=rec_baseline_official \
  trainer.max_epochs=30 \
  +train/logger=wandb

# Quick test (5 min)
uv run python scripts/runners/train.py \
  experiment=parseq_flash_fast \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=100 \
  trainer.limit_val_batches=50
```

## Verification

```bash
# Check diverse predictions
uv run python dev_tools/experiment_manager/experiments/20260212_191401_flashattentionlearningfailu/scripts/check_model_learning.py

# Step-by-step logit analysis
uv run python dev_tools/experiment_manager/experiments/20260212_191401_flashattentionlearningfailu/scripts/test_inference_fix.py
```

## Architecture Change

**Before (Broken):**
```python
nn.TransformerDecoder  # Single-stream
→ Identical logits across all steps
```

**After (Fixed):**
```python
TwoStreamDecoder  # Query + content streams
→ Logits change: Step1=mean:-0.0276, Step2=mean:-0.0238
```

## References

- Original: `__DEBUG__/training_failures_2026-02-07/implementations/baudm_parseq/`
- Handover: `dev_tools/experiment_manager/experiments/20260212_191401_flashattentionlearningfailu/SESSION_HANDOVER.md`
- Bug report: `docs/artifacts/bug_reports/2026-02-13_0134_bug_001_attention-plm-cuda-error.md`

---

## Common Repomix CLI

```bash
repomix --style markdown \
  --include 'ocr/domains/recognition/models/' \
  --output ocr_recognition_models_2026-02-14.md
```
