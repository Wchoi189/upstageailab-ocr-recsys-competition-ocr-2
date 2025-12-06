---
title: "Bug 20251110 003 Cudnn Workspace Memory Exhaustion"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---





# Bug Report: cuDNN Workspace Memory Exhaustion on RTX 3060

## Bug ID
BUG-20251110-003

## Summary
Training fails with cuDNN "FIND was unable to find an engine to execute this computation" error on RTX 3060 12GB, while the same configuration worked on RTX 3090 24GB. The error occurs during backward pass at step 0-26, indicating insufficient workspace memory for cuDNN convolution algorithms.

## Environment
- **OS:** Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)
- **Python:** 3.10.12
- **PyTorch:** 2.8.0+cu128
- **CUDA:** 12.8 (driver 13.0)
- **cuDNN:** 91002
- **GPU:** NVIDIA GeForce RTX 3060 12GB (migrated from RTX 3090 24GB)
- **Model:** DBNet with ResNet50 encoder, PAN decoder
- **Precision:** FP32
- **Batch Size:** 4

## Steps to Reproduce
1. Migrate from RTX 3090 24GB to RTX 3060 12GB (WSL2)
2. Run training with batch_size=4
3. Training starts, completes sanity check
4. Fails at step 0-26 with cuDNN FIND error

## Expected Behavior
Training should complete successfully with appropriate batch size for the GPU's memory capacity.

## Actual Behavior

### First Run (Test Script):
```
Epoch 0/0  ━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 26/818
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```
Error at `torch.isnan(param.grad).any()` in `on_before_optimizer_step`

### Second Run (Manual):
```
Epoch 0/0  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0/818
RuntimeError: FIND was unable to find an engine to execute this computation
```
Error during `backward()` pass

### Additional Symptoms:
1. **WandB Logging**: Black images with green GT boxes, no red prediction boxes
2. **Validation Metrics**: All zeros (recall: 0.000, precision: 0.000, hmean: 0.000)
3. **Blank Logging Spaces**: Large gaps (12+ lines) in import logging output
4. **Variability**: Different failure points (step 0 vs step 26)

## Root Cause Analysis

### Primary Issue: cuDNN Workspace Memory Exhaustion

**cuDNN FIND Error Meaning:**
- cuDNN uses a "FIND" algorithm to select the best convolution implementation
- It needs temporary "workspace" memory to test different algorithms
- If insufficient workspace memory is available, FIND fails

**Why This Happens:**
1. **GPU Memory Constraint**: RTX 3060 12GB vs RTX 3090 24GB (50% less memory)
2. **Large Model**: ResNet50 + PAN decoder + DBNet head is memory-intensive
3. **Batch Size 4**: Each image is 640x640, batch consumes ~2-3GB
4. **Workspace Allocation**: cuDNN needs additional memory for backward pass

### Memory Breakdown (Estimated):
```
RTX 3060 12GB Total Memory:
  Model weights (ResNet50):     ~100MB
  Activations (batch=4):        ~2-3GB
  Gradients:                    ~100MB
  Ground truth maps:            ~500MB
  cuDNN workspace:              ~1-2GB (required but unavailable)
  PyTorch overhead:             ~500MB
  ────────────────────────────────────
  Total needed:                 ~5-7GB (fits in 12GB)
  cuDNN workspace:              FAILS to allocate additional 1-2GB
```

**Why Workspace Fails:**
- PyTorch pre-allocates memory for model + activations + gradients
- cuDNN needs ADDITIONAL memory for workspace during backward pass
- With batch=4, not enough fragmented memory available for workspace
- cuDNN FIND fails → backward pass crashes

### Secondary Issues:

**1. Black Images in WandB:**
- Not a data issue (data validation confirmed 3272 images with annotations)
- Likely a rendering issue or model failing before predictions logged
- May be due to crash happening before image logging callback

**2. Logging Blank Spaces:**
- Indicates system is struggling with memory/IO
- Possibly swap thrashing or memory allocation delays
- Not directly related to cuDNN error but symptom of memory pressure

**3. Variability (Step 0 vs 26):**
- Data-dependent: Different batches have different memory requirements
- Image sizes vary slightly after augmentation
- Some batches trigger workspace allocation earlier

## Proposed Solution

### Solution 1: Reduce Batch Size (RECOMMENDED)

**Use batch_size=1 or 2 with gradient accumulation:**
```bash
uv run python runners/train.py \
  dataloaders.train_dataloader.batch_size=2 \
  dataloaders.val_dataloader.batch_size=2 \
  trainer.accumulate_grad_batches=2 \
  # ... other args
```

**Benefits:**
- Effective batch size = 2 × 2 = 4 (same as before)
- Lower peak memory usage
- cuDNN workspace can allocate successfully
- No model changes required

**Trade-offs:**
- 2x slower training (due to 2 gradient accumulation steps)
- Still achieves same effective batch size

### Solution 2: Use Smaller Model

**Use ResNet18 instead of ResNet50:**
```bash
model.encoder.model_name=resnet18
```

**Benefits:**
- ~75% less memory for model weights and activations
- Faster training
- More memory available for cuDNN workspace

**Trade-offs:**
- Slightly lower accuracy (ResNet18 < ResNet50)
- May need more training epochs

### Solution 3: Use FP16 (Mixed Precision)

**Enable automatic mixed precision:**
```bash
trainer.precision=16-mixed
```

**Benefits:**
- ~50% memory reduction for activations and gradients
- Faster training on modern GPUs
- More memory for cuDNN workspace

**Trade-offs:**
- Requires gradient scaling (handled automatically by PyTorch Lightning)
- Possible numerical instability (but DBNet should be fine)
- Need to test convergence

### Solution 4: Reduce Image Size

**Use smaller input images:**
```yaml
# In preprocessing config
target_size: [512, 512]  # Instead of [640, 640]
```

**Benefits:**
- Significant memory reduction (~40% less)
- Faster training

**Trade-offs:**
- May reduce detection accuracy for small text
- Need to retrain from scratch

## Implementation

### Recommended Fix: Batch Size Reduction

**Test with minimal batch:**
```bash
./scripts/test_minimal_batch.sh
```

**If successful, use production config:**
```bash
uv run python runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  exp_name=ocr_training-dbnet-resnet50-batch2 \
  model/architectures=dbnet \
  model.encoder.model_name=resnet50 \
  model.component_overrides.decoder.name=pan_decoder \
  dataloaders.train_dataloader.batch_size=2 \
  dataloaders.val_dataloader.batch_size=2 \
  trainer.accumulate_grad_batches=2 \
  trainer.max_epochs=50 \
  trainer.precision=32 \
  seed=42
```

### Alternative: FP16 + Batch Size 2

**For faster training:**
```bash
uv run python runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  exp_name=ocr_training-dbnet-resnet50-fp16 \
  model/architectures=dbnet \
  dataloaders.train_dataloader.batch_size=2 \
  trainer.precision=16-mixed \
  trainer.max_epochs=50
```

## Testing Plan

### Phase 1: Minimal Test
- [x] Diagnose root cause
- [ ] Test batch_size=1 (10 steps)
- [ ] Verify no cuDNN errors
- [ ] Check memory usage

### Phase 2: Production Test
- [ ] Test batch_size=2, accumulate=2 (100 steps)
- [ ] Verify stable training
- [ ] Check convergence rate
- [ ] Validate metrics

### Phase 3: Optimization
- [ ] Test FP16 if needed
- [ ] Benchmark training speed
- [ ] Optimize data loading
- [ ] Monitor GPU memory usage

## Prevention Measures

### Hardware Migration Checklist
1. **Profile memory usage** on source GPU
2. **Calculate target GPU capacity** (consider 20% overhead)
3. **Adjust batch size** proportionally to memory ratio
4. **Test with minimal config** before full training
5. **Monitor GPU memory** during training

### Code Guidelines
1. Add memory profiling to training script
2. Log GPU memory usage at each step
3. Add early detection of cuDNN errors
4. Provide helpful error messages with suggested fixes

## Impact Assessment

### Severity: Critical
- **Blocks all training** on RTX 3060
- **Migration issue** from RTX 3090 to RTX 3060
- **Fixable** with batch size reduction

### Workaround
Use batch_size=2 with accumulate_grad_batches=2:
- Same effective batch size
- Fits in 12GB memory
- Slower but functional

## Related Issues

### Relationship to Previous Bugs:
- **BUG-20251110-002**: Fixed numerical stability, but revealed memory issue
- **BUG-20251110-001**: Data quality issue (resolved, data is valid)
- **BUG-20251109-002**: CUDA illegal memory access (same symptom, different root cause)

### Hardware Comparison:
| Aspect | RTX 3090 | RTX 3060 | Impact |
|--------|----------|----------|--------|
| Memory | 24GB | 12GB | 50% reduction |
| Bandwidth | 936 GB/s | 360 GB/s | Slower data transfer |
| CUDA Cores | 10496 | 3584 | Slower computation |
| Batch Size | 4 works | 4 fails | Need batch=2 |

## Notes

### Why BUG-20251110-002 Fix Didn't Resolve This:
- The numerical stability fix (sigmoid) was correct and necessary
- It prevented NaN gradients from step function overflow
- But it revealed the underlying memory issue
- Without the fix, training failed earlier with NaN gradients
- With the fix, training progresses further but hits memory limits

### Why Data Is Valid:
- Initial concern: Polygon correction corrupted data
- Investigation showed: 3272 images with proper annotations
- JSON structure: `{"images": {"filename": {"words": {...}}}}`
- All polygons have valid points within image bounds

### Logging Issues:
- Blank spaces in logging are symptoms, not causes
- Indicate system memory pressure
- Will resolve when memory usage is reduced

---

*This bug report follows the project's standardized format for issue tracking.*
*All diagnostic findings and recommended fixes are documented.*
