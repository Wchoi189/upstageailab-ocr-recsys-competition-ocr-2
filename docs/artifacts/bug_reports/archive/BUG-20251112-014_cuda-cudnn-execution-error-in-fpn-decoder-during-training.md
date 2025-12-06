---
title: "Bug 20251112 014 Cuda Cudnn Execution Error In Fpn Decoder During Training"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---





## Summary

Training fails with CUDA/cuDNN execution error during forward pass in FPN decoder. Error occurs after processing 14 batches (14/818) with `CUDNN_STATUS_EXECUTION_FAILED` followed by `CUDA error: an illegal instruction was encountered`.

## Environment

- **Pipeline Version:** Current main branch (post data-contract integration)
- **Components:**
  - `ocr.models.decoder.fpn_decoder.FPNDecoder` (line 68: `self.fusion(fused)`)
  - `torch.nn.modules.conv.Conv2d` forward pass
  - Lightning training loop
- **Configuration:**
  - Model: ResNet18 encoder
  - Batch size: 2 (test dataloader)
  - Devices: 1 GPU
  - Strategy: auto
  - Max steps: 10
- **Hardware:** CUDA-enabled GPU (specific model unknown from logs)
- **Error Location:** Training step 14/818, during optimizer step closure

## Steps to Reproduce

1. Run training with configuration:
   ```bash
   # Command with overrides:
   model.encoder.model_name=resnet18
   dataloaders.test_dataloader.num_workers=0
   dataloaders.test_dataloader.batch_size=2
   trainer.max_steps=10
   trainer.devices=1
   trainer.strategy=auto
   ```

2. Training starts successfully, processes 14 batches
3. Error occurs during forward pass in FPN decoder fusion layer
4. CUDA error propagates, causing training failure

## Expected Behavior

Training should proceed through all batches without CUDA/cuDNN errors. The FPN decoder should successfully execute conv2d operations.

## Actual Behavior

**Primary Error:**
```
RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
```

**Stack Trace:**
```
File "ocr/models/decoder/fpn_decoder.py", line 68, in forward
    return self.fusion(fused)
File "torch/nn/modules/conv.py", line 548, in forward
    return self._conv_forward(input, self.weight, self.bias)
File "torch/nn/modules/conv.py", line 543, in _conv_forward
    return F.conv2d(...)
```

**Secondary Error (during teardown):**
```
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

**Training Progress:**
- Successfully processed: 14/818 batches (1.7%)
- Error occurred at: Batch 14, during optimizer step
- Time elapsed: ~8 seconds before failure

## Root Cause Analysis

**Potential Causes:**

1. **CUDA/cuDNN Version Mismatch**: The error `CUDNN_STATUS_EXECUTION_FAILED` suggests a compatibility issue between PyTorch, CUDA toolkit, and cuDNN versions.

2. **Hardware Compatibility**: The "illegal instruction" error may indicate:
   - GPU architecture incompatibility (e.g., older GPU with newer CUDA)
   - Corrupted CUDA driver installation
   - GPU hardware failure

3. **Memory Corruption**: The error occurs after processing multiple batches, suggesting:
   - Gradual memory corruption
   - Out-of-bounds tensor access
   - Invalid tensor shapes/values passed to conv2d

4. **Tensor Validation Gap**: Despite recent data contract integration, the error occurs in the decoder which may not have comprehensive tensor validation:
   - Invalid tensor values (NaN/Inf) propagating to conv2d
   - Shape mismatches in decoder fusion layer
   - Device mismatch (tensors on wrong device)

**Code Path:**
```
training_step()
├── self.model(**batch)
│   ├── encoder.forward()
│   ├── decoder.forward()  # FPN decoder
│   │   └── self.fusion(fused)  # Line 68 - ERROR HERE
│   │       └── Conv2d.forward()
│   │           └── F.conv2d()  # cuDNN call fails
│   └── head.forward()
└── loss computation
```

**Observations:**
- Error is not immediate (occurs after 14 batches)
- Suggests state-dependent issue (memory, tensor values, or batch-specific)
- The "illegal instruction" error during teardown suggests CUDA context corruption

## Resolution

**Immediate Actions:**

1. **Enable CUDA Device-Side Assertions:**
   ```bash
   export TORCH_USE_CUDA_DSA=1
   # Re-run training to get more detailed error information
   ```

2. **Add Tensor Validation in Decoder:**
   - Add `ValidatedTensorData` checks before fusion layer
   - Validate tensor shapes, devices, and values (NaN/Inf)
   - Add logging to capture tensor state before failure

3. **Verify CUDA Environment:**
   ```bash
   # Check CUDA version compatibility
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"
   nvidia-smi  # Check GPU and driver version
   ```

4. **Add Defensive Checks:**
   ```python
   # In fpn_decoder.py, before fusion:
   def forward(self, features):
       fused = self._fuse_features(features)

       # Add validation
       if torch.isnan(fused).any() or torch.isinf(fused).any():
           raise ValueError(f"Invalid values in fused features: nan={torch.isnan(fused).any()}, inf={torch.isinf(fused).any()}")

       # Validate device
       if not fused.is_cuda:
           raise ValueError(f"Fused features not on CUDA: device={fused.device}")

       return self.fusion(fused)
   ```

**Long-term Solutions:**

1. **Extend Data Contract Validation:**
   - Add `ValidatedTensorData` checks in decoder forward pass
   - Validate intermediate decoder outputs
   - Add shape/device validation at decoder boundaries

2. **CUDA Environment Documentation:**
   - Document required CUDA/cuDNN versions
   - Add compatibility matrix
   - Provide troubleshooting guide for CUDA errors

3. **Error Handling:**
   - Add graceful degradation for CUDA errors
   - Implement fallback to CPU if CUDA fails
   - Add better error messages with diagnostic information

## Testing

- [ ] Reproduce error with `TORCH_USE_CUDA_DSA=1` for detailed diagnostics
- [ ] Verify CUDA/cuDNN version compatibility
- [ ] Test with different batch sizes (may be batch-specific)
- [ ] Test with CPU fallback to isolate CUDA-specific issue
- [ ] Add tensor validation in decoder and verify it catches issues
- [ ] Test with different model architectures (ResNet variants)
- [ ] Verify error occurs consistently or is intermittent

## Prevention

1. **Pre-commit Checks:**
   - Add CUDA version validation in CI/CD
   - Test with multiple CUDA versions
   - Add hardware compatibility tests

2. **Runtime Validation:**
   - Extend data contract validation to decoder layers
   - Add CUDA error detection and graceful handling
   - Implement automatic fallback mechanisms

3. **Documentation:**
   - Document CUDA/cuDNN requirements
   - Add troubleshooting section for CUDA errors
   - Provide compatibility matrix

## Related Issues

- May be related to BUG-20251112-013 (CUDA memory access errors) - similar CUDA error pattern
- Data contract validation (recently integrated) may help prevent this if extended to decoder

## Additional Context

**Import Time Profiling:**
The logs show very long import times:
- `numba`: 1.49 seconds
- `tensorboard`: 1.0+ seconds
- `ocr.lightning_modules`: 41+ seconds total

While not directly related to this bug, these long import times suggest optimization opportunities (PLAN-003: Import-Time Optimization).

**Training Configuration:**
- Performance preset: none (baseline)
- No optimizations enabled
- Standard training configuration

**Error Timing:**
- Error occurs after successful processing of 14 batches
- Suggests state-dependent or batch-specific issue
- Not an immediate initialization problem
