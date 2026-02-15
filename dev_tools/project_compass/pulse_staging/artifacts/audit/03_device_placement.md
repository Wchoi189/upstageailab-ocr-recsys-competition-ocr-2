# Device Placement Audit Report

**Phase**: 6.2 - Memory Safety
**Focus**: Device placement verification for PLM masks, padding masks, and tensor consistency
**Date**: 2026-02-12
**Status**: ✅ PASS

---

## Executive Summary

Comprehensive audit of tensor device placement across PARSeq model with PLM and Flash Attention. All critical device placement issues have been addressed in previous fixes. Current implementation correctly handles device migration and maintains tensor consistency throughout forward/backward passes.

**Key Findings**:
- ✅ PLM mask device placement fix verified (architecture.py:251)
- ✅ Padding masks created on correct device
- ✅ Device migration properly propagated to PLM module
- ✅ No CPU→GPU transfers detected in hot path
- ⚠️ Test suite created to monitor device consistency

---

## 1. PLM Mask Device Placement

### Status: ✅ FIXED (Verified)

**Location**: `ocr/domains/recognition/models/architecture.py:251`

**Issue**: PLM attention masks were created on CPU and needed explicit device transfer to match target tensors.

**Current Implementation**:
```python
# Line 247-251
tgt_mask = masks.content_mask.float()
tgt_mask = tgt_mask.masked_fill(tgt_mask == 1.0, float('-inf'))
tgt_mask = tgt_mask.masked_fill(tgt_mask == 0.0, 0.0)
# Fix: Move mask to same device as targets
tgt_mask = tgt_mask.to(targets.device)
```

**Verification**:
- ✅ Mask explicitly moved to `targets.device` before decoder call
- ✅ No runtime errors in production runs
- ✅ Works with both CPU and CUDA devices

**Root Cause**:
PLM module creates masks on `self._device` which may not match input tensor device during initial module creation. The `.to()` call ensures consistency.

**Alternative Solution** (Not Implemented):
Could update PLM module to infer device from input tensors:
```python
# In plm.py:gen_tgt_perms
device = tgt.device  # Infer from input instead of self._device
perms = torch.arange(max_num_chars, device=device)
```

**Recommendation**: Keep current fix. It's explicit and safe. Alternative would require PLM API changes.

---

## 2. PLM Module Device Propagation

### Status: ✅ CORRECT

**Location**: `ocr/domains/recognition/models/decoder.py:110-125`

**Implementation**:
```python
def to(self, *args, **kwargs):
    """Override to() to move PLM module to the correct device."""
    super().to(*args, **kwargs)
    if self.plm is not None:
        # Extract device from args/kwargs
        device = None
        if args:
            if isinstance(args[0], torch.device):
                device = str(args[0])
            elif isinstance(args[0], str):
                device = args[0]
        if device is None and 'device' in kwargs:
            device = str(kwargs['device'])
        if device is not None:
            self.plm.to(device)
    return self
```

**Analysis**:
- ✅ Correctly overrides `to()` method to propagate device changes
- ✅ Handles both positional and keyword arguments
- ✅ Converts `torch.device` to string for PLM API compatibility
- ✅ Preserves return value for method chaining

**Validation**:
Created test suite to verify device migration:
- `test_model_to_cuda()` - CPU → CUDA migration
- `test_model_to_cpu()` - CUDA → CPU migration

**Coverage**:
```
pulse_staging/artifacts/tests/test_device_placement.py:TestDeviceMigration
```

---

## 3. PLM Tensor Creation

### Status: ✅ CORRECT

**Location**: `ocr/domains/recognition/models/plm.py`

**Implementation Analysis**:

All tensor creation in PLM module uses explicit `device` parameter:

```python
# Line 102: Permutation generation for 1-char sequences
return torch.arange(3, device=self._device).unsqueeze(0)

# Line 103: Initial permutation
perms = [torch.arange(max_num_chars, device=self._device)]

# Line 115-118: Permutation pool
perm_pool = torch.as_tensor(
    list(permutations(range(max_num_chars), max_num_chars)),
    device=self._device,
)

# Line 127: Random permutations
torch.randperm(max_num_chars, device=self._device)

# Line 154: Attention masks
mask = torch.zeros((sz, sz), dtype=torch.bool, device=self._device)

# Line 160: Eye mask for self-attention
mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = True
```

**Verification**:
- ✅ All tensor operations specify `device=self._device`
- ✅ No implicit CPU tensor creation
- ✅ Consistent device handling across all methods

---

## 4. Padding Mask Device Consistency

### Status: ✅ CORRECT

**Location**: `ocr/domains/recognition/models/decoder.py`

**Memory Key Padding Mask**:
```python
# Line 158: Default mask creation
if memory_key_padding_mask is None:
    memory_key_padding_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
```
- ✅ Created on same device as input `memory` tensor

**Target Key Padding Mask**:
```python
# Line 194: Derived from target tokens
tgt_key_padding_mask = (targets == self.pad_token_id)
```
- ✅ Derived from `targets` tensor, automatically inherits device

**Flash Attention Mask Handling**:
```python
# flash_attention.py:210-213
key_padding_mask = key_padding_mask.view(B, 1, 1, S)
if attn_mask is None:
    attn_mask = torch.zeros(1, 1, T, S, dtype=q.dtype, device=q.device)
attn_mask = attn_mask.masked_fill(key_padding_mask, float('-inf'))
```
- ✅ Attention mask created on `q.device` (query tensor device)
- ✅ Mask operations preserve device

---

## 5. Causal Mask Device Placement

### Status: ✅ CORRECT

**Location**: `ocr/domains/recognition/models/decoder.py:190`

**Implementation**:
```python
if tgt_mask is None:
    # Standard causal mask for AR decoding
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
```

**Analysis**:
- ✅ Explicitly passes `device` parameter (requires PyTorch >= 2.1)
- ✅ Mask created on same device as input tensors
- ✅ No device transfer overhead

**Note**: PyTorch 2.0 doesn't support `device` parameter in `generate_square_subsequent_mask`. Current codebase requires PyTorch >= 2.1.

---

## 6. Device Consistency Test Suite

### Status: ✅ IMPLEMENTED

**Location**: `pulse_staging/artifacts/tests/test_device_placement.py`

**Test Coverage**:

1. **Forward Hook Monitoring** (`DeviceTracker` class)
   - Registers hooks on all leaf modules
   - Tracks input/output tensor devices
   - Validates all tensors on expected device

2. **Configuration Tests**:
   - `test_baseline_ar_device_consistency` - Standard AR mode
   - `test_plm_device_consistency` - PLM training mode
   - `test_flash_attention_device_consistency` - Flash Attention mode

3. **Component Tests**:
   - `test_plm_mask_device_placement` - PLM mask generation
   - `test_padding_mask_device_consistency` - Padding mask creation

4. **Device Migration Tests**:
   - `test_model_to_cuda` - CPU → CUDA migration
   - `test_model_to_cpu` - CUDA → CPU migration

5. **Mixed Precision Test**:
   - `test_mixed_precision_device_consistency` - Device consistency under autocast

**Usage**:
```bash
# Run all device placement tests
uv run pytest dev_tools/project_compass/pulse_staging/artifacts/tests/test_device_placement.py -v

# Run specific test
uv run pytest dev_tools/project_compass/pulse_staging/artifacts/tests/test_device_placement.py::TestDevicePlacement::test_plm_device_consistency -v -s
```

---

## 7. Potential Issues (Low Priority)

### Issue: Device String Conversion

**Location**: `decoder.py:110-125`
**Severity**: LOW
**Category**: Code Quality

**Description**:
PLM module uses string device representation while PyTorch uses `torch.device`. Requires conversion in `to()` override.

**Current Code**:
```python
device = str(args[0])  # Convert torch.device → str
self.plm.to(device)
```

**Impact**:
- No functional issue
- Minor inefficiency (string parsing)
- Inconsistent with PyTorch conventions

**Recommendation**:
Low priority. Consider updating PLM to accept `torch.device` in future refactor:
```python
# In plm.py
def to(self, device: Union[str, torch.device]):
    if isinstance(device, torch.device):
        device = str(device)
    self._device = device
    return self
```

---

## 8. Performance Considerations

### No CPU→GPU Transfers in Hot Path

**Analysis**:
Profiled training loop with device hooks. No unexpected device transfers detected during:
- Forward pass through decoder
- PLM permutation loop
- Flash Attention operations
- Backward pass gradient flow

**Validation Method**:
```python
# Register CUDA memory transfer hooks
torch.cuda.synchronize()
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    output = model(**batch)
    loss = output["loss"]
    loss.backward()
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**Result**: No `[CUDA memcpy HtoD]` or `[CUDA memcpy DtoH]` events in hot path.

---

## Validation Checklist

- ✅ PLM mask device placement verified (architecture.py:251)
- ✅ PLM module `to()` override tested (decoder.py:110-125)
- ✅ All tensor creation uses explicit device parameter (plm.py)
- ✅ Padding masks created on correct device (decoder.py:158, 194)
- ✅ Causal masks created on correct device (decoder.py:190)
- ✅ Flash Attention mask handling verified (flash_attention.py:210-213)
- ✅ Device migration tests implemented and passing
- ✅ Forward hook tests verify runtime device consistency
- ✅ No CPU→GPU transfers in training loop

---

## Recommendations

### Immediate Actions

**None Required** - All device placement issues addressed.

### Future Improvements (Optional)

1. **PLM Device API** (Low Priority)
   - Update PLM to accept `torch.device` natively
   - Remove string conversion in decoder `to()` override
   - **Effort**: 1 hour
   - **Impact**: Code clarity

2. **Continuous Monitoring** (Medium Priority)
   - Add device placement tests to CI/CD pipeline
   - Run on both CPU and CUDA environments
   - **Effort**: 2 hours
   - **Impact**: Prevent regressions

3. **Performance Profiling** (Medium Priority)
   - Add CUDA memory transfer tracking to training logs
   - Alert on unexpected device synchronization
   - **Effort**: 3 hours
   - **Impact**: Performance monitoring

---

## References

- **Phase 6.2 Schema**: `MERGED_AUDIT_SCHEMA.yaml:phase_6_2.device_placement`
- **Previous Fix**: Phase 6.1 - PLM mask device placement (architecture.py:251)
- **Test Suite**: `pulse_staging/artifacts/tests/test_device_placement.py`
- **PyTorch Docs**: [torch.nn.Module.to()](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to)

---

**Audit Completed**: 2026-02-12
**Auditor**: Claude Sonnet 4.5
**Status**: ✅ All device placement checks passed
