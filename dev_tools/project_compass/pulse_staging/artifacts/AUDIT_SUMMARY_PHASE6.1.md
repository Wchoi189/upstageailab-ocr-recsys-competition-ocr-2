# Phase 6.1 Audit Summary

**Session**: 2026-02-12
**Directives**: Perplexity #1, #2, #3 (HIGH priority)
**Status**: ✅ Complete

---

## Critical Discovery

🔴 **Flash Attention NOT using Flash Attention 2 backend**

**Backend Used**: `_scaled_dot_product_cudnn_attention` (cuDNN/Memory-Efficient)
**Expected**: Flash Attention 2 kernel (`fmha_*`, `flash_fwd_*`)

**Evidence**:
```
Profiler trace shows:
  aten::_scaled_dot_product_cudnn_attention  8.02% total time
  No Flash2 kernels detected
```

---

## Root Causes

1. **Custom PLM masks** force backend fallback
2. **Context manager missing** in training loop (only in tests)
3. **Head dim=32** suboptimal (Flash prefers 64/128)
4. **Seq len=25** too short (Flash optimal >128)

---

## Impact

| Metric | Expected | Actual | Delta |
|--------|----------|--------|-------|
| Speedup | 2-4x | 0.92x | -108% |
| CER | 1.76 | 2.67 | +52% |
| Numerical drift | <1e-3 | 0.00195 | +95% |

---

## Fixes

**Priority 1** (1-2 hours):
```python
# ocr/domains/recognition/module.py:training_step()
from ocr.domains.recognition.models.flash_attention import enable_flash_attention_kernel

def training_step(self, batch, batch_idx):
    with enable_flash_attention_kernel():  # ADD THIS
        pred = self.model(**batch)
    # ... rest
```

**Priority 2** (Test):
- Use `is_causal=True` for AR decoding (bypass custom masks)
- Re-run benchmarks with batch=128, seq=128
- Verify Flash kernels in profiler

---

## Artifacts

**Created**:
- `audit/02_flash_attention.md` - Full report
- `audit/diagnostic_backend_check.py` - Backend check tool
- `audit/profiler_warmup_directive3.py` - Profiler script
- `tests/test_flash_equivalence_directive2.py` - Drift tests
- `SESSION_HANDOVER_PHASE6.1.md` - Handover doc
- `AUDIT_SUMMARY_PHASE6.1.md` - This file

**Updated**:
- `INDEX.md` - Phase 6.1 status

---

## Quick Commands

```bash
# Verify backend
uv run python dev_tools/project_compass/pulse_staging/artifacts/audit/diagnostic_backend_check.py

# Test numerical equivalence
uv run pytest dev_tools/project_compass/pulse_staging/artifacts/tests/test_flash_equivalence_directive2.py -v -s

# Profile training
uv run python dev_tools/project_compass/pulse_staging/artifacts/audit/profiler_warmup_directive3.py
```

---

## Next Actions

**Option A: Deploy Fix**
1. Add context manager to training
2. Test is_causal=True
3. Re-run 1000-step benchmark
4. Validate CER improves

**Option B: Continue Audit**
1. Phase 6.2: Device placement
2. Phase 6.2: Gradient flow
3. Phase 6.3: Configuration audit

**Recommendation**: Deploy fix first (1-2 hours), then continue audit with working Flash backend.

---

## Context for Next Session

**Schema**: `MERGED_AUDIT_SCHEMA.yaml`
**Continuation**: `CONTINUATION_PROMPT_V2.md`
**Handover**: `SESSION_HANDOVER_PHASE6.1.md`

**Key Insight**: PyTorch SDP backend auto-selection is permissive. Without forcing Flash backend, it falls back to slower cuDNN for custom masks. This is NOT a Flash Attention bug - it's a configuration issue.

**Production Ready**: ❌ Fix required before deployment
**Blocking**: None (workaround available: use PLM baseline)
