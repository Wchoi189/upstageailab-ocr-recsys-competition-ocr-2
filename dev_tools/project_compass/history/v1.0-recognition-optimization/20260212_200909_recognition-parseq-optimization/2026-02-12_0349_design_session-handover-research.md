# Session Handover: PARSeq Optimization - Research Complete

**Date**: 2026-02-12
**Pulse ID**: recognition-parseq-optimization
**Milestone**: v1.0-recognition-optimization
**Phase**: Recognition
**Status**: Research Complete - Ready for Implementation

---

## 🎯 Executive Summary

Comprehensive research completed for PARSeq refactoring. **Ready to proceed with Phase 1 implementation**.

**Key Finding**: NO existing modular PARSeq implementations found - we must extract carefully from working reference.

**Strategy**: Extract and validate, NOT reimplement from scratch.

**Risk Level**: HIGH - PLM logic is complex, debugging is costly (~8-12 days estimated).

---

## 📊 Research Summary

### Perplexity Research Findings

#### 1. PLM Core Mechanics
- **K Permutations**: Uses 6 random permutations (not all T! permutations)
- **Mirrored Pairs**: K/2 base + K/2 complementary (reversed)
- **Loss Weighting**: Weighted by character count, EOS removed after 2nd permutation
- **Inference**: Switches to standard left-to-right (no permutations)

#### 2. Flash Attention Constraints
✅ **RTX 3090 Compatible** (Ampere, sm_80)
- Requires fp16/bfloat16 (NO fp32)
- Head dimension % 8 == 0 (our 384/12=32 ✓)
- PyTorch ≥2.0 (≥2.2 for optimal)
- Auto-selects on Ampere+ via `sdp_kernel`
- Expected: 2-4x throughput improvement

#### 3. Numerical Behavior
- Flash Attention: Exact math, fp16 precision (ε ≤ 1e-3)
- Standard Attention: fp32 precision (ε ≤ 1e-5)
- Some drift acceptable when switching precision

### Implementation Search Results
- ❌ NO modular PARSeq implementations found on GitHub
- ✅ Official baudm/parseq (monolithic only)
- ✅ Our parseq_official_adapter.py (working reference)

**Implication**: We're pioneering this refactoring - must be extremely careful.

---

## 📋 Artifacts Generated

All artifacts in: `dev_tools/project_compass/pulse_staging/artifacts/`

### 1. Constitution
**File**: `constitution.md`

Establishes research-backed principles:
- Extract, don't reimplement
- PLM correctness over optimization
- Incremental validation (PLM → Flash)
- Numerical equivalence requirements

### 2. Specification
**File**: `specification.md`

Comprehensive requirements:
- **FR1-4**: Functional requirements (PLM extraction, training loop, inference, decoder interface)
- **PF1-3**: Performance requirements (Flash Attention 2x, memory, convergence)
- **AR1-4**: Architecture requirements (PLM module, decoder, Flash layer, configs)
- **VR1-4**: Validation requirements (unit tests, integration, benchmarks, edge cases)
- **Risk Assessment**: 5 critical risks with mitigation strategies

### 3. Implementation Plan
**File**: `implementation_plan.md`

5-phase execution strategy:
- **Phase 1**: PLM Module Extraction (2-3 days) - CRITICAL
- **Phase 2**: Atomic Decoder Integration (2-3 days)
- **Phase 3**: Flash Attention (1-2 days)
- **Phase 4**: Configuration (1 day)
- **Phase 5**: Full Validation (2-3 days)

**Total**: 8-12 days conservative estimate

### 4. Walkthrough
**File**: `walkthrough_parseq_plm_flash_refactor.md`

Detailed implementation guide:
- Research findings consolidated
- Phase-by-phase code examples
- Critical gotchas & debugging
- Validation checklist
- Performance expectations

---

## ⚠️ Critical Insights

### 1. EOS Removal Logic (Most Common Bug)
```python
for i, perm in enumerate(perms):
    # ... compute loss ...

    # CRITICAL: Remove EOS after 2nd permutation
    if i == 1:
        tgt_out = torch.where(tgt_out == eos_id, pad_id, tgt_out)
        n = (tgt_out != pad_id).sum().item()
```

**Why**: Prevents over-weighting EOS token across all permutations.
**Bug**: Forgetting this causes loss divergence.

### 2. Hardcoded Selector for 4-Char Sequences
```python
if max_num_chars == 4 and self.perm_mirrored:
    selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
```

**Why**: Exhaustive enumeration with specific subset for 4-char.
**Bug**: Wrong selector breaks permutation distribution.

### 3. Flash Attention Precision
```python
# MUST use fp16/bf16
with torch.cuda.amp.autocast():
    attn_output = F.scaled_dot_product_attention(...)
```

**Why**: Flash Attention doesn't support fp32.
**Bug**: Using fp32 falls back to slow attention.

---

## 📁 Key Source Files

### Reference Implementation (Extract From)
- **parseq_official_adapter.py**:
  - Lines 92-140: `gen_tgt_perms()`
  - Lines 142-156: `generate_attn_masks()`
  - Lines 204-248: `forward_train()` with PLM loop

### Target Files (Implement In)
- **ocr/domains/recognition/models/plm.py** (NEW)
  - `PermutationLanguageModeling` class
  - Extract functions from adapter
- **ocr/domains/recognition/models/decoder.py** (MODIFY)
  - Add PLM integration
  - Add mode parameter (train/inference)
- **ocr/domains/recognition/models/flash_attention.py** (NEW)
  - `FlashDecoderLayer`
  - `FlashMultiheadAttention`

### Configuration Files (Create)
- **configs/model/architectures/parseq_atomic_flash.yaml**
- **configs/experiment/rec_atomic_flash.yaml**

---

## 🧪 Testing Strategy

### Phase 1: Unit Tests (Blocking)
```python
# tests/unit/recognition/test_plm.py
def test_gen_tgt_perms_equivalence():
    # Test 1-char, 4-char, random cases
    # MUST match parseq_official_adapter.py exactly

def test_generate_attn_masks_equivalence():
    # Verify content_mask, query_mask
    # MUST match reference

def test_loss_aggregation():
    # Verify EOS removal logic
    # Verify character count weighting
```

**Success Criteria**: 100% pass rate before Phase 2.

### Phase 2: Integration Tests
```python
# tests/integration/test_atomic_plm.py
def test_atomic_vs_monolithic_equivalence():
    # Numerical equivalence: ε ≤ 1e-5
    assert torch.allclose(atomic_loss, ref_loss, atol=1e-5)
```

### Phase 3: Performance Benchmarks
```python
def benchmark_throughput():
    # Measure img/sec for:
    # - Monolithic baseline
    # - Atomic + Standard Attn
    # - Atomic + Flash Attn
    # Target: 2x improvement
```

---

## 📈 Performance Expectations

| Configuration | Throughput | VRAM | Training Time |
|---------------|-----------|------|---------------|
| Monolithic Baseline | ~120 img/sec | ~18 GB | ~8 hrs/epoch |
| Atomic + Standard | ~100-110 img/sec | ~18 GB | ~8-9 hrs/epoch |
| **Atomic + Flash** | **~240-300 img/sec** | **~16 GB** | **~3-4 hrs/epoch** |

**Target**: 2-4x speedup (✓ achievable)

---

## 🚦 Phase 1 Start Checklist

Before starting implementation:

- [x] Research complete (Perplexity, PyTorch docs)
- [x] Specification finalized
- [x] Implementation plan approved
- [x] Walkthrough document created
- [x] Reference code identified (parseq_official_adapter.py)
- [x] Test strategy defined
- [x] Success criteria established

**Status**: ✅ **READY TO START PHASE 1**

---

## 🎬 Next Session: Phase 1 Start

### Immediate Tasks

1. **Create PLM Module** (`ocr/domains/recognition/models/plm.py`):
   ```bash
   touch ocr/domains/recognition/models/plm.py
   ```

2. **Extract gen_tgt_perms()**:
   - Copy lines 92-140 from parseq_official_adapter.py
   - Preserve logic exactly
   - Add device handling

3. **Extract generate_attn_masks()**:
   - Copy lines 142-156 from parseq_official_adapter.py
   - Preserve logic exactly

4. **Create Unit Tests** (`tests/unit/recognition/test_plm.py`):
   ```bash
   mkdir -p tests/unit/recognition
   touch tests/unit/recognition/test_plm.py
   ```

5. **Run Tests** (MUST PASS):
   ```bash
   pytest tests/unit/recognition/test_plm.py -v
   ```

### Continuation Prompt

```
Continue PARSeq optimization implementation - Phase 1: PLM Module Extraction.

Context:
- Research complete (see session_handover_research_complete.md)
- Specification finalized (see specification.md)
- Implementation plan ready (see implementation_plan.md)
- Walkthrough guide available (see walkthrough_parseq_plm_flash_refactor.md)

Task:
1. Create ocr/domains/recognition/models/plm.py
2. Extract gen_tgt_perms() from parseq_official_adapter.py:92-140
3. Extract generate_attn_masks() from parseq_official_adapter.py:142-156
4. Create unit tests in tests/unit/recognition/test_plm.py
5. Verify 100% test pass rate

References:
- parseq_official_adapter.py (source)
- walkthrough_parseq_plm_flash_refactor.md (guide)
- specification.md (requirements)

CRITICAL: Do NOT reimplement from scratch - extract exact logic.
```

---

## 📚 Documentation References

### Framework Standards
- **Hydra V5**: `AgentQMS/specs/tier2-framework/patterns.spec.md`
- **Core Interfaces**: `AgentQMS/specs/tier2-framework/core-interfaces.spec.md`

### Research Sources
- **Perplexity Research**: PLM mechanics, Flash Attention constraints
- **PyTorch Docs**: F.scaled_dot_product_attention API
- **PARSeq Paper**: https://arxiv.org/abs/2207.06966
- **Official Repo**: https://github.com/baudm/parseq

### Project Compass
- **Pulse State**: `dev_tools/project_compass/.vessel/vessel_state.json`
- **Artifacts**: `dev_tools/project_compass/pulse_staging/artifacts/`

---

## ✅ Success Metrics

### Must Have (Blocking)
- ✅ PLM logic matches reference exactly (ε ≤ 1e-5)
- ✅ Training converges (loss ↓ monotonically)
- ✅ No NaN/Inf gradients

### Should Have (High Priority)
- ✅ Flash Attention 2x speedup achieved
- ✅ CER within ±0.5% of baseline
- ✅ VRAM usage ≤ baseline

### Nice to Have (Optional)
- ✅ 2.5x+ speedup (exceeds target)
- ✅ Improved convergence speed
- ✅ Lower memory footprint

---

## 🔄 Session Transfer

**From**: Requirements gathering + research session
**To**: Implementation session (Phase 1)

**Context Loaded**:
- ✅ Audit directives reviewed
- ✅ Research completed (Perplexity)
- ✅ Specifications finalized
- ✅ Implementation plan generated
- ✅ Walkthrough guide created

**Next Agent Should**:
1. Read walkthrough document first
2. Start Phase 1 immediately
3. Follow extraction strategy exactly
4. Validate against reference continuously
5. Do NOT proceed to Phase 2 until all Phase 1 tests pass

**Estimated Time to Completion**: 8-12 days (conservative)

---

**Status**: 🟢 Research Complete - Ready for Implementation
**Blocking Issues**: None
**Risk Level**: HIGH (manageable with careful extraction)
**Confidence**: HIGH (working reference exists)
