# Phase 6 Audit - Continuation Prompt

## Quick Start

Execute the comprehensive PARSeq recognition pipeline audit using the **merged schema**:

📋 **Schema**: `/workspaces/dev_tools/project_compass/pulse_staging/artifacts/MERGED_AUDIT_SCHEMA.yaml`

This schema integrates:
- ✅ Original continuation prompt structure (Phases 6.1-6.3)
- ✅ Perplexity's 8 targeted directives with references
- ✅ Priority matrix with effort/impact assessment
- ✅ Production recommendations and actionable fixes

---

## Context Snapshot

**Previous Pulse**: `recognition-parseq-optimization` (Phase 1-5)
- ✅ All 4 configs validated (baseline, flash, plm, plm_flash)
- ✅ Bugs fixed: multiprocessing spawn, PLM device placement
- ⚠️ Flash Attention: 0.92x speedup (disappointing), potential numerical drift

**Current State**:
- Baseline AR: 675 img/sec
- PLM: ~169 img/sec (expected 4x slowdown for K=6)
- Flash: 618 img/sec (slower than baseline)
- PLM+Flash CER: 2.67 vs 1.76 baseline (⚠️ drift concern)

**Root Causes Identified** (from Perplexity):
1. Warmup overhead (JIT compilation eating first steps)
2. Seq/batch mismatch (seq=25, batch=64 too small for Flash benefits)
3. Custom masks may force fallback to MATH kernel
4. Short runs (100 steps) don't amortize compilation cost

---

## Execution Strategy

### High-Priority Path (Start Here)

Follow the **priority_matrix** from merged schema:

#### 🔴 HIGH (Do First - Low Effort, High Impact)

1. **Backend Confirmation** (`perplexity_directive_1`)
   ```python
   # In decoder.py, log SDP backend selection
   import torch.backends.cuda
   print(f"SDP Backend: {torch.backends.cuda.sdp_kernel()}")
   # Expected: should show FLASH, not MATH
   ```
   - **Location**: `ocr/domains/recognition/models/decoder.py:FlashDecoderLayer`
   - **Deliverable**: `audit/02_flash_attention.md` section 1

2. **Numerical Drift Check** (`perplexity_directive_2`)
   ```python
   # Compare Flash vs MHA outputs
   max_diff = torch.max(torch.abs(flash_out - mha_out))
   assert max_diff < 1e-3, f"Drift: {max_diff}"
   ```
   - **Location**: `pulse_staging/artifacts/tests/test_flash_equivalence.py`
   - **Deliverable**: `audit/02_flash_attention.md` section 2
   - **Tolerance**: atol=1e-3, rtol=1e-3

3. **Warmup Profiler** (`perplexity_directive_3`)
   ```python
   # Add torch.profiler to training loop
   with torch.profiler.profile(
       activities=[torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA],
       schedule=torch.profiler.schedule(wait=0, warmup=2, active=98),
   ) as prof:
       # training loop
   ```
   - **Location**: `scripts/runners/train.py` or `ocr/domains/recognition/module.py`
   - **Deliverable**: `audit/06_performance.md` section 1

#### 🟡 MEDIUM (Do After High Priority)

4. **Long-Run Benchmark** (`perplexity_directive_8`)
   - Rerun with 1000 steps, batch=128, seq_len padded to 128
   - **Expected**: 2-4x speedup when amortized
   - **Deliverable**: `audit/06_performance.md` section 2

5. **Custom Layer Logic Audit** (`perplexity_directive_4`)
   - Review `FlashDecoderLayer` for unnecessary transposes/copies
   - **Deliverable**: `audit/02_flash_attention.md` section 3

6. **Loss Divergence Check** (`perplexity_directive_7`)
   - Verify EOS removal at i=1 (not i=0)
   - Confirm loss_numel normalization
   - **Deliverable**: `audit/01_plm_correctness.md` section 2

---

## Phase Execution Order

### Phase 6.1: Critical Correctness (🔴 CRITICAL)
**Focus**: Ensure implementation is logically correct

**Tasks**:
1. PLM Implementation Review
   - Permutation generation (K=6)
   - Loss computation (verify `loss / loss_numel`)
   - EOS removal timing (after 2nd permutation, i=1)
   - Attention mask conversion (boolean → additive)

2. Flash Attention Validation
   - Backend confirmation (directive 1)
   - Numerical equivalence (directive 2)
   - Custom layer audit (directive 4)
   - Mask compatibility (directive 5)

**Deliverables**:
- `audit/01_plm_correctness.md`
- `audit/02_flash_attention.md`
- `tests/test_plm_correctness.py`
- `tests/test_flash_equivalence.py`

---

### Phase 6.2: Memory Safety (🟡 HIGH)
**Focus**: Verify device placement and gradient flow

**Tasks**:
1. Device Placement
   - PLM mask device consistency (already fixed, verify)
   - Padding mask device matching
   - Forward hook validation

2. Gradient Flow
   - Backprop through K permutations
   - No detached tensors
   - Mixed precision scaling (directive 6)

**Deliverables**:
- `audit/03_device_placement.md`
- `audit/04_gradient_flow.md`
- `tests/test_device_placement.py`

---

### Phase 6.3: Performance Analysis (🟢 MEDIUM)
**Focus**: Understand and optimize Flash Attention performance

**Tasks**:
1. Root Cause Investigation
   - Warmup profiling (directive 3)
   - Long-run benchmark (directive 8)
   - Batch/sequence sweep

2. Configuration Validation
   - Hydra composition correctness
   - Vocab size injection
   - Architecture overrides

**Deliverables**:
- `audit/06_performance.md`
- `audit/05_configuration.md`
- `recommendations/production_config.md`

---

## Deliverable Structure

### Audit Reports (`pulse_staging/artifacts/audit/`)
1. `01_plm_correctness.md` - Permutation logic, loss, EOS
2. `02_flash_attention.md` - Backend, numerical equivalence, performance
3. `03_device_placement.md` - Tensor device consistency
4. `04_gradient_flow.md` - Backpropagation verification
5. `05_configuration.md` - Hydra composition
6. `06_performance.md` - Flash investigation with profiling

### Test Suite (`pulse_staging/artifacts/tests/`)
- `test_plm_correctness.py` - Unit tests for PLM
- `test_flash_equivalence.py` - Numerical tolerance tests
- `test_edge_cases.py` - Empty sequences, max length
- `test_device_placement.py` - Forward hook device checks

### Findings (`pulse_staging/artifacts/findings/`)
- `critical_issues.md` (🔴) - Must fix before production
- `high_priority_issues.md` (🟡) - Should fix soon
- `medium_priority_issues.md` (🟢) - Can defer

### Recommendations (`pulse_staging/artifacts/recommendations/`)
- `immediate_fixes.md` - Code patches for critical issues
- `production_config.md` - Deployment recommendations
  - **Training**: PLM+Flash, warmup enabled, batch≥128
  - **Inference**: Pure Flash for autoregressive
  - **Fallback**: PLM baseline if timeline tight

---

## Issue Reporting Template

Use this format in audit reports (from merged schema):

```markdown
### Issue: [Short title]
**Location**: `file/path.py:line_number`
**Severity**: CRITICAL | HIGH | MEDIUM
**Category**: Flash Attention | PLM Correctness | Device Placement
**Reference**: perplexity_directive_N (if applicable)

**Description**:
[What's wrong and why it matters]

**Current Code**:
```python
# Problematic code
```

**Recommendation**:
```python
# Fixed code
```

**Validation**:
[How to test the fix]

**Impact**:
[What breaks without this fix]
```

---

## Success Criteria

Audit complete when:

- ✅ Flash backend confirmed (FLASH not MATH)
- ✅ Numerical equivalence proven (max_diff < 1e-3)
- ✅ Performance bottlenecks identified via profiling
- ✅ All critical correctness issues documented
- ✅ Device placement verified for all paths
- ✅ Gradient flow validated
- ✅ Production config recommended

**Minimum Exit**: All ✅ items above completed
**Ideal Exit**: + test suite implemented + long-run benchmark complete

---

## Key Files Reference

### Critical Path
- `ocr/domains/recognition/models/architecture.py:217-279` - PLM implementation
- `ocr/domains/recognition/models/decoder.py` - Flash Attention + PLM decoder
- `ocr/domains/recognition/module.py` - Training/validation steps
- `scripts/runners/train.py` - Entry point, multiprocessing

### Configuration
- `configs/experiment/parseq_*.yaml` - 4 variants
- `configs/model/architectures/parseq_*.yaml` - Architecture defs
- `configs/model/decoder/parseq_*.yaml` - Decoder config

---

## Quick Commands

```bash
# Run specific config for testing
uv run train experiment=parseq_flash_attention trainer.max_steps=1000

# Run test suite (once created)
pytest pulse_staging/artifacts/tests/ -v

# Profile training run
uv run train experiment=parseq_flash_attention \
  trainer.max_steps=100 \
  trainer.profiler=pytorch
```

---

## Next Actions

1. **Read merged schema**: Familiarize with all 8 directives and their mapping
2. **Start with HIGH priority**: Backend confirmation, numerical drift, warmup profiler
3. **Create test suite first**: Enables systematic validation as you audit
4. **Document as you go**: Use issue template for findings
5. **Long-run test last**: After high-priority checks to avoid wasting compute

---

**Estimated Duration**: 2-4 sessions
**Risk Level**: MEDIUM (complex implementation, needs thorough validation)
**Blocking Issues**: None identified (previous fixes resolved CUDA/device errors)

---

**📌 Remember**: The merged schema is your single source of truth. It contains all phase details, directive mappings, testing templates, and success criteria.
