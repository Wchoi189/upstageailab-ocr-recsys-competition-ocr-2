# Continuation Prompt for Next Session

## Start PARSeq Recognition Pipeline Audit - Phase 6

### Context

**Previous Pulse**: `recognition-parseq-optimization` (Phase 1-5)
- Exported to: `/history/v1.0-recognition-optimization/20260212_200909_recognition-parseq-optimization/`
- Status: ✅ All 4 configs validated (baseline, flash, plm, plm_flash)
- Bugs Fixed: Multiprocessing (fork→spawn), PLM device placement
- Outstanding: Flash Attention speedup not achieved, untested edge cases

**Current Pulse**: `recognition-parseq-audit` (Phase 6)
- Status: ✅ Initialized and ready
- Objective: Validate correctness, robustness, performance
- Workspace: `/pulse_staging/artifacts/` with audit structure

---

## Task: Execute Comprehensive Audit

Use the audit prompt from the previous pulse to systematically validate the implementation:

**Audit Prompt Location**:
- Exported pulse: `/history/v1.0-recognition-optimization/20260212_200909_recognition-parseq-optimization/artifacts/darft_audit_prompt.md`

### Audit Phases

#### Phase 6.1: Critical Correctness (Priority: 🔴 CRITICAL)
1. **PLM Implementation**
   - Permutation generation correctness
   - Attention mask shape and conversion
   - Loss computation and averaging (verify `loss / loss_numel`)
   - EOS removal timing (after 2nd permutation)
   - Sequence handling (BOS/EOS/padding consistency)

2. **Flash Attention Integration**
   - Numerical equivalence vs standard attention
   - Mask compatibility with PLM
   - Fallback logic for non-Ampere GPUs
   - Mixed precision compatibility

#### Phase 6.2: Memory Safety (Priority: 🟡 HIGH)
3. **Device Placement**
   - Verify all tensors on correct device
   - PLM mask device placement (already fixed, verify)
   - Padding mask device consistency
   - Memory padding mask device

4. **Gradient Flow**
   - Backpropagation through PLM permutations
   - No detached tensors breaking flow
   - No in-place ops causing errors
   - Mixed precision gradient scaling

#### Phase 6.3: Performance Analysis (Priority: 🟢 MEDIUM)
5. **Flash Attention Investigation**
   - Why no speedup in validation? (0.92x vs baseline)
   - Test with longer runs (1000+ steps)
   - Test with larger batches (128, 256)
   - Kernel compilation overhead analysis

6. **Configuration Validation**
   - Hydra composition correctness
   - Vocab size injection
   - Architecture overrides
   - CLI override handling

---

## Key Files to Audit

### Critical Path
1. `/workspaces/ocr/domains/recognition/models/architecture.py` - PLM `_forward_train_plm()` lines 217-279
2. `/workspaces/ocr/domains/recognition/models/decoder.py` - Flash Attention + PLM decoder
3. `/workspaces/ocr/domains/recognition/module.py` - Training/validation steps
4. `/workspaces/scripts/runners/train.py` - Entry point, multiprocessing

### Configuration
5. `/workspaces/configs/experiment/parseq_*.yaml` (4 variants)
6. `/workspaces/configs/model/architectures/parseq_*.yaml`
7. `/workspaces/configs/model/decoder/parseq_*.yaml`

---

## Expected Deliverables

### 1. Audit Reports (in `pulse_staging/artifacts/audit/`)
Create separate reports for each area:
- `01_plm_correctness.md` - Permutation logic, loss computation
- `02_flash_attention.md` - Numerical equivalence, performance
- `03_device_placement.md` - Tensor device consistency
- `04_gradient_flow.md` - Backpropagation verification
- `05_configuration.md` - Hydra composition validation
- `06_performance.md` - Flash Attention investigation

### 2. Test Suite (in `pulse_staging/artifacts/tests/`)
- `test_plm_correctness.py` - Unit tests for PLM logic
- `test_flash_equivalence.py` - Numerical equivalence tests
- `test_edge_cases.py` - Empty sequences, max length, etc.
- `test_device_placement.py` - Device consistency checks

### 3. Findings Report (in `pulse_staging/artifacts/findings/`)
- `critical_issues.md` - 🔴 Must fix before production
- `high_priority_issues.md` - 🟡 Should fix soon
- `medium_priority_issues.md` - 🟢 Can defer

### 4. Recommendations (in `pulse_staging/artifacts/recommendations/`)
- `immediate_fixes.md` - Critical fixes with code
- `production_config.md` - Recommended config for deployment

---

## Output Format (from Audit Prompt)

For each issue found, use this structure:

```markdown
### Issue: [Short title]
**Location**: `file/path.py:line_number`
**Severity**: CRITICAL | HIGH | MEDIUM
**Category**: [PLM Correctness / Flash Attention / Device Placement / Gradient Flow]

**Description**: [What's wrong and why it matters]

**Current Code**:
\`\`\`python
# Show problematic code
\`\`\`

**Recommendation**:
\`\`\`python
# Show fixed code
\`\`\`

**Validation**: [How to test the fix]

**Impact**: [What breaks without this fix]
```

---

## Testing Methodology

### Correctness Validation
```python
# Test PLM loss computation
def test_plm_loss_correctness():
    model = PARSeq(plm_enabled=True)
    batch = {...}
    loss = model(**batch)
    assert not torch.isnan(loss)
    assert 0 < loss < 10
```

### Numerical Equivalence
```python
# Test Flash vs Standard Attention
def test_flash_equivalence():
    torch.manual_seed(42)
    model_flash = PARSeq(use_flash=True)
    model_std = PARSeq(use_flash=False)
    batch = {...}
    with torch.no_grad():
        out_flash = model_flash(**batch)
        out_std = model_std(**batch)
    torch.testing.assert_close(out_flash, out_std, atol=1e-3, rtol=1e-3)
```

### Device Placement
```python
# Test all tensors on correct device
def test_device_consistency():
    model = PARSeq().cuda()
    batch = {k: v.cuda() for k, v in batch.items()}

    def check_device(module, input, output):
        if isinstance(output, torch.Tensor):
            assert output.device.type == 'cuda'

    for module in model.modules():
        module.register_forward_hook(check_device)

    model(**batch)
```

---

## Success Criteria

The audit is complete when:
- ✅ All critical correctness issues identified and documented
- ✅ Device placement verified for all code paths
- ✅ Gradient flow validated through PLM and Flash Attention
- ✅ Configuration consistency checked across all variants
- ✅ Testing gaps identified with specific recommendations
- ✅ Performance optimization opportunities prioritized
- ✅ Production deployment recommendations provided

---

<!-- ## Pulse Management Commands (DEPRECATED in favor of agent skills)

```bash
# Check pulse status
uv run compass pulse-status

# Register new artifact
uv run compass pulse-sync --path "audit/01_plm_correctness.md" --type "audit"

# Update token burden as work progresses
uv run compass pulse-checkpoint --burden medium

# Export when complete
uv run compass pulse-export -->
```

---

**Pulse Status**: ✅ Ready for Phase 6.1
**Next Action**: Review audit prompt and start PLM correctness analysis
**Risk Level**: MEDIUM - Complex implementation needs thorough validation
**Estimated Duration**: 2-4 sessions
