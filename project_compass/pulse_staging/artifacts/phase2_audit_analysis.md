# Phase 2 Audit Resolution Analysis

**Date:** 2026-01-29 04:05
**Session:** Continuing from audit-resolution-2026-01-29
**Context:** Phase 1 Complete (Hydra fixed, 46→16 broken imports)

---

## Executive Summary

### Phase 1 Results ✅
- **Achieved:** Fixed corrupted Hydra installation
- **Impact:** Reduced broken imports from 46 to 16 (65% reduction)
- **Status:** All core OCR functionality restored

### Phase 2 Scope
- **Remaining:** 16 broken imports + 8 Hydra config issues
- **Classification:**
  - 11 imports: Dependency conflicts (fixable)
  - 4 imports: Expected/deferred (not blocking)
  - 8 configs: Related to onnxruntime issue

---

## Current Audit State (2026-01-29 04:04)

### Broken Imports Breakdown

| Category | Count | Impact | Fixable |
|----------|-------|--------|---------|
| Pygments issues (rich/icecream) | 2 | Dev tools | ✅ Reinstall |
| Multidict issues (aiohttp/datasets) | 5 | Optional features | ✅ Reinstall |
| Rembg (onnxruntime) | 4 | Background removal | ⚠️ Version issue |
| Doctr | 1 | Alternative OCR | ✅ Reinstall |
| UI modules | 2 | Separate package | ❌ Expected |
| AgentQMS | 2 | Separate package | ❌ Expected |
| **Total** | **16** | - | **11 fixable** |

### Hydra Config Issues
- **Count:** 8 broken targets
- **Root Cause:** Same as rembg import issue (onnxruntime)
- **Impact:** Background removal transform configs
- **Criticality:** Low (optional preprocessing)

---

## Detailed Analysis

### Category 1: Pygments Corruption (2 imports)

**Files Affected:**
```
ocr/core/utils/logging.py:23
  → Import: rich.logging ['RichHandler']
ocr/core/utils/logging.py:35
  → Import: icecream ['ic', 'install']
```

**Root Cause:**
- Transitive dependency `pygments.lexers` missing `guess_lexer_for_filename`
- Both `rich` and `icecream` depend on pygments for syntax highlighting

**Impact:**
- Pretty terminal logging (rich)
- Debug logging helper (icecream)
- **Non-critical:** Development convenience only

**Fix:**
```bash
uv pip install --reinstall pygments
```

**Verification:**
```bash
uv run python -c "from rich.logging import RichHandler; from icecream import ic"
```

**Estimated Effort:** 2 minutes
**Risk:** Low (dev dependency)

---

### Category 2: Multidict Corruption (5 imports)

**Files Affected:**
```
runners/batch_pseudo_labels.py:32
  → Import: aiohttp
scripts/finetune_ppocr.py:41
  → Import: datasets ['load_dataset']
scripts/data/download_hf_datasets.py:8
  → Import: datasets ['load_dataset']
scripts/data/generate_pseudo_labels.py:16
  → Import: aiohttp
scripts/data/process_kie_to_rec.py:11
  → Import: datasets ['load_dataset']
```

**Root Cause:**
- Transitive dependency `multidict` missing `istr` attribute
- Both `aiohttp` and `datasets` (HuggingFace) depend on multidict

**Impact:**
- Batch processing scripts (aiohttp)
- HuggingFace datasets loading
- **Medium priority:** Used in data preprocessing scripts

**Fix:**
```bash
uv pip install --reinstall multidict
```

**Verification:**
```bash
uv run python -c "import aiohttp; from datasets import load_dataset"
```

**Estimated Effort:** 2 minutes
**Risk:** Low (reinstall should fix)

---

### Category 3: Rembg/OnnxRuntime Issues (4 imports + 8 configs)

**Import Files Affected:**
```
ocr/domains/detection/data/preprocessing/background_removal.py:13
  → Import: rembg ['remove']
ocr/domains/detection/inference/preprocessing_pipeline.py:159
  → Import: rembg ['remove']
ocr/domains/detection/inference/perspective/core.py:15
  → Import: rembg ['remove']
scripts/data/generate_pseudo_labels.py:21
  → Import: rembg ['remove']
```

**Hydra Config Files Affected:**
```
configs/data/transforms/background_removal.yaml (4 targets)
configs/data/transforms/with_background_removal.yaml (4 targets)
```

**Root Cause:**
- `onnxruntime-gpu` module missing `SessionOptions` attribute
- Error: `module 'onnxruntime' has no attribute 'SessionOptions'`
- `rembg` depends on onnxruntime for ML inference

**Impact:**
- Background removal preprocessing feature
- Optional feature (most OCR workflows don't use this)
- **Low priority:** Nice-to-have preprocessing

**Fix Strategy:**

**Option A: Try older stable version**
```bash
uv pip uninstall onnxruntime-gpu
uv pip install onnxruntime-gpu==1.18.0
```

**Option B: Switch to CPU version (if GPU not needed)**
```bash
uv pip uninstall onnxruntime-gpu
uv pip install onnxruntime
```

**Option C: Defer (accept as optional)**
- Keep background removal as a known optional feature
- Document in baseline

**Recommended:** Try Option A first. If fails, use Option C (defer).

**Verification:**
```bash
uv run python -c "import onnxruntime; print(onnxruntime.SessionOptions)"
uv run python -c "from rembg import remove"
```

**Estimated Effort:** 5-10 minutes (testing versions)
**Risk:** Medium (may require investigation)

---

### Category 4: Doctr/Anyascii (1 import)

**Files Affected:**
```
ocr/domains/detection/data/preprocessing/detector.py:109
  → Import: doctr.models ['zoo']
```

**Root Cause:**
- Transitive dependency `anyascii` corruption
- `doctr` uses anyascii for text normalization

**Impact:**
- Alternative OCR detector (doctr is competitor to our OCR)
- **Very low priority:** Experimental/comparison feature

**Fix:**
```bash
uv pip install --reinstall anyascii
```

**Verification:**
```bash
uv run python -c "from doctr.models import zoo"
```

**Estimated Effort:** 2 minutes
**Risk:** Low

---

### Category 5: Expected/Deferred (4 imports)

**UI Modules (2 imports):**
```
scripts/checkpoints/convert_legacy_checkpoints.py:95
  → Import: ui.apps.inference.services.checkpoint.types [...]
scripts/validation/checkpoints/validate_coordinate_consistency.py:143
  → Import: ui.utils.inference.engine ['run_inference_on_image']
```

**AgentQMS Modules (2 imports):**
```
scripts/mcp/unified_server.py:354
  → Import: AgentQMS.tools.core.context_bundle ['auto_suggest_context']
scripts/mcp/unified_server.py:201
  → Import: AgentQMS.tools.core.context_bundle [...]
```

**Status:** Expected - Separate packages
**Action:** No fix needed, document as expected baseline
**Rationale:**
- UI modules are in separate package (previously deferred)
- AgentQMS is separate development tool (not OCR core)

---

## Recommended Fix Plan

### Priority 1: Quick Wins (4 imports, ~10 minutes)

**Goal:** Fix obvious dependency corruptions

1. ✅ **Reinstall pygments** (fixes rich + icecream)
2. ✅ **Reinstall multidict** (fixes aiohttp + datasets)
3. ✅ **Reinstall anyascii** (fixes doctr)

**Command sequence:**
```bash
uv pip install --reinstall pygments multidict anyascii
```

**Expected Result:** 16 → 8 broken imports

---

### Priority 2: Investigate Onnxruntime (4 imports + 8 configs)

**Goal:** Attempt to fix rembg/background removal

**Approach:**
1. Try downgrade to onnxruntime-gpu==1.18.0
2. If fails, check compatibility matrix
3. If still fails, accept as deferred (optional feature)

**Time Budget:** 10-15 minutes
**Fallback:** Accept as optional baseline

---

### Priority 3: Document Baseline (4 imports)

**Goal:** Establish expected baseline for future audits

**Action:**
- Document UI modules (2) as expected
- Document AgentQMS (2) as expected
- Update baseline documentation

---

## Success Criteria

### Must Have ✅
- [ ] Fix all dependency corruption issues (pygments, multidict, anyascii)
- [ ] Reduce broken imports to ≤8 (only onnxruntime + expected)
- [ ] Document final baseline

### Should Have 🎯
- [ ] Attempt onnxruntime fix
- [ ] Test background removal configs if fixed
- [ ] Update pyproject.toml if versions need pinning

### Nice to Have 🌟
- [ ] Zero broken imports (ambitious, depends on onnxruntime)
- [ ] All Hydra configs working

---

## Effort Estimate

### Quick Dependency Fixes
- Reinstall 3 packages: 5 minutes
- Verify fixes: 5 minutes
- Re-run audit: 2 minutes
- **Subtotal:** 12 minutes

### Onnxruntime Investigation
- Try version downgrade: 5 minutes
- Test and verify: 5 minutes
- Document if defer: 3 minutes
- **Subtotal:** 13 minutes

### Documentation
- Update baseline doc: 10 minutes
- Update session handover: 5 minutes
- **Subtotal:** 15 minutes

### Total Estimated Time: 40 minutes

---

## Risk Assessment

### Low Risk ✅
- **Dependency reinstalls:** Standard operations, easily reversible
- **No code changes:** Zero risk of introducing bugs
- **Documented rollback:** Can revert with `uv sync`

### Medium Risk ⚠️
- **Onnxruntime version change:** May affect GPU inference
- **Mitigation:** Test thoroughly before accepting

### High Risk ❌
- None identified

---

## Verification Plan

### Post-Fix Audit
```bash
# Run full audit
uv run python scripts/audit/master_audit.py > phase2_audit_results.txt

# Verify specific imports
uv run python -c "
from rich.logging import RichHandler
from icecream import ic
import aiohttp
from datasets import load_dataset
from doctr.models import zoo
print('✅ All dependency fixes verified')
"
```

### Onnxruntime Specific
```bash
# If attempting fix
uv run python -c "
import onnxruntime
print(f'Version: {onnxruntime.__version__}')
print(f'SessionOptions: {onnxruntime.SessionOptions}')
from rembg import remove
print('✅ Onnxruntime working')
"
```

### Expected Final State

**Best Case (if onnxruntime fixes):**
- 4 broken imports (UI + AgentQMS only)
- 0 Hydra config issues
- All optional features working

**Realistic Case (if onnxruntime deferred):**
- 8 broken imports (4 onnxruntime + 4 expected)
- 8 Hydra config issues (background removal)
- Core OCR + most features working

**Acceptable Baseline:**
Either case is acceptable since:
- Core OCR functionality: ✅ Working
- Background removal: Optional preprocessing
- UI/AgentQMS: Separate packages

---

## Next Steps

1. **Review this analysis** with user (if needed)
2. **Execute Priority 1 fixes** (quick wins)
3. **Attempt Priority 2 fix** (onnxruntime)
4. **Document Priority 3** (baseline)
5. **Update session handover**

---

## Deferred Items (Future Pulses)

### Scripts Directory Cleanup
- **Current:** 128 scripts analyzed, 48 need review
- **Action:** Separate pulse after environment stable
- **Effort:** 4-6 hours

### Import Linter Enhancements
- Pre-audit dependency health check
- Better error categorization
- Dependency graph visualization

---

## File Organization

### New Artifacts (This Session)
```
project_compass/pulse_staging/artifacts/
└── phase2_audit_analysis.md (THIS FILE)
```

### To Update
```
project_compass/pulse_staging/artifacts/
├── FINAL_SESSION_HANDOVER.md (update with Phase 2 results)
└── VERIFICATION_REPORT.md (append Phase 2 verification)
```

---

**Status:** Ready for execution
**Estimated Total Time:** 40 minutes
**Confidence:** 🟢 High (simple dependency fixes)
