# Audit Resolution Implementation Plan

**Session:** audit-resolution-2026-01-29
**Date:** 2026-01-29
**Status:** 📋 Planning Complete - Ready for Review

---

## Executive Summary

**Root Cause Identified:** Corrupted Hydra installation causing cascade import failures

**Current State:**
- 46 broken imports (down from 164 false positives after torch fix)
- Core modules **DO EXIST** but cannot be imported due to dependency failures
- Hydra corruption affects 12+ direct imports and 20+ cascade failures

**Target State:**
- 6-10 broken imports (only optional dependencies)
- All core OCR functionality working
- Clean baseline for future audits

---

## Critical Discovery

### ✅ Core Modules Are NOT Missing

Previous audit conclusion was **INCORRECT**. Verification shows:

| Module | Status | Location |
|--------|--------|----------|
| `OCRPLModule` | ✅ EXISTS | `ocr/core/lightning/base.py:18` |
| `OCRModel` | ✅ EXISTS | `ocr/core/models/architecture.py:16` |
| `TimmBackbone` | ✅ EXISTS | `ocr/core/models/encoder/timm_backbone.py:11` |

All three "missing" modules exist and were verified via AST symbol search.

### 🔴 Actual Root Cause: Hydra Corruption

```python
>>> import hydra
>>> from hydra.utils import instantiate
ModuleNotFoundError: No module named 'hydra.core'
```

**Impact:** Hydra corruption causes cascade import failures:
1. 12 direct hydra imports fail
2. 20+ modules that import hydra fail
3. 10+ modules that import those modules fail (cascade)

This explains why "missing" modules exist but audit reports them as broken.

---

## Current Audit Results (2026-01-29)

**Total: 46 Broken Imports**

### Category Breakdown

**Comment: NO LEGACY SHIMS AND DEPRECATION WARNINGS**

#### 1. Corrupted Hydra (12 imports) 🔴 CRITICAL
- `hydra.utils.instantiate` (9 locations)
- `hydra.compose`, `hydra.initialize` (3 locations)

**Files affected:**
- `ocr/core/lightning/base.py`
- `ocr/core/models/architecture.py`
- `ocr/core/models/{encoder,decoder,head}/__init__.py`
- `ocr/data/lightning_data.py`
- `ocr/data/datasets/__init__.py`
- `ocr/core/analysis/data/calculate_normalization.py`

#### 2. Cascade Failures (24 imports) 🟡 SECONDARY
These fail ONLY because hydra imports fail:
- `ocr.core.lightning.base.OCRPLModule` (3 locations)
- `ocr.core.models.architecture.OCRModel` (2 locations)
- `ocr.core.models.{encoder,decoder,head}.get_*_by_cfg` (3 locations)
- `ocr.data.lightning_data.OCRDataPLModule` (1 location)
- `ocr.domains.*.module.*PLModule` (3 locations)
- Various model imports (PARSeq, etc.)

#### 3. Missing Optional Dependencies (10 imports) 🟢 LOW PRIORITY
- `tiktoken` (2) - Optional, only for LLM clients
- `rembg` (3) - Background removal (optional preprocessing)
- `doctr` (1) - Document OCR (optional feature)  (comment: doctr might be unused and not needed anymore)
- `aiohttp` (1) - Async HTTP (batch processing script)
- `rich.logging` (1) - Pretty logging (optional)
- `icecream` (1) - Debug logging (dev tool)

---

## Proposed Changes

### Phase 1: Environment Repair (CRITICAL 🔴)

#### Fix Corrupted Hydra Installation

**Problem:** `No module named 'hydra.core'` indicates partial/corrupted installation

**Solution:**
```bash
# Uninstall completely
uv pip uninstall hydra-core omegaconf

# Reinstall with explicit versions
uv pip install hydra-core==1.3.2 omegaconf==2.3.0

# Verify installation
uv run python -c "from hydra.utils import instantiate; print('✅ Hydra OK')"
```

**Expected outcome:**
- All 12 direct hydra imports fixed
- 24 cascade failures resolved automatically
- Total broken imports reduced from 46 → 10

---

### Phase 2: Optional Dependencies (LOW PRIORITY 🟢)

#### Install Missing Packages

These are **optional** but used by some features:

```bash
# Core optional dependencies
uv pip install rembg      # Background removal preprocessing
uv pip install aiohttp    # Async HTTP for batch processing
uv pip install python-doctr  # Alternative OCR engine

# Development/logging tools (truly optional)
uv pip install tiktoken   # Token counting for LLM clients
uv pip install rich       # Pretty terminal output
uv pip install icecream   # Debug logging helper
```

**Expected outcome:**
- Reduce to ~2-4 broken imports (truly optional/deferred)

---

## Verification Plan

### Automated Tests

#### 1. Test Hydra Import
```bash
uv run python -c "from hydra.utils import instantiate; print('✅ Hydra OK')"
```

#### 2. Test Core Module Imports
```bash
uv run python -c "
from ocr.core.lightning.base import OCRPLModule
from ocr.core.models.architecture import OCRModel
from ocr.core.models.encoder.timm_backbone import TimmBackbone
print('✅ Core modules OK')
"
```

#### 3. Re-run Master Audit
```bash
uv run python scripts/audit/master_audit.py > audit_post_fix.txt
```

**Success criteria:**
- Hydra imports: ✅ PASS
- Core module imports: ✅ PASS
- Broken imports: ≤10 (only optional deps)

### Manual Verification

#### 4. Test Training Pipeline
```bash
# Verify orchestrator can be imported
uv run python -c "from ocr.pipelines.orchestrator import OCRProjectOrchestrator; print('✅ Pipeline OK')"
```

#### 5. Test Domain Modules
```bash
# Detection module
uv run python -c "from ocr.domains.detection.module import DetectionPLModule; print('✅ Detection OK')"

# Recognition module
uv run python -c "from ocr.domains.recognition.module import RecognitionPLModule; print('✅ Recognition OK')"
```

---

## User Review Required

> [!IMPORTANT]
> **Hydra Version Selection**
>
> Plan uses `hydra-core==1.3.2` (current stable). Verify this is compatible with:
> - Project's other dependencies
> - Existing hydra configs
> - CI/CD environment
>
> Alternative: Use `hydra-core>=1.3,<2.0` for flexibility

> [!WARNING]
> **Optional Dependencies Decision Needed**
>
> Some "optional" deps may be required for specific features:
> - `rembg` - If background removal preprocessing is used
> - `doctr` - If alternative OCR detector is needed
> - `aiohttp` - If batch pseudo-labeling script is actively used
>
> Review which features are actually needed before installing.

---

## Deferred Items

### Scripts Directory Pruning (Future Pulse)

**Current state:** 128 scripts analyzed
- 55 files: KEEP
- 25 files: REFACTOR
- 48 files: REVIEW (manual inspection needed)

**Recommendation:** Create separate pulse after environment fix

**Scope:**
1. Archive experimental prototypes (`scripts/prototypes/`)
2. Remove obsolete migrations (`scripts/migration_refactoring/`)
3. Update valuable but outdated tools (`scripts/performance/`)
4. Review MCP tools status (`scripts/mcp/`)

**Estimated effort:** 4-6 hours for full review and cleanup

---

## Timeline & Effort Estimates

### Phase 1: Environment Repair
- **Fix Hydra:** 15 minutes (uninstall + reinstall + verify)
- **Re-audit:** 5 minutes
- **Test imports:** 10 minutes
- **Total:** ~30 minutes

### Phase 2: Optional Dependencies (if needed)
- **Install deps:** 10 minutes
- **Test features:** 20 minutes (if testing all)
- **Total:** ~30 minutes

### Phase 3: Documentation
- **Update findings:** 15 minutes
- **Create baseline doc:** 15 minutes
- **Session handover:** 10 minutes
- **Total:** ~40 minutes

**Grand Total:** 1-2 hours (with verification and documentation)

---

## Success Criteria

### Must Have ✅
- [ ] Hydra imports working
- [ ] Core modules importable (OCRPLModule, OCRModel, TimmBackbone)
- [ ] Training pipeline functional (orchestrator, domain modules)
- [ ] Broken imports ≤10 (only truly optional)

### Should Have 🎯
- [ ] All optional deps installed (for full feature set)
- [ ] Audit baseline documented (expected broken imports)
- [ ] Scripts pruning plan created

### Nice to Have 🌟
- [ ] Scripts directory cleaned up
- [ ] Import linter CI/CD integration plan
- [ ] Hydra config validation added to audit script

---

## Next Steps After Approval

1. **Fix Hydra** (Critical, ~15 min)
   - Uninstall hydra-core and omegaconf
   - Reinstall with pinned versions
   - Verify with import test

2. **Verify Fix** (Critical, ~15 min)
   - Re-run master audit
   - Test core module imports
   - Confirm broken imports ≤10

3. **Install Optional Deps** (Optional, ~10 min)
   - Based on user decision
   - Install needed packages only

4. **Document Baseline** (Important, ~20 min)
   - Create expected baseline doc
   - Update session handover
   - Close audit pulse

5. **Plan Scripts Cleanup** (Future)
   - Create new pulse for scripts pruning
   - Separate from environment fix

---

## Files to Modify

### No Code Changes Required

This is purely an **environment fix**:
- No Python files need modification
- No config files need updates
- No imports need changing

### Documentation Updates

After fix, update:
- `project_compass/pulse_staging/artifacts/AUDIT_FINDINGS.md` - Update with correct root cause
- `project_compass/pulse_staging/artifacts/BASELINE.md` - Create expected baseline doc
- `project_compass/history/.../SESSION_HANDOVER.md` - Final handover with resolution

---

## Risk Assessment

### Low Risk ✅
- **Hydra reinstall:** Standard package operation, easily reversible
- **Optional deps:** Don't affect core functionality
- **No code changes:** Zero risk of introducing bugs

### Medium Risk ⚠️
- **Version compatibility:** Hydra 1.3.2 may conflict with other deps (verify uv.lock)
- **Environment drift:** Ensure fix works in CI/CD, not just local

### Mitigation
- Test in isolated environment first (using uv)
- Document exact versions used
- Verify with comprehensive import tests before declaring success

---

## Appendix: Key Findings Summary

### What We Learned

1. **Audit tools have limitations** - Environment issues cause false "missing module" reports
2. **Cascade failures are real** - One corrupted dependency breaks dozens of imports
3. **AST analysis is reliable** - Symbol search confirmed modules exist despite import failures
4. **Scripts need cleanup** - 48 of 128 scripts need review (separate effort)

### Updated Understanding

Previous conclusion: "Missing core modules need to be created"
**CORRECTED:** Core modules exist, hydra corruption prevents import

This changes the intervention from "code development" to "environment repair" - much simpler and lower risk.

---

**Status:** Ready for user review and approval

**Recommended Action:** Approve Phase 1 (Hydra fix), defer Phase 2 (optional deps) decision until after verification
