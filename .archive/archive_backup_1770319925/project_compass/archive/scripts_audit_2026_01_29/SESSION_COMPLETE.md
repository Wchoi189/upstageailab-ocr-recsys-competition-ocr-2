# Session Complete: Import & Script Audit

**Session ID:** import-script-audit-2026-01-29
**Date:** 2026-01-29
**Status:** ✅ Complete - Ready for Next Session

---

## What Was Accomplished

### 1. Environment Issue Resolved
- ✅ Identified torch corruption causing false import errors
- ✅ Reinstalled torch 2.6.0+cu124 with CUDA support
- ✅ Reduced broken imports from 164 to 36 (78% were false positives)

### 2. Critical Discovery: Missing Core Modules
Found 3 missing core modules that are referenced but don't exist:
- 🔴 `ocr/core/lightning/base.py` (OCRPLModule) - 3 imports fail
- 🔴 `ocr/core/models/architecture.py` (OCRModel) - 2 imports fail
- 🔴 `ocr/core/models/encoder/timm_backbone.py` (TimmBackbone) - 4+ imports fail

### 3. Comprehensive Analysis Completed
- ✅ 36 real broken imports categorized by type and priority
- ✅ 13 broken hydra targets identified (onnxruntime issue)
- ✅ 128 scripts audited and categorized (55 keep, 25 refactor, 48 review)

### 4. Analysis Tools Created
Three reusable audit tools:
1. `analyze_broken_imports_adt.py` - Categorizes import errors
2. `categorize_internal_ocr_imports.py` - Separates core vs scripts
3. `audit_scripts_directory.py` - Analyzes script complexity

---

## Deliverables

### Primary Documents (Project Compass)
- **AUDIT_FINDINGS.md** - Complete consolidated report (36 imports, priorities)
- **TOOLS_INDEX.md** - Index of all tools with purposes and usage
- **SESSION_HANDOVER.md** - Full session summary and recommendations
- **walkthrough.md** - Detailed session walkthrough

### Data Files (Project Compass)
- **broken_imports_analysis.json** - Initial 164 import categorization
- **scripts_categorization.json** - 128 scripts audit results

### Reusable Tools (scripts/audit/)
- **analyze_broken_imports_adt.py**
- **categorize_internal_ocr_imports.py**
- **audit_scripts_directory.py**

---

## Critical Next Steps

### 🔴 Immediate Priority (Blocking Issues)

**1. Locate or Create Missing Core Modules**
```bash
# Check if modules exist elsewhere
find ocr -name "base.py" | grep lightning
find ocr -name "architecture.py" | grep models
find ocr -name "timm_backbone.py"

# If not found, they need to be created or imports updated
```

**Impact:** 9+ import failures, 5 hydra config failures

**2. Fix Pipeline Module Imports**
Check if these are just wrong paths from refactoring:
- `ocr.data.lightning_data.OCRDataPLModule`
- `ocr.domains.detection.module.DetectionPLModule`
- `ocr.domains.recognition.module.RecognitionPLModule`

### 🟡 Short Term (Environment)

**3. Install Missing Dependencies**
```bash
uv pip install rembg aiohttp datasets python-doctr icecream
```

**4. Fix ONNX Runtime**
```bash
uv pip uninstall onnxruntime-gpu
uv pip install onnxruntime-gpu==1.18.0  # Try specific version
```

**5. Investigate Hydra Import Failures**
Verify hydra.utils.instantiate can be imported

### 🟢 Long Term (Cleanup)

**6. Scripts Directory Pruning**
- Review 48 scripts marked for manual inspection
- Archive experimental prototypes (6 files)
- Remove obsolete migrations (3 files)
- Update valuable tools (25 files)

**7. Update Hydra Configs**
Remove references to missing modules in:
- `configs/model/architectures/dbnetpp.yaml`
- `configs/model/architectures/dbnet_atomic.yaml`
- `configs/model/architectures/parseq.yaml`
- `configs/data/transforms/background_removal.yaml` (8 configs)

---

## File Organization

### ✅ Kept in scripts/audit/
- `analyze_broken_imports_adt.py` - Reusable tool
- `categorize_internal_ocr_imports.py` - Reusable tool
- `audit_scripts_directory.py` - Reusable tool
- `broken_imports_analysis.json` - Historical data
- `scripts_categorization.json` - Action guide
- Other existing audit tools (master_audit.py, etc.)

### ✅ Moved to Project Compass
- `AUDIT_FINDINGS.md` - Consolidated report
- `TOOLS_INDEX.md` - Tools documentation
- `SESSION_HANDOVER.md` - Handover document
- JSON data files (copies)

### ✅ Removed (Duplicates/Superseded)
- `IMPORT_AUDIT_SUMMARY.md` - Merged into AUDIT_FINDINGS
- `FINAL_AUDIT_RESULTS.md` - Superseded by AUDIT_FINDINGS
- `internal_import_categorization.json` - Less useful after torch fix
- `constitution.md` - Not needed
- `specification.md` - Not needed
- `implementation_plan.md` - Outdated

---

## Quick Reference

### Current Baseline
- **36 broken imports** (down from 164 false positives)
- **13 broken hydra targets** (onnxruntime issue)
- **5-8 missing core modules** (critical)
- **8 missing dependencies** (medium priority)

### Target Baseline (After Fixes)
- **8-12 broken imports** (optional deps + UI modules)
- **0 broken hydra targets**
- **0 missing core modules**
- **0 missing required dependencies**

### Key Files
- **Main Report:** `project_compass/pulse_staging/artifacts/AUDIT_FINDINGS.md`
- **Tools Index:** `project_compass/pulse_staging/artifacts/TOOLS_INDEX.md`
- **Scripts Audit:** `project_compass/pulse_staging/artifacts/scripts_categorization.json`

---

## For Next Session

**Recommended Focus:**
1. 🔴 **Find/create missing modules** (2-4 hours or 30 mins if just moved)
2. 🔴 **Fix pipeline imports** (30 mins - likely simple path fixes)
3. 🟡 **Install dependencies** (15 mins)
4. 🟡 **Fix onnxruntime** (30 mins)

**Expected Outcome:**
- Reduce to 8-12 broken imports (acceptable baseline)
- All core OCR functionality working
- Clear path to scripts cleanup

**Tools Available:**
- Run `uv run python scripts/audit/master_audit.py` to check current status
- Use tools created to re-categorize if needed
- Reference AUDIT_FINDINGS.md for complete context

---

## Project Compass Status

```
Pulse ID: import-script-audit-2026-01-29
Artifacts Registered: 6
- walkthrough.md
- broken_imports_analysis.json
- AUDIT_FINDINGS.md
- TOOLS_INDEX.md
- scripts_categorization.json
- SESSION_HANDOVER.md

Status: Exported
Location: project_compass/history/ocr-domain-refactor/20260129_020232_import-script-audit/
```

---

**Ready for New Session** - All findings documented, tools organized, priorities clear.
