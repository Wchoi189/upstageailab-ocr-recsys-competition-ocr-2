# Audit Tools Index

**Session:** import-script-audit-2026-01-29
**Created:** 2026-01-29
**Purpose:** Comprehensive index of all analysis tools and artifacts created during import audit

---

## Analysis Tools (Python Scripts)

### 1. analyze_broken_imports_adt.py
**Path:** `scripts/audit/analyze_broken_imports_adt.py`
**Purpose:** Parse master_audit.py output and categorize broken imports by type
**Input:** Runs master_audit.py internally
**Output:** `broken_imports_analysis.json`
**Categories:** 7 types (torch, lightning, hydra, internal_ocr, tiktoken, ui, other)
**Status:** ✅ Complete and functional
**Usage:**
```bash
uv run python scripts/audit/analyze_broken_imports_adt.py
```

**Key Features:**
- Automatic categorization of import errors
- Identifies environment vs code issues
- Structured JSON output for analysis
- Handles 164 initial broken imports

---

### 2. categorize_internal_ocr_imports.py
**Path:** `scripts/audit/categorize_internal_ocr_imports.py`
**Purpose:** Separate core ocr/ package imports from scripts/ directory candidates
**Input:** `broken_imports_analysis.json`
**Output:** `internal_import_categorization.json`
**Status:** ✅ Complete and functional
**Usage:**
```bash
uv run python scripts/audit/categorize_internal_ocr_imports.py
```

**Key Features:**
- Separates core (73) from scripts (8) imports
- Prioritizes by module type (validation, interfaces, lightning, etc.)
- Identifies which imports are critical vs deferrable
- Provides fix plan ordering

**Output Categories:**
- Core fixes: validation_module, base_interfaces, lightning_utils, registry, model_components, metrics_evaluation
- Script candidates: By subdirectory (troubleshooting, data, performance)

---

### 3. audit_scripts_directory.py
**Path:** `scripts/audit/audit_scripts_directory.py`
**Purpose:** Full scripts/ directory audit with complexity analysis and categorization
**Input:** Scans entire `scripts/` directory
**Output:** `scripts_categorization.json`
**Status:** ✅ Complete and functional
**Usage:**
```bash
uv run python scripts/audit/audit_scripts_directory.py
```

**Key Features:**
- Analyzes 128 Python scripts
- AST-based complexity metrics (functions, classes, imports, lines)
- Categorizes as: KEEP, REFACTOR, REVIEW, ARCHIVE, REMOVE
- Groups by subdirectory for organized review

**Categorization Logic:**
- `audit/` → KEEP (audit tools are critical)
- `data/` → KEEP if has __main__, else REVIEW
- `performance/` → REFACTOR (valuable but outdated)
- `troubleshooting/` → REVIEW (may be obsolete)
- `prototypes/` → REVIEW (experimental)
- `utils/` → KEEP (utilities valuable)

---

## Data Outputs (JSON Files)

### 4. broken_imports_analysis.json
**Path:** `scripts/audit/broken_imports_analysis.json`
**Purpose:** Structured breakdown of initial 164 broken imports
**Size:** ~50KB
**Structure:**
```json
{
  "total": 164,
  "categories": {
    "internal_ocr": 81,
    "torch_missing": 21,
    "lightning_missing": 12,
    "hydra_missing": 12,
    "other": 34,
    "tiktoken_optional": 2,
    "ui_modules": 2
  },
  "details": {
    "category_name": [
      {
        "file": "path/to/file.py",
        "line": 123,
        "module": "module.name",
        "symbols": ["Symbol1", "Symbol2"],
        "error": "error message"
      }
    ]
  }
}
```

**Use Cases:**
- Identify patterns in import errors
- Batch fix similar issues
- Track resolution progress

---

### 5. internal_import_categorization.json
**Path:** `scripts/audit/internal_import_categorization.json`
**Purpose:** Core vs scripts separation with fix priorities
**Size:** ~30KB
**Structure:**
```json
{
  "core_fixes": {
    "validation_module": [...],
    "base_interfaces": [...],
    "lightning_utils": [...],
    "registry": [...],
    "model_components": [...],
    "metrics_evaluation": [...],
    "other_core": [...]
  },
  "script_candidates": {
    "troubleshooting": [...],
    "data": [...],
    "performance": [...]
  },
  "summary": {
    "total_internal": 81,
    "core_to_fix": 73,
    "scripts_deferred": 8
  }
}
```

**Use Cases:**
- Prioritize core package fixes
- Defer non-critical script imports
- Strategic fix planning

---

### 6. scripts_categorization.json
**Path:** `scripts/audit/scripts_categorization.json`
**Purpose:** Complete scripts directory audit with categorization
**Size:** ~100KB
**Structure:**
```json
{
  "audit_date": "2026-01-29T...",
  "total_scripts": 128,
  "categorization": {
    "keep": [
      {
        "file": "scripts/audit/analyze_broken_imports_adt.py",
        "complexity": {
          "functions": 10,
          "classes": 2,
          "imports": 15,
          "has_main": true,
          "lines": 224
        },
        "size_bytes": 7890
      }
    ],
    "refactor": [...],
    "review": [...],
    "archive": [...],
    "remove": [...]
  }
}
```

**Use Cases:**
- Guide scripts pruning effort
- Identify candidates for archival
- Prioritize refactoring work

---

## Documentation Artifacts (Markdown)

### 7. AUDIT_FINDINGS.md
**Path:** `project_compass/pulse_staging/artifacts/AUDIT_FINDINGS.md`
**Purpose:** Consolidated final audit report with all findings
**Sections:**
- Executive Summary
- Critical Findings (missing core modules)
- 36 Real Broken Imports Breakdown
- 13 Broken Hydra Targets
- Scripts Audit (128 files)
- Priority Action Items
- Before/After Comparison

**Status:** ✅ Primary reference document
**Audience:** Developers, next session planning

---

### 8. SESSION_HANDOVER.md
**Path:** `project_compass/history/.../SESSION_HANDOVER.md`
**Purpose:** Comprehensive session summary for pulse export
**Sections:**
- Critical Discovery (false positives)
- Verified Status (core, scripts, categorization)
- Artifacts Created
- Recommendations (immediate, short, medium, long term)
- Deferred Items
- Next Steps

**Status:** ✅ Complete handover
**Audience:** Future sessions, project tracking

---

### 9. IMPORT_AUDIT_SUMMARY.md
**Path:** `scripts/audit/IMPORT_AUDIT_SUMMARY.md`
**Purpose:** Root cause analysis of environment issue
**Focus:**
- Torch corruption explanation
- Environment vs real issues breakdown
- Verification of core modules existence
- False alarm categories table

**Status:** ✅ Historical reference
**Audience:** Understanding the investigation process

---

### 10. FINAL_AUDIT_RESULTS.md
**Path:** `scripts/audit/FINAL_AUDIT_RESULTS.md`
**Purpose:** Post-torch-fix audit results (36 real imports)
**Focus:**
- Detailed breakdown of 36 imports
- Missing dependencies list
- Hydra target failures
- Priority action items

**Status:** ⚠️ Superseded by AUDIT_FINDINGS.md (can be removed)

---

## Compass Artifacts

### 11. constitution.md
**Path:** `project_compass/pulse_staging/artifacts/constitution.md`
**Purpose:** Project principles and success criteria
**Status:** ⏸️ Not synced (will be part of pulse export)

### 12. specification.md
**Path:** `project_compass/pulse_staging/artifacts/specification.md`
**Purpose:** Requirements specification
**Status:** ⏸️ Not synced

### 13. implementation_plan.md
**Path:** `project_compass/pulse_staging/artifacts/implementation_plan.md`
**Purpose:** Initial implementation strategy
**Status:** ⏸️ Outdated (findings changed approach)

---

## Recommended Consolidation

### Files to Keep
1. **AUDIT_FINDINGS.md** ← Primary reference (KEEP)
2. **analyze_broken_imports_adt.py** ← Reusable tool (KEEP)
3. **categorize_internal_ocr_imports.py** ← Reusable tool (KEEP)
4. **audit_scripts_directory.py** ← Reusable tool (KEEP)
5. **broken_imports_analysis.json** ← Historical data (KEEP)
6. **scripts_categorization.json** ← Action guide (KEEP)
7. **SESSION_HANDOVER.md** ← Pulse export (KEEP)

### Files to Consolidate/Remove
- **IMPORT_AUDIT_SUMMARY.md** → Merged into AUDIT_FINDINGS (REMOVE)
- **FINAL_AUDIT_RESULTS.md** → Superseded by AUDIT_FINDINGS (REMOVE)
- **internal_import_categorization.json** → Less useful after torch fix (OPTIONAL KEEP)
- **constitution.md** → Sync or remove (RESOLVE)
- **specification.md** → Sync or remove (RESOLVE)
- **implementation_plan.md** → Outdated (REMOVE or UPDATE)

---

## Usage Guide

### Running a Fresh Audit
```bash
# 1. Ensure environment is healthy
uv run python -c "import torch; print(f'Torch {torch.__version__}')"

# 2. Run master audit
uv run python scripts/audit/master_audit.py > audit_output.txt

# 3. Analyze imports (if needed)
uv run python scripts/audit/analyze_broken_imports_adt.py

# 4. Categorize scripts (if needed)
uv run python scripts/audit/audit_scripts_directory.py
```

### Tracking Progress
```bash
# Compare current vs baseline
uv run python scripts/audit/master_audit.py | grep "BROKEN IMPORTS" | wc -l
# Target: 36 or fewer

# Check specific categories
cat scripts/audit/broken_imports_analysis.json | jq '.categories'
```

---

## Maintenance Notes

**When to Rerun:**
- After major refactoring
- Before/after dependency updates
- When adding new modules
- Quarterly health checks

**Expected Baseline:**
- 8-12 broken imports (optional deps + deferred)
- 0 broken core imports
- 0-5 broken hydra targets

**Current Status:**
- 36 broken imports (includes missing core modules)
- 13 broken hydra targets (onnxruntime issue)

---

**End of Index** - Created 2026-01-29 for import-script-audit session
