# Legacy Cleanup Summary - 2026-01-19

**Status**: ✅ Complete (Categories 1-2, 4)
**Deferred**: Category 3 (duplicates), Category 5 (checkpoints - needs review)

---

## Completed Actions

### ✅ Category 1: Obsolete Runners (3 files)
- `train_fast.py` → `_archive/2026-01-19/runners/`
- `train_kie.py` → `_archive/2026-01-19/kie_domain/`
- `kie_predictor.py` → `_archive/2026-01-19/kie_domain/`

### ✅ Category 2: Deprecated Scripts (4 files)
- `scripts/__REVIEW__/` → `_archive/2026-01-19/deprecated/__REVIEW__/`
  - deprecated_extract_paddleocr_data.py
  - deprecated_optimize_images_v1.py
  - duplicate_ci_update_diagrams_no_uv.sh
  - duplicate_manage_diagrams.sh

### ✅ Category 4: Audit & Troubleshooting (3 files)
- `rotation_comparison.py` → `_archive/2026-01-19/audits/`
- `diagnose_cuda_issue.py` → `_archive/2026-01-19/troubleshooting/`
- `debug_imports.py` → `_archive/2026-01-19/troubleshooting/`

**Total Archived**: 10 files

---

## Deferred for Future Review

### ⏸️ Category 3: Potential Duplicates
User will review at later time:
- `scripts/data/preprocess*.py` (3 potential duplicates)
- `scripts/data/inspect_*.py` (3 inspection scripts)
- `runners/*pseudo_labels*.py` (4 pseudo-label scripts)

### ⏸️ Category 5: Checkpoint Migration
**Recommendation**: Keep for now (still potentially useful)

**Files**:
- `scripts/checkpoints/convert_legacy_checkpoints.py` - Converts old checkpoint format
- `scripts/checkpoints/generate_metadata.py` - Generates checkpoint metadata
- `scripts/checkpoints/migrate.py` - Migrates checkpoints

**Reason to keep**: These may still be needed for converting older checkpoints if resuming from previous training runs.

**Future review**: Check if all legacy checkpoints have been converted. If done, archive these scripts.

---

## Archive Structure Created

```
_archive/2026-01-19/
├── README.md                    # This file
├── runners/
│   └── train_fast.py           (5.9 KB)
├── kie_domain/
│   ├── train_kie.py            (9.7 KB)
│   └── kie_predictor.py        (8.3 KB)
├── deprecated/
│   └── __REVIEW__/
│       ├── deprecated_extract_paddleocr_data.py
│       ├── deprecated_optimize_images_v1.py
│       ├── duplicate_ci_update_diagrams_no_uv.sh
│       └── duplicate_manage_diagrams.sh
├── audits/
│   └── rotation_comparison.py  (9.4 KB)
└── troubleshooting/
    ├── diagnose_cuda_issue.py  (14.8 KB)
    └── debug_imports.py        (1.9 KB)
```

---

## Next Git Commit

```bash
git status  # Review changes
git add _archive/ runners/ scripts/
git commit -m "chore: archive legacy scripts (Phase 4 cleanup)

Archived 10 obsolete files:
- Category 1: train_fast.py, KIE scripts (2 files)
- Category 2: __REVIEW__ deprecated scripts (4 files)
- Category 4: One-time audit/troubleshooting (3 files)

Deferred:
- Category 3: Potential duplicates (for future review)
- Category 5: Checkpoint scripts (kept - may still be useful)

See _archive/2026-01-19/README.md for details
"
```

---

## Cleanup Impact

### Before
- `runners/`: 12 files
- `scripts/__REVIEW__/`: 4 files
- `scripts/audit/`: Multiple files including one-time audits
- `scripts/troubleshooting/`: 11 files

### After
- `runners/`: 9 files (**-3**)
- `scripts/__REVIEW__/`: Removed (**-4**)
- `scripts/audit/`: Cleaner (**-1**)
- `scripts/troubleshooting/`: Cleaner (**-2**)

**Total Reduction**: 10 files archived (7.5% of scripts)

---

## Restoration

If you accidentally archived something useful:

```bash
# List archived files
ls -R _archive/2026-01-19/

# Restore specific file
cp _archive/2026-01-19/PATH/TO/FILE original/location/

# Restore entire directory
cp -r _archive/2026-01-19/kie_domain/* runners/
```

---

## Related Work

This cleanup was part of Phase 4 post-Orchestrator implementation:

1. ✅ **Orchestrator Implementation** - `ocr/pipelines/orchestrator.py`
2. ✅ **Performance Audit** - Identified 60+ second startup bottlenecks
3. ✅ **Legacy Cleanup** - This cleanup (10 files)

**See Also**:
- `Performance_Audit_Supplement_Orchestrator_Findings.md`
- `Legacy_Scripts_Cleanup_Recommendations.md`
- `.gemini/antigravity/brain/.../walkthrough.md`
