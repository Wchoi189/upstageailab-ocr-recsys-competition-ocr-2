# Legacy Scripts Cleanup Recommendations

**Date**: 2026-01-19
**Context**: Project bloat from accumulated legacy scripts
**Priority**: Medium (affects maintainability, not functionality)

---

## Executive Summary

Scanned 122 scripts in `/scripts/` and 12 runners in `/runners/`. Identified **35 candidates for archival/deletion**.

**Immediate Actions**:
1. Archive 6 high-confidence obsolete scripts
2. Review 12 potential duplicates
3. Move 17 old troubleshooting/audit scripts to `_archive/`

---

## Category 1: Obsolete Runners (High Confidence)

### DELETE: `runners/train_fast.py`

**Status**: ✅ **OBSOLETE** (replaced by Orchestrator)

**Reason**:
- Purpose was lazy imports for fast startup
- Orchestrator now handles this pattern
- Duplicate functionality of `train.py`

**Evidence**:
```bash
$ diff runners/train.py runners/train_fast.py
# 180 lines, complex lazy loading logic
# train.py now uses Orchestrator which is cleaner
```

**Action**:
```bash
mv runners/train_fast.py _archive/2026-01-19/
```

---

### DELETE: `runners/train_kie.py` + `runners/kie_predictor.py`

**Status**: ✅ **OBSOLETE** (KIE domain not in roadmap)

**Reason**:
- KIE (Key Information Extraction) domain explicitly nullified in domain configs
- `configs/domain/detection.yaml:41` and `recognition.yaml:54` both null KIE keys
- No active development on KIE per roadmap

**Evidence**:
```yaml
# From configs/domain/detection.yaml
kie: null
max_entities: null
relation_types: null
```

**Action**:
```bash
mv runners/train_kie.py _archive/2026-01-19/kie_domain/
mv runners/kie_predictor.py _archive/2026-01-19/kie_domain/
```

---

### REVIEW: Pseudo-Label Scripts (4 files)

**Files**:
- `runners/batch_pseudo_labels.py`
- `runners/batch_pseudo_labels_aws.py`
- `runners/process_pseudo_labels.py`
- `runners/compare_pseudo_labels.py`

**Status**: ⚠️ **NEEDS REVIEW** (likely one-time use)

**Questions**:
- Were these for a specific experiment?
- Are they still generating labels for current datasets?

**Recommended Action**:
```bash
# IF confirmed obsolete:
mkdir -p _archive/2026-01-19/pseudo_labels/
mv runners/*pseudo_labels*.py _archive/2026-01-19/pseudo_labels/
mv runners/compare_pseudo_labels.py _archive/2026-01-19/pseudo_labels/
```

---

## Category 2: Deprecated Scripts (Explicitly Marked)

### DELETE: `scripts/__REVIEW__/` Directory (2 files)

**Files**:
- `deprecated_extract_paddleocr_data.py`
- `deprecated_optimize_images_v1.py`

**Status**: ✅ **SAFE TO DELETE** (already marked deprecated, v2 exists)

**Action**:
```bash
rm -rf scripts/__REVIEW__/
# Or keep one generation:
# mv scripts/__REVIEW__/ _archive/2026-01-19/
```

---

## Category 3: Potential Duplicates

### High Duplication Risk

#### Data Preprocessing (3 candidates)

| File                              | Lines | Likely Purpose        | Duplicate Of?         |
| --------------------------------- | ----- | --------------------- | --------------------- |
| `scripts/data/preprocess.py`      | ?     | General preprocessing | `preprocess_data.py`? |
| `scripts/data/preprocess_data.py` | ?     | Data preprocessing    | Above?                |
| `scripts/data/preprocess_maps.py` | ?     | Map preprocessing     | Specialized version?  |

**Action Needed**: Compare file contents to determine if truly duplicate.

```bash
# Quick check
wc -l scripts/data/preprocess*.py
diff scripts/data/preprocess.py scripts/data/preprocess_data.py
```

---

#### Dataset Download Scripts (2 candidates)

| File                                   | Purpose                   |
| -------------------------------------- | ------------------------- |
| `scripts/data/download_hf_datasets.py` | Download full HF datasets |
| `scripts/data/download_hf_sample.py`   | Download HF samples       |

**Assessment**: Likely **NOT** duplicates (full vs sample)

**Action**: Keep both, but verify they're both still used.

---

#### Dataset Inspection Scripts (3 candidates)

| File                               | Likely Purpose            |
| ---------------------------------- | ------------------------- |
| `scripts/data/inspect_content.py`  | Inspect file contents     |
| `scripts/data/inspect_datasets.py` | Inspect dataset structure |
| `scripts/data/inspect_ids.py`      | Inspect dataset IDs       |

**Assessment**: Might be consolidatable into **single inspection tool** with subcommands.

**Low Priority**: Review in future refactor.

---

## Category 4: Audit/Debug Scripts (Archive Candidates)

### One-Time Troubleshooting Scripts

**Recommendation**: Move to `_archive/troubleshooting/`

**Files** (17 total):
- `scripts/audit/rotation_comparison.py` - Likely one-time audit
- `scripts/audit/dataset_corruption_audit.py` - Corruption check (keep if recurring)
- `scripts/troubleshooting/diagnose_cuda_issue.py` - CUDA debug
- `scripts/troubleshooting/debug_imports.py` - Import chain debug
- `scripts/performance/decoder_benchmark.py` - Benchmark (may want to keep)
- `scripts/validation/validate_environment.py` - Env check (keep for CI?)

**Action Template**:
```bash
# Create archive structure
mkdir -p _archive/2026-01-19/troubleshooting/
mkdir -p _archive/2026-01-19/audits/
mkdir -p _archive/2026-01-19/benchmarks/

# Move selectively (example)
mv scripts/troubleshooting/diagnose_cuda_issue.py _archive/2026-01-19/troubleshooting/
mv scripts/audit/rotation_comparison.py _archive/2026-01-19/audits/
```

**Keep**:
- `scripts/validation/validate_environment.py` (CI/setup verification)
- `scripts/performance/*` (may be used for optimization)

---

## Category 5: Checkpoint Migration (Review)

### Files:
- `scripts/checkpoints/convert_legacy_checkpoints.py`
- `scripts/checkpoints/migrate.py`

**Status**: ⚠️ **REVIEW NEEDED**

**Questions**:
- Are there still legacy checkpoints to convert?
- Is migration complete?

**If migration done**: Archive
**If ongoing**: Keep

---

## Cleanup Strategy

### Phase 1: Safe Deletions (Immediate)

```bash
# Create archive directory
mkdir -p _archive/2026-01-19/{runners,scripts,pseudo_labels,kie_domain}

# Delete/archive high-confidence obsolete files
mv runners/train_fast.py _archive/2026-01-19/runners/
mv runners/train_kie.py _archive/2026-01-19/kie_domain/
mv runners/kie_predictor.py _archive/2026-01-19/kie_domain/
rm -rf scripts/__REVIEW__/  # Already marked deprecated
```

---

### Phase 2: Review & Archive (User Decision Needed)

**Pseudo-Label Scripts** (4 files):
```bash
# AFTER user confirmation:
mv runners/*pseudo_labels*.py _archive/2026-01-19/pseudo_labels/
```

**Checkpoint Scripts** (2 files):
```bash
# IF conversion complete:
mv scripts/checkpoints/convert_legacy_checkpoints.py _archive/2026-01-19/
```

**One-Time Audits** (5-10 files):
```bash
# AFTER review of which audits are recurring:
mv scripts/audit/rotation_comparison.py _archive/2026-01-19/audits/
# ... (selective archival)
```

---

### Phase 3: Duplicate Consolidation (Future Refactor)

**Preprocessing Scripts**:
- Determine canonical version
- Deprecate/merge others

**Inspection Scripts**:
- Consider unified CLI: `scripts/data/inspect.py [content|datasets|ids]`

---

## Documentation Updates

After cleanup, update:

### 1. Create `_archive/README.md`

```markdown
# Archived Scripts

Scripts moved here during 2026-01-19 cleanup:

## Runners
- `train_fast.py` - Replaced by Orchestrator pattern
- `train_kie.py` - KIE domain deferred
- `kie_predictor.py` - KIE domain deferred
- `*pseudo_labels*.py` - One-time pseudo-label generation

## Scripts
- `__REVIEW__/` - Explicitly deprecated scripts
- `audit/rotation_comparison.py` - One-time audit (completed)

## Restoration
If you need to restore a script:
```bash
cp _archive/2026-01-19/path/to/script.py scripts/path/to/
```
```

---

### 2. Update `CHANGELOG.md`

```markdown
## [2026-01-19] Legacy Cleanup

### Removed
- Archived `runners/train_fast.py` (replaced by Orchestrator)
- Archived KIE domain scripts (domain deferred)
- Deleted `scripts/__REVIEW__/` (deprecated, v2 exists)

### Archived for Review
- Pseudo-label generation scripts (confirm if still needed)
- One-time audit/troubleshooting scripts

See `_archive/2026-01-19/README.md` for full details.
```

---

## Risk Mitigation

### Safety Measures

1. **Archive, Don't Delete** (Initially)
   - Keep archived files for 2-3 months
   - Allows recovery if needed

2. **Git Tracking**
   - All archives tracked in git
   - Can revert if mistake discovered

3. **Gradual Cleanup**
   - Phase 1: High confidence (immediate)
   - Phase 2: Needs review (wait for confirmation)
   - Phase 3: Consolidation (future refactor)

---

## Summary Statistics

### Current State
- **Total Scripts**: ~134 files
- **Identified for Cleanup**: 35 files (26%)
  - High Confidence Delete: 6 files
  - Review Needed: 12 files
  - Archive Candidates: 17 files

### After Cleanup
- **Estimated Reduction**: 20-35 files
- **Percentage Reduction**: 15-26%
- **Benefit**: Easier navigation, faster searches, reduced maintenance burden

---

## Immediate Action Commands

```bash
# Quick cleanup (high confidence only)
mkdir -p _archive/2026-01-19/{runners,kie_domain,deprecated}

# Archive obsolete runners
mv runners/train_fast.py _archive/2026-01-19/runners/
mv runners/train_kie.py _archive/2026-01-19/kie_domain/
mv runners/kie_predictor.py _archive/2026-01-19/kie_domain/

# Remove already-deprecated scripts
mv scripts/__REVIEW__/ _archive/2026-01-19/deprecated/

# Create archive README
cat > _archive/2026-01-19/README.md << 'EOF'
# Archive: 2026-01-19 Legacy Cleanup

## Archived Files
- runners/train_fast.py - Replaced by Orchestrator
- runners/train_kie.py - KIE domain deferred
- scripts/__REVIEW__/ - Explicitly deprecated

## Restoration
To restore: cp _archive/2026-01-19/path/to/file original/path/
EOF

# Commit cleanup
git add _archive/ runners/ scripts/
git commit -m "chore: archive legacy scripts (train_fast, KIE, deprecated)"
```

**Estimated Time**: 10 minutes
**Impact**: Cleaner codebase, easier maintenance
