# Merge Plan: Main + 12_refactor/streamlit

**Date**: 2025-11-12
**Status**: In Progress
**Complexity**: High (805 files changed, 66k+ insertions, 153k+ deletions)
**Estimated Sessions**: 3-4 sessions

---

## Progress Tracker

- **STATUS**: Completed
- **CURRENT SESSION**: Session 3 - Verification & Cleanup (Completed)
- **CURRENT STEP**: All phases complete
- **LAST COMPLETED TASK**: Updated CHANGELOG, documented manual merges, updated CI/CD workflows
- **NEXT TASK**: Ready for final review and merge to main

### Session Breakdown

#### Session 1: Preparation & Critical Fixes (2-3 hours)
- [x] **Phase 1: Preparation**
  - [x] Create backup branch
  - [x] Create working branch
  - [x] Check pyproject.toml dependencies
  - [x] Document current state
- [x] **Phase 2: Port Critical Fixes**
  - [x] Port `ocr/models/head/db_head.py` fix from main (already had fix)
  - [x] Port AgentQMS system from main
  - [x] Port scripts refactor from main
  - [x] Test after each port

#### Session 2: Test & Documentation Merge (2-3 hours)
- [x] **Phase 3: Merge Test Directories**
  - [x] Compare test/ vs tests/ structures
  - [x] Merge main's test/ into streamlit's tests/
  - [x] Resolve conflicts
  - [x] Verify tests still run
- [x] **Phase 4: Port Documentation Structure**
  - [x] Port docs/agents/ from main
  - [x] Merge useful content from streamlit
  - [x] Resolve conflicts
- [x] **Phase 5: Manual OCR Merges**
  - [x] Merge `ocr/utils/polygon_utils.py`
  - [x] Merge `ocr/datasets/db_collate_fn.py` (no differences)
  - [x] Review other ocr/ files

#### Session 3: Verification & Cleanup (1-2 hours)
- [x] **Phase 6: Verification and Testing**
  - [x] Run full test suite
  - [x] Test unified app
  - [x] Test AgentQMS
  - [x] Test training pipeline
  - [x] Test inference pipeline
- [x] **Phase 7: Clean Up and Documentation**
  - [x] Update CHANGELOG.md
  - [x] Update README.md if needed (not needed)
  - [x] Document manual merges
- [x] **Phase 8: CI/CD Updates**
  - [x] Review CI/CD workflows
  - [x] Update for merged structure (added agent_qms/** to paths)
  - [x] Test CI/CD (workflows compatible)

#### Session 4: Final Review (if needed)
- [ ] Final verification
- [ ] Address any remaining issues
- [ ] Create implementation plan for redoing 3 plans

---

## Executive Summary

This plan outlines the strategy to merge `main` and `12_refactor/streamlit` branches, preserving critical improvements from both while minimizing conflicts and maintaining code quality.

### Key Decisions

1. **Base Branch**: Use `12_refactor/streamlit` as base (has unified app, checkpoint catalog, experiment registry)
2. **Selective Merge**: Cherry-pick/port specific improvements from `main`
3. **Preserve Streamlit Improvements**: Unified app and checkpoint catalog are too valuable to lose

---

## Branch Comparison Summary

### What's in Main (Keep These)
- ✅ **AgentQMS** (163 files) - Artifact management system
- ✅ **Scripts refactor** - Reorganized scripts directory
- ✅ **Documentation purge/refactor** - Cleaned up docs structure
- ✅ **test/** directory - Unit/integration tests
- ✅ **PLAN-001/002/003 fixes** in `ocr/`:
  - Numerical stability fixes (db_head.py)
  - Polygon validation consolidation
  - WandB lazy import optimization

### What's in Streamlit (Keep These)
- ✅ **Unified OCR App** (29 files, 6,400+ lines) - Multi-page Streamlit app
- ✅ **Checkpoint Catalog System** (10 files, 2,774+ lines) - Major refactoring
- ✅ **Experiment Registry** (`ocr/experiment_registry.py`) - New feature
- ✅ **Metadata Callback** (`ocr/lightning_modules/callbacks/metadata_callback.py`) - New feature
- ✅ **Enhanced tests/** directory - More comprehensive test coverage
- ✅ **UI improvements** - Inference app enhancements
- ✅ **WandB fix** (`runners/train.py`) - Proper enable check

### Conflicts to Resolve
- ⚠️ **ocr/** directory - Both branches have improvements
- ⚠️ **tests/** vs **test/** - Different test directory structures
- ⚠️ **Documentation** - Different structures and content
- ⚠️ **Scripts** - Different organizations

---

## Detailed File-by-File Analysis

### 1. OCR Directory (`ocr/`) - 24 files changed

#### Files with Changes in Both Branches

| File | Main Changes | Streamlit Changes | Decision |
|------|-------------|-------------------|----------|
| `ocr/models/head/db_head.py` | ✅ Numerical stability fix | ❌ Original buggy code | **Take from MAIN** |
| `ocr/models/loss/dice_loss.py` | Basic validation | Better edge case handling | **Take from STREAMLIT** (more defensive) |
| `ocr/lightning_modules/callbacks/wandb_completion.py` | Lazy imports | Lazy imports + directory fix | **Take from STREAMLIT** (has fix) |
| `ocr/lightning_modules/callbacks/unique_checkpoint.py` | Lazy imports | Enhanced + lazy imports | **Take from STREAMLIT** |
| `ocr/utils/polygon_utils.py` | PLAN-002 improvements | Different improvements | **Merge manually** |
| `ocr/datasets/db_collate_fn.py` | PLAN-002 improvements | Different improvements | **Merge manually** |

#### New Files in Streamlit (Keep)
- ✅ `ocr/experiment_registry.py` (126 lines) - New feature
- ✅ `ocr/lightning_modules/callbacks/metadata_callback.py` (509 lines) - New feature

#### Files Only in Main (Review)
- Review all 24 files for context-specific improvements

**Recommendation**: Take `ocr/` from streamlit, then manually port critical fixes from main (especially db_head.py numerical stability).

---

### 2. Tests Directory

#### Main: `test/` (Unit/Integration Tests)
- `tests/integration/` - Integration tests
- `tests/ocr/` - OCR-specific tests
- `tests/unit/` - Unit tests

#### Streamlit: `tests/` (More Comprehensive)
- `tests/integration/` - Integration tests (enhanced)
- `tests/debug/` - Debug utilities (NEW)
- `tests/demos/` - Demo scripts (NEW)
- `tests/manual/` - Manual testing scripts (NEW)
- `tests/scripts/` - Script tests (NEW)
- `tests/performance/` - Performance tests

**Decision**:
- ✅ **Keep streamlit's `tests/`** (more comprehensive)
- ✅ **Merge main's `test/` content** into streamlit's `tests/` structure
- ⚠️ **Resolve conflicts** in shared test files

---

### 3. AgentQMS System

**Location**: `agent_qms/`
**Status**: Only in main (163 files)
**Decision**: ✅ **Port from main** - Critical for artifact management

**Files to Port**:
```
agent_qms/
├── toolbelt/
│   ├── __init__.py
│   ├── core.py (467 lines)
│   └── validation.py (82 lines)
├── schemas/
│   ├── assessment.json
│   ├── bug_report.json
│   └── implementation_plan.json
├── templates/
│   ├── assessment.md
│   ├── bug_report.md
│   └── implementation_plan.md
└── q-manifest.yaml
```

---

### 4. Scripts Directory

**Main**: Refactored structure (163 files)
**Streamlit**: Older structure (92 files)
**Decision**: ✅ **Take from main** - Better organization

**Key Changes in Main**:
- Reorganized into logical subdirectories
- Better tool discovery
- Improved documentation

---

### 5. Documentation

**Main**:
- Purged/refactored structure
- `docs/agents/` system
- Cleaner organization

**Streamlit**:
- `docs/ai_handbook/` system
- More comprehensive guides
- Different structure

**Decision**: ✅ **Take from main** (purged/refactored), then merge useful content from streamlit

**Strategy**:
1. Keep main's structure
2. Port useful guides from streamlit's `docs/ai_handbook/`
3. Merge protocol documentation

---

### 6. UI/Streamlit Apps

**Streamlit Branch**:
- ✅ Unified OCR App (29 files, 6,400+ lines)
- ✅ Enhanced Inference App
- ✅ Checkpoint Catalog System (10 files, 2,774+ lines)

**Main**: Basic inference app only

**Decision**: ✅ **Keep from streamlit** - This is the main value

---

### 7. Configuration Files

**Review Needed**:
- `configs/ui/unified_app.yaml` - Only in streamlit
- `configs/logger/wandb.yaml` - Check for differences
- Other config differences

---

## Merge Strategy

### Phase 1: Preparation (Current Branch: streamlit)

1. **Create backup branch**
   ```bash
   git checkout -b merge-backup-streamlit
   git push origin merge-backup-streamlit
   ```

2. **Document current state**
   - List all critical files
   - Note any uncommitted changes

### Phase 2: Port Critical Fixes from Main

#### 2.1 Port AgentQMS System
```bash
# From main branch
git checkout main
git checkout streamlit -- agent_qms/
# Resolve any conflicts
```

#### 2.2 Port Scripts Refactor
```bash
# From main branch
git checkout main
git checkout streamlit -- scripts/
# Resolve conflicts, keep main's structure
```

#### 2.3 Port Critical OCR Fixes
```bash
# Port numerical stability fix
git show main:ocr/models/head/db_head.py > ocr/models/head/db_head.py
# Review and merge other ocr/ files manually
```

#### 2.4 Port Documentation Structure
```bash
# Take main's docs structure
git checkout main -- docs/agents/
# Then merge useful content from streamlit
```

#### 2.5 Port Test Directory
```bash
# Merge test/ from main into tests/ in streamlit
# Keep streamlit's structure, add main's tests
```

### Phase 3: Verify and Test

1. **Run tests**
   ```bash
   pytest tests/ -v
   ```

2. **Check imports**
   ```bash
   python -m py_compile ocr/**/*.py
   python -m py_compile ui/**/*.py
   ```

3. **Verify AgentQMS**
   ```bash
   python scripts/agent_tools/core/discover.py
   ```

4. **Test unified app**
   ```bash
   streamlit run ui/apps/unified_ocr_app/app.py
   ```

### Phase 4: Clean Up and Documentation

1. Update CHANGELOG.md
2. Update README.md if needed
3. Document any manual merges
4. Create implementation plan for redoing the 3 plans if needed

---

## Critical Files to Handle Manually

### High Priority
1. ✅ `ocr/models/head/db_head.py` - **Already had fix in streamlit** (numerical stability verified)
2. ✅ `runners/train.py` - **Kept from streamlit** (wandb fix preserved)
3. ✅ `ocr/experiment_registry.py` - **Kept from streamlit** (new feature preserved)
4. ✅ `ocr/lightning_modules/callbacks/metadata_callback.py` - **Kept from streamlit** (new feature preserved)

### Medium Priority
5. ✅ `ocr/utils/polygon_utils.py` - **Merged both improvements**
   - Kept streamlit's backward-compatible signature for `filter_degenerate_polygons()`
   - Added main's PLAN-002 validation functions: `validate_polygon_finite()`, `validate_polygon_area()`, `has_duplicate_consecutive_points()`, `is_valid_polygon()`
6. ✅ `ocr/datasets/db_collate_fn.py` - **No differences** (already merged)
7. ✅ `ocr/models/loss/dice_loss.py` - **Kept from streamlit** (more defensive, already in streamlit)

### Low Priority
8. ✅ Other `ocr/` files - **Reviewed, no conflicts requiring manual merge**

## Manual Merge Summary

### Completed Manual Merges

1. **polygon_utils.py** (2025-11-12)
   - **Decision**: Merge both versions
   - **Action**: 
     - Preserved streamlit's `filter_degenerate_polygons()` signature with backward-compatible parameters
     - Added main's validation functions after `filter_degenerate_polygons()`
     - All functions are now available in the merged version
   - **Result**: Best of both branches - backward compatibility + enhanced validation

2. **db_collate_fn.py** (2025-11-12)
   - **Decision**: No action needed
   - **Action**: Verified files are identical between branches
   - **Result**: No merge required

3. **db_head.py** (2025-11-12)
   - **Decision**: No action needed
   - **Action**: Verified streamlit already has numerical stability fix
   - **Result**: No merge required

---

## AI Assistance Prompts for Conflict Resolution

### When Encountering Merge Conflicts

**Prompt Template for AI Assistant:**
```
I'm merging branches and encountered a conflict in [FILE_PATH].

Context:
- Base branch: 12_refactor/streamlit
- Source branch: main
- Conflict type: [content conflict / file deletion / rename]

The conflict markers show:
- <<<<<<< HEAD (streamlit version)
- =======
- >>>>>>> main (main version)

Decision criteria:
- [SPECIFIC_DECISION_CRITERIA from this plan]

Please help me resolve this conflict by:
1. Analyzing both versions
2. Recommending the correct resolution based on the merge plan
3. Explaining why this resolution preserves the best of both branches
```

### Example Prompts for Specific Files

**For `ocr/models/head/db_head.py`:**
```
I need to port the numerical stability fix from main to streamlit branch.

Current file (streamlit) has the buggy code:
[PASTE_BUGGY_CODE]

Main branch has the fix:
[PASTE_FIXED_CODE]

Please help me:
1. Replace the _step_function method with the fixed version from main
2. Ensure no other changes are lost
3. Verify the fix is correctly applied
```

**For `ocr/utils/polygon_utils.py`:**
```
I need to merge improvements from both branches.

Streamlit version has: [DESCRIBE_STREAMLIT_CHANGES]
Main version has: [DESCRIBE_MAIN_CHANGES]

Please help me:
1. Identify which functions/features are unique to each branch
2. Combine both sets of improvements
3. Ensure no functionality is lost
4. Check for any conflicts or duplicate code
```

**For test directory conflicts:**
```
I'm merging test directories. Streamlit has `tests/` with more comprehensive structure, main has `test/` with different content.

Please help me:
1. Identify which tests exist in both
2. Identify unique tests in each
3. Recommend how to merge into streamlit's `tests/` structure
4. Ensure no tests are lost
```

---

## Implementation Steps

### Step 1: Create Working Branch
```bash
git checkout 12_refactor/streamlit
git checkout -b merge-main-into-streamlit
```

### Step 2: Port AgentQMS (High Priority)
```bash
git checkout main -- agent_qms/
git add agent_qms/
git commit -m "Port AgentQMS system from main"
```

### Step 3: Port Scripts Refactor (High Priority)
```bash
# Backup current scripts
cp -r scripts scripts_backup

# Take from main
git checkout main -- scripts/

# Resolve conflicts, test
git add scripts/
git commit -m "Port scripts refactor from main"
```

### Step 4: Port Critical OCR Fixes (High Priority)
```bash
# Port db_head.py fix
git show main:ocr/models/head/db_head.py > ocr/models/head/db_head.py
git add ocr/models/head/db_head.py
git commit -m "Port numerical stability fix from main (PLAN-001)"

# Review and merge other ocr/ files
# Use git diff to see differences
```

### Step 5: Merge Test Directories (Medium Priority)
```bash
# Keep streamlit's tests/ structure
# Manually copy/merge test/ content from main
# Resolve conflicts
```

### Step 6: Port Documentation (Medium Priority)
```bash
# Take main's structure
git checkout main -- docs/agents/

# Merge useful content from streamlit
# Manual review needed
```

### Step 7: Verify Everything Works
```bash
# Run tests
pytest tests/ -v

# Check imports
python -c "import ocr; import ui; import agent_qms"

# Test unified app
streamlit run ui/apps/unified_ocr_app/app.py --server.headless=true
```

---

## Risk Assessment

### High Risk Areas
1. **OCR directory** - Many files changed, need careful merging
2. **Test directories** - Different structures, potential conflicts
3. **Documentation** - Different philosophies, need manual review

### Medium Risk Areas
1. **Scripts** - Should be straightforward port
2. **Config files** - Need to check for conflicts

### Low Risk Areas
1. **AgentQMS** - Self-contained, should port cleanly
2. **UI apps** - Already in streamlit, just need to preserve

---

## Rollback Plan

If merge fails:
1. `git checkout merge-backup-streamlit` - Restore backup
2. Or `git reset --hard origin/12_refactor/streamlit` - Reset to original

---

## Post-Merge Tasks

1. ✅ Update CHANGELOG.md
2. ✅ Run full test suite
3. ✅ Update documentation
4. ✅ Create new implementation plan if needed for redoing the 3 plans
5. ✅ Verify all features work:
   - Unified app
   - Checkpoint catalog
   - AgentQMS
   - Training pipeline
   - Inference pipeline

---

## Decision Log

### Key Decisions Made
1. **Base branch**: `12_refactor/streamlit` (has unified app)
2. **OCR directory**: Take from streamlit, port critical fixes from main
3. **Tests**: Keep streamlit's structure, merge main's content
4. **AgentQMS**: Port from main (not in streamlit)
5. **Scripts**: Port from main (better organization)
6. **Documentation**: Take main's structure, merge useful content

### Open Questions (RESOLVED)

- [x] **Should we redo the 3 implementation plans after merge?**
  - **Answer**: Yes, will redo implementation with web workers

- [x] **Are there any streamlit-specific dependencies we need to handle?**
  - **Answer**: Streamlit is installed. Enhanced preprocessor dependencies (rembg) are probably defined in pyproject.toml in streamlit branch. Check and merge if needed.

- [x] **Do we need to update CI/CD for the merged structure?**
  - **Answer**: Yes, CI/CD needs updating for the merged structure

---

## Next Steps

1. Review this plan
2. Create backup branch
3. Start with Phase 1 (AgentQMS port)
4. Proceed step by step, testing at each phase
5. Document any issues encountered
6. Update plan as needed

---

## Session Handover Protocol

### For Continuing in a New Session

**When starting a new session, provide this context to the AI:**

```
I'm continuing the merge plan execution. Current status:

- Progress Tracker STATUS: [CURRENT_STATUS]
- Current Session: [SESSION_NUMBER]
- Last Completed Task: [TASK_DESCRIPTION]
- Next Task: [TASK_DESCRIPTION]

The merge plan is in MERGE_PLAN.md. Please:
1. Review the Progress Tracker
2. Identify the next task
3. Continue from where we left off
4. Update the Progress Tracker as you complete tasks
```

### Progress Tracker Update Protocol

**After completing each task:**
1. Mark the task as `[x]` completed
2. Update `LAST COMPLETED TASK` with task description
3. Update `NEXT TASK` with the next task
4. Update `CURRENT STEP` if moving to a new phase
5. If starting a new session, update `CURRENT SESSION`

**Example update:**
```markdown
- **STATUS**: In Progress
- **CURRENT SESSION**: Session 1 - Preparation & Critical Fixes
- **CURRENT STEP**: Phase 2, Task 2.1 - Port db_head.py fix
- **LAST COMPLETED TASK**: Created backup branch and working branch
- **NEXT TASK**: Port numerical stability fix from main to streamlit
```

---

## Dependency Check

### Before Starting Session 1

**Check pyproject.toml for dependencies:**
```bash
# Check if rembg or other streamlit-specific deps exist
git show 12_refactor/streamlit:pyproject.toml | grep -i "rembg\|streamlit"
git show main:pyproject.toml | grep -i "rembg\|streamlit"

# Compare and merge if needed
```

**Action Items:**
- [ ] Compare pyproject.toml between branches
- [ ] Identify streamlit-specific dependencies
- [ ] Merge dependency lists if needed

---

**Last Updated**: 2025-11-12
**Status**: Ready for execution
**Next Session**: Session 1 - Preparation & Critical Fixes

