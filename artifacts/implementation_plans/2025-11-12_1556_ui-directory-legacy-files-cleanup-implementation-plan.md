---
title: "UI Directory Legacy Files Cleanup Implementation Plan"
author: "ai-agent"
date: "2025-11-12"
timestamp: "2025-11-12 15:56 KST"
type: "implementation_plan"
category: "development"
status: "draft"
tags: ["legacy", "cleanup", "ui", "refactoring", "migration"]
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **UI Directory Legacy Files Cleanup Implementation Plan**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: UI Directory Legacy Files Cleanup Implementation Plan

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Not Started
- **CURRENT STEP:** Phase 1, Task 1.1 - Remove ui/_visualization/comparison.py
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** Verify and remove obsolete files with zero dependencies

### Implementation Outline (Checklist)

#### **Phase 1: Remove Obsolete Files (Week 1) - Low Risk - PURGE COMPLETE**
1. [ ] **Task 1.1: Remove ui/_visualization/comparison.py**
   - [ ] Verify no imports exist: `grep -r "_visualization" .`
   - [ ] Confirm `ui/visualization/comparison.py` provides all needed functionality
   - [ ] Delete `ui/_visualization/comparison.py`
   - [ ] Verify deletion: `ls ui/_visualization/comparison.py` (should fail)
   - [ ] Check git status to confirm removal

2. [ ] **Task 1.2: Remove ui/test_viewer.py**
   - [ ] Verify not referenced in `run_ui.py`
   - [ ] Confirm `ui/evaluation/app.py` is the production version
   - [ ] Check no imports: `grep -r "test_viewer" .`
   - [ ] Delete `ui/test_viewer.py`
   - [ ] Verify deletion: `ls ui/test_viewer.py` (should fail)
   - [ ] Check git status to confirm removal

3. [ ] **Task 1.3: Remove ui/apps/unified_ocr_app/backup/app_monolithic_backup_2025-10-21.py**
   - [ ] Verify not imported anywhere: `grep -r "app_monolithic_backup" .`
   - [ ] Confirm backup is not needed (check git history if needed)
   - [ ] Delete backup file
   - [ ] Verify deletion
   - [ ] Check git status to confirm removal

#### **Phase 2: Add Deprecation Notices (Week 1) - Low Risk**
4. [ ] **Task 2.1: Add deprecation notice to ui/command_builder.py**
   - [ ] Add prominent deprecation warning at top of file
   - [ ] Reference actual implementation: `ui/apps/command_builder/app.py`
   - [ ] Add warning about future removal
   - [ ] Verify deprecation notice is visible

5. [ ] **Task 2.2: Add deprecation notice to ui/inference_ui.py**
   - [ ] Enhance existing deprecation notice
   - [ ] Make it more prominent in docstring
   - [ ] Reference actual implementation: `ui/apps/inference/app.py`
   - [ ] Verify deprecation notice is visible

6. [ ] **Task 2.3: Add deprecation notice to ui/evaluation_viewer.py**
   - [ ] Add prominent deprecation warning
   - [ ] Reference actual implementation: `ui/evaluation/app.py`
   - [ ] Add warning about future removal
   - [ ] Verify deprecation notice is visible

#### **Phase 3: Update Documentation (Week 1-2) - Low Risk**
7. [ ] **Task 3.1: Update ui/README.md**
   - [ ] Add architecture diagram showing wrapper vs. actual implementations
   - [ ] Add "Legacy Files" section explaining wrapper files
   - [ ] Update all code examples to use actual implementations
   - [ ] Add migration guide
   - [ ] Verify documentation is clear

8. [ ] **Task 3.2: Update other documentation references**
   - [ ] Search and update references in `docs/` directory
   - [ ] Update any markdown files referencing wrapper files
   - [ ] Update code comments
   - [ ] Verify all references point to actual implementations

#### **Phase 4: Migrate Deprecated Utilities (Week 2) - Medium Risk**
9. [ ] **Task 4.1: Update scripts/demos/demo_ui.py**
   - [ ] Change import from `from ui.utils.command_builder import CommandBuilder`
   - [ ] To: `from ui.utils.command import CommandBuilder`
   - [ ] Test the demo script still works
   - [ ] Verify functionality is unchanged

10. [ ] **Task 4.2: Verify no other imports**
    - [ ] Search codebase: `grep -r "from ui.utils.command_builder" .`
    - [ ] Search: `grep -r "import.*command_builder" .`
    - [ ] Update any found references
    - [ ] Verify all imports updated

11. [ ] **Task 4.3: Remove ui/utils/command_builder.py**
    - [ ] Run all tests to verify no breakage
    - [ ] Verify no broken imports
    - [ ] Check demo script works
    - [ ] Delete `ui/utils/command_builder.py`
    - [ ] Verify deletion
    - [ ] Check git status to confirm removal

#### **Phase 5: Migrate Checkpoint Catalog (Months 1-3) - High Risk**
12. [ ] **Task 5.1: Create migration tracking**
    - [ ] Document all current usages of `checkpoint_catalog.py`
    - [ ] Create checklist of files to update
    - [ ] Set up monitoring for usage
    - [ ] Document migration plan

13. [ ] **Task 5.2: Update ui/apps/inference/app.py**
    - [ ] Change import from `from .services.checkpoint_catalog import ...`
    - [ ] To: `from .services.checkpoint.catalog import ...`
    - [ ] Update function calls if API differs
    - [ ] Test inference UI works
    - [ ] Verify no regressions

14. [ ] **Task 5.3: Update ui/apps/unified_ocr_app/services/inference_service.py**
    - [ ] Update import to V2 catalog
    - [ ] Test unified OCR app works
    - [ ] Verify no regressions

15. [ ] **Task 5.4: Update ui/apps/inference/models/checkpoint.py**
    - [ ] Update import to V2 catalog
    - [ ] Verify model loading still works
    - [ ] Test checkpoint loading functionality

16. [ ] **Task 5.5: Update test files**
    - [ ] Update `tests/integration/test_checkpoint_fixes.py`
    - [ ] Update any other test files using legacy catalog
    - [ ] Run full test suite
    - [ ] Verify all tests pass

17. [ ] **Task 5.6: Add deprecation warnings to legacy file**
    - [ ] Add `warnings.warn()` to all public functions in `checkpoint_catalog.py`
    - [ ] Document removal timeline
    - [ ] Verify warnings appear when used

18. [ ] **Task 5.7: Monitor and validate (Month 2-3)**
    - [ ] Monitor for any new usages of legacy catalog
    - [ ] Verify all functionality works with V2
    - [ ] Collect feedback
    - [ ] Document any issues

19. [ ] **Task 5.8: Remove legacy catalog (Month 3) - PURGE COMPLETE**
    - [ ] Verify all imports updated to V2
    - [ ] Run full test suite
    - [ ] Delete `ui/apps/inference/services/checkpoint_catalog.py`
    - [ ] Verify deletion
    - [ ] Check git status to confirm removal
    - [ ] Verify no broken imports

#### **Phase 6: Remove Wrapper Files (Months 3-6) - High Risk - PURGE COMPLETE**
20. [ ] **Task 6.1: Update run_ui.py**
    - [ ] Change `run_command_builder()` to call `ui.apps.command_builder.app.main()` directly
    - [ ] Change `run_evaluation_viewer()` to call `ui.evaluation.app.main()` directly
    - [ ] Change `run_inference_ui()` to call `ui.apps.inference.app.main()` directly
    - [ ] Test all UI apps launch correctly
    - [ ] Verify functionality unchanged

21. [ ] **Task 6.2: Update all documentation**
    - [ ] Remove references to wrapper files
    - [ ] Update all examples to use direct imports
    - [ ] Update README files
    - [ ] Verify documentation is accurate

22. [ ] **Task 6.3: Verify no direct usage**
    - [ ] Search for `streamlit run ui/command_builder.py`
    - [ ] Search for `streamlit run ui/inference_ui.py`
    - [ ] Search for `streamlit run ui/evaluation_viewer.py`
    - [ ] Update any scripts or docs found
    - [ ] Verify all references updated

23. [ ] **Task 6.4: Remove wrapper files - PURGE COMPLETE**
    - [ ] Delete `ui/command_builder.py`
    - [ ] Delete `ui/inference_ui.py`
    - [ ] Delete `ui/evaluation_viewer.py`
    - [ ] Verify all deletions
    - [ ] Verify `run_ui.py` works
    - [ ] Run all UI apps
    - [ ] Check no broken references
    - [ ] Check git status to confirm removals

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] All obsolete files completely removed (PURGED, not deprecated)
- [ ] Wrapper files marked with clear deprecation notices
- [ ] All imports migrated to actual implementations
- [ ] Documentation clearly identifies active vs. legacy files

### **Integration Points**
- [ ] All UI apps work correctly after changes
- [ ] `run_ui.py` calls implementations directly (after Phase 6)
- [ ] Demo scripts use correct imports
- [ ] Checkpoint catalog fully migrated to V2

### **Quality Assurance**
- [ ] All tests pass after each phase
- [ ] No broken imports or references
- [ ] All UI apps launch and function correctly
- [ ] No regressions in functionality

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] All obsolete files completely removed (PURGED)
- [ ] All wrapper files have clear deprecation notices
- [ ] All UI apps work correctly
- [ ] No confusion about which files are source of truth
- [ ] AI agents can clearly identify active vs. legacy files

### **Technical Requirements**
- [ ] All imports migrated to actual implementations
- [ ] No broken references or imports
- [ ] All tests pass
- [ ] Documentation is accurate and up-to-date
- [ ] Legacy files completely removed (not just deprecated)

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW (Phase 1-3), MEDIUM (Phase 4), HIGH (Phase 5-6)
### **Active Mitigation Strategies**:
1. **Incremental Development**: Phased approach minimizes risk
2. **Complete Removal**: Legacy files PURGED completely to prevent confusion (not just deprecated)
3. **Comprehensive Testing**: All tests pass after each phase
4. **Git History**: All deleted files preserved in git history for rollback
5. **Validation Steps**: Verify no dependencies before deletion

### **Fallback Options**:
1. **Phase 1-3**: Git revert if issues found, files can be restored from git history
2. **Phase 4**: Revert import changes, restore `ui/utils/command_builder.py` from git
3. **Phase 5**: Keep legacy file until fully validated, can switch back via imports (feature flag exists)
4. **Phase 6**: Restore wrapper files from git, update `run_ui.py` to use wrappers again

---

## 🔄 **Blueprint Update Protocol**

**Update Triggers:**
- Task completion (move to next task)
- Blocker encountered (document and propose solution)
- Technical discovery (update approach if needed)
- Quality gate failure (address issues before proceeding)

**Update Format:**
1. Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
2. Mark completed items with [x]
3. Add any new discoveries or changes to approach
4. Update risk assessment if needed

---

## 🚀 **Immediate Next Action**

**TASK:** Remove ui/_visualization/comparison.py

**OBJECTIVE:** Completely PURGE obsolete duplicate file that causes confusion. This file has no dependencies and should be removed immediately.

**APPROACH:**
1. Verify no imports exist: `grep -r "_visualization" .` (excluding the file itself)
2. Confirm `ui/visualization/comparison.py` provides all needed functionality
3. Delete `ui/_visualization/comparison.py` completely
4. Verify deletion: `ls ui/_visualization/comparison.py` (should fail)
5. Check git status to confirm removal
6. Update progress tracker

**SUCCESS CRITERIA:**
- File completely removed (PURGED, not deprecated)
- No broken imports or references
- Git status shows deletion
- Ready to proceed to next obsolete file removal

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
## Objective

**PURGE legacy, deprecated, and obsolete files in the `ui/` directory completely** to eliminate confusion for AI agents and users. Legacy scripts and modules that remain in the project continue to get updated, causing major confusion. This plan addresses 8 identified files through a phased approach that **completely removes** obsolete files (not just deprecates them) to prevent future confusion and accidental updates.

## Approach

The implementation follows a risk-stratified, phased approach with **complete removal (PURGE)** as the goal:
1. **Immediate**: **PURGE** obsolete files with zero dependencies (low risk) - Complete removal, not deprecation
2. **Short-term**: Add deprecation notices to wrappers and migrate simple utilities, then **PURGE** deprecated utilities (medium risk)
3. **Long-term**: Migrate complex services and **PURGE** wrappers after migration (high risk, requires careful planning)

**Key Principle**: Legacy files that cause confusion must be **completely removed (PURGED)**, not just deprecated. Deprecation is only a temporary step for files that require migration. Once migration is complete, files are PURGED.

Each phase includes validation steps to ensure no regressions before removal.

## Implementation Steps

### Phase 1: PURGE Obsolete Files (Week 1) - Low Risk

**Goal**: **Completely remove (PURGE)** files with no dependencies that cause confusion. These files will be deleted entirely, not deprecated.

#### Step 1.1: PURGE `ui/_visualization/comparison.py`
- **Action**: **Completely delete (PURGE)** the file - no deprecation, immediate removal
- **Validation**:
  - Verify no imports exist: `grep -r "_visualization" .`
  - Confirm `ui/visualization/comparison.py` provides all needed functionality
- **Files to modify**: None (deletion only)
- **Risk**: Very Low - No dependencies found

#### Step 1.2: PURGE `ui/test_viewer.py`
- **Action**: **Completely delete (PURGE)** the file - test files should not be in production directories
- **Validation**:
  - Verify not referenced in `run_ui.py`
  - Confirm `ui/evaluation/app.py` is the production version
  - Check no imports: `grep -r "test_viewer" .`
- **Files to modify**: None (deletion only)
- **Risk**: Very Low - Test file not in production use

#### Step 1.3: PURGE `ui/apps/unified_ocr_app/backup/app_monolithic_backup_2025-10-21.py`
- **Action**: **Completely delete (PURGE)** the backup file - backup files should not be in version control
- **Validation**:
  - Verify not imported anywhere
  - Confirm backup is not needed (check git history if needed)
- **Files to modify**: None (deletion only)
- **Risk**: Very Low - Backup file should not be in version control

**Phase 1 Completion Criteria**:
- [ ] All three files deleted
- [ ] No broken imports or references
- [ ] Git commit with clear message

### Phase 2: Add Deprecation Notices (Week 1) - Low Risk

**Goal**: Clearly mark wrapper files as deprecated to prevent accidental updates.

#### Step 2.1: Add deprecation notice to `ui/command_builder.py`
- **Action**: Add prominent deprecation warning at top of file
- **Content to add**:
  ```python
  """
  ⚠️ DEPRECATED: This is a backward compatibility wrapper.

  This file exists only for backward compatibility with existing launch commands.
  The actual implementation is in: ui/apps/command_builder/app.py

  New code should import directly from: ui.apps.command_builder
  This wrapper will be removed in a future version.
  """
  ```
- **Files to modify**: `ui/command_builder.py`
- **Risk**: Low - Only adds documentation

#### Step 2.2: Add deprecation notice to `ui/inference_ui.py`
- **Action**: Enhance existing deprecation notice
- **Content to add**: Update docstring to be more prominent
- **Files to modify**: `ui/inference_ui.py`
- **Risk**: Low - Only adds documentation

#### Step 2.3: Add deprecation notice to `ui/evaluation_viewer.py`
- **Action**: Add prominent deprecation warning
- **Content to add**: Similar to Step 2.1
- **Files to modify**: `ui/evaluation_viewer.py`
- **Risk**: Low - Only adds documentation

**Phase 2 Completion Criteria**:
- [ ] All wrapper files have clear deprecation notices
- [ ] Notices are visible in docstrings and comments
- [ ] Notices reference actual implementation locations

### Phase 3: Update Documentation (Week 1-2) - Low Risk

**Goal**: Update all documentation to reference actual implementations.

#### Step 3.1: Update `ui/README.md`
- **Action**:
  - Add architecture diagram showing wrapper vs. actual implementations
  - Add "Legacy Files" section explaining wrapper files
  - Update all code examples to use actual implementations
  - Add migration guide
- **Files to modify**: `ui/README.md`
- **Risk**: Low - Documentation only

#### Step 3.2: Update other documentation references
- **Action**: Search and update references in:
  - `docs/` directory
  - Any markdown files referencing wrapper files
  - Code comments
- **Files to modify**: Various documentation files
- **Risk**: Low - Documentation only

**Phase 3 Completion Criteria**:
- [ ] `ui/README.md` has clear architecture section
- [ ] All documentation references point to actual implementations
- [ ] Migration guide is available

### Phase 4: Migrate Deprecated Utilities (Week 2) - Medium Risk

**Goal**: Migrate from `ui.utils.command_builder` to `ui.utils.command`.

#### Step 4.1: Update `scripts/demos/demo_ui.py`
- **Action**:
  - Change import from `from ui.utils.command_builder import CommandBuilder`
  - To: `from ui.utils.command import CommandBuilder`
  - Test the demo script still works
- **Files to modify**: `scripts/demos/demo_ui.py`
- **Risk**: Medium - Functional change, needs testing

#### Step 4.2: Verify no other imports
- **Action**:
  - Search codebase: `grep -r "from ui.utils.command_builder" .`
  - Search: `grep -r "import.*command_builder" .`
  - Update any found references
- **Files to modify**: Any files found with imports
- **Risk**: Medium - Need to verify all usages

#### Step 4.3: PURGE `ui/utils/command_builder.py`
- **Action**: **Completely delete (PURGE)** the deprecated file after migration is complete
- **Validation**:
  - Run all tests
  - Verify no broken imports
  - Check demo script works
- **Files to modify**: None (deletion)
- **Risk**: Medium - Final removal step

**Phase 4 Completion Criteria**:
- [ ] All imports updated to `ui.utils.command`
- [ ] `ui/utils/command_builder.py` removed
- [ ] All tests pass
- [ ] Demo script works correctly

### Phase 5: Migrate Checkpoint Catalog (Months 1-3) - High Risk

**Goal**: Migrate from legacy `checkpoint_catalog.py` to V2 catalog system.

#### Step 5.1: Create migration tracking
- **Action**:
  - Document all current usages of `checkpoint_catalog.py`
  - Create checklist of files to update
  - Set up monitoring for usage
- **Files to modify**: Documentation
- **Risk**: Low - Planning phase

#### Step 5.2: Update `ui/apps/inference/app.py`
- **Action**:
  - Change import from `from .services.checkpoint_catalog import ...`
  - To: `from .services.checkpoint.catalog import ...`
  - Update function calls if API differs
  - Test inference UI works
- **Files to modify**: `ui/apps/inference/app.py`
- **Risk**: High - Core functionality

#### Step 5.3: Update `ui/apps/unified_ocr_app/services/inference_service.py`
- **Action**:
  - Update import to V2 catalog
  - Test unified OCR app works
- **Files to modify**: `ui/apps/unified_ocr_app/services/inference_service.py`
- **Risk**: High - Core functionality

#### Step 5.4: Update `ui/apps/inference/models/checkpoint.py`
- **Action**:
  - Update import to V2 catalog
  - Verify model loading still works
- **Files to modify**: `ui/apps/inference/models/checkpoint.py`
- **Risk**: High - Model loading critical

#### Step 5.5: Update test files
- **Action**:
  - Update `tests/integration/test_checkpoint_fixes.py`
  - Update any other test files using legacy catalog
  - Run full test suite
- **Files to modify**: Test files
- **Risk**: Medium - Test updates

#### Step 5.6: Add deprecation warnings to legacy file
- **Action**:
  - Add `warnings.warn()` to all public functions in `checkpoint_catalog.py`
  - Document removal timeline
- **Files to modify**: `ui/apps/inference/services/checkpoint_catalog.py`
- **Risk**: Low - Adds warnings only

#### Step 5.7: Monitor and validate (Month 2-3)
- **Action**:
  - Monitor for any new usages of legacy catalog
  - Verify all functionality works with V2
  - Collect feedback
- **Files to modify**: None
- **Risk**: Low - Monitoring phase

#### Step 5.8: PURGE legacy catalog (Month 3)
- **Action**:
  - **Completely delete (PURGE)** `ui/apps/inference/services/checkpoint_catalog.py` after migration is complete
  - Verify no broken imports
  - Run full test suite
- **Files to modify**: None (deletion)
- **Risk**: High - Final removal

**Phase 5 Completion Criteria**:
- [ ] All imports updated to V2 catalog
- [ ] All tests pass
- [ ] All UI apps work correctly
- [ ] Legacy catalog removed
- [ ] No regressions

### Phase 6: Remove Wrapper Files (Months 3-6) - High Risk

**Goal**: Remove wrapper files after all references are updated.

#### Step 6.1: Update `run_ui.py`
- **Action**:
  - Change `run_command_builder()` to call `ui.apps.command_builder.app.main()` directly
  - Change `run_evaluation_viewer()` to call `ui.evaluation.app.main()` directly
  - Change `run_inference_ui()` to call `ui.apps.inference.app.main()` directly
- **Files to modify**: `run_ui.py`
- **Risk**: Medium - Entry point changes

#### Step 6.2: Update all documentation
- **Action**:
  - Remove references to wrapper files
  - Update all examples to use direct imports
  - Update README files
- **Files to modify**: Documentation files
- **Risk**: Low - Documentation only

#### Step 6.3: Verify no direct usage
- **Action**:
  - Search for `streamlit run ui/command_builder.py`
  - Search for `streamlit run ui/inference_ui.py`
  - Search for `streamlit run ui/evaluation_viewer.py`
  - Update any scripts or docs found
- **Files to modify**: Scripts and documentation
- **Risk**: Medium - Need to find all usages

#### Step 6.4: PURGE wrapper files
- **Action**:
  - **Completely delete (PURGE)** `ui/command_builder.py`
  - **Completely delete (PURGE)** `ui/inference_ui.py`
  - **Completely delete (PURGE)** `ui/evaluation_viewer.py`
- **Validation**:
  - Verify `run_ui.py` works
  - Run all UI apps
  - Check no broken references
- **Files to modify**: None (deletions)
- **Risk**: High - Final removal

**Phase 6 Completion Criteria**:
- [ ] `run_ui.py` calls implementations directly
- [ ] All documentation updated
- [ ] Wrapper files removed
- [ ] All UI apps work correctly
- [ ] No broken references

## Testing Strategy

### Unit Tests
- Run existing test suite after each phase
- Add tests for deprecated utility migration (Phase 4)
- Verify checkpoint catalog migration (Phase 5)

### Integration Tests
- Test all UI apps launch correctly
- Test `run_ui.py` commands work
- Test demo scripts work
- Verify no import errors

### Manual Testing
- Launch each UI app manually
- Verify functionality works as expected
- Check deprecation warnings appear (where applicable)
- Verify documentation is accurate

### Validation Commands
```bash
# Check for broken imports
grep -r "from ui.utils.command_builder" .
grep -r "from.*checkpoint_catalog" .
grep -r "_visualization" .
grep -r "test_viewer" .

# Run tests
pytest tests/

# Test UI apps
python run_ui.py command_builder
python run_ui.py inference
python run_ui.py evaluation_viewer
```

## Success Criteria

### Immediate (Phase 1-3)
- [ ] All obsolete files removed
- [ ] All wrapper files have clear deprecation notices
- [ ] Documentation updated with architecture diagram
- [ ] No broken imports or references

### Short-term (Phase 4)
- [ ] Deprecated utilities migrated
- [ ] `ui/utils/command_builder.py` removed
- [ ] All tests pass
- [ ] Demo scripts work

### Long-term (Phase 5-6)
- [ ] Checkpoint catalog fully migrated to V2
- [ ] Legacy catalog removed
- [ ] Wrapper files removed
- [ ] All UI apps work correctly
- [ ] No confusion about source of truth
- [ ] AI agents can clearly identify active vs. legacy files

## Risk Mitigation

### Low Risk Phases (1-3)
- Simple deletions and documentation updates
- Easy to rollback if needed
- No functional changes

### Medium Risk Phases (4)
- Functional changes but isolated
- Can test independently
- Easy to rollback

### High Risk Phases (5-6)
- Core functionality changes
- Requires thorough testing
- Staged rollout recommended
- Keep legacy code until fully validated

## Rollback Plan

### Phase 1-3: Documentation/Deletion
- Git revert if issues found
- Files can be restored from git history

### Phase 4: Utility Migration
- Revert import changes
- Restore `ui/utils/command_builder.py` from git

### Phase 5: Checkpoint Catalog
- Keep legacy file until fully validated
- Can switch back via imports
- Feature flag already exists

### Phase 6: Wrapper Removal
- Restore wrapper files from git
- Update `run_ui.py` to use wrappers again

## Timeline

- **Week 1**: Phases 1-3 (Remove obsolete files, add deprecation notices, update docs)
- **Week 2**: Phase 4 (Migrate utilities)
- **Month 1**: Phase 5 Steps 1-4 (Migrate checkpoint catalog - core files)
- **Month 2**: Phase 5 Steps 5-7 (Update tests, monitor)
- **Month 3**: Phase 5 Step 8 (Remove legacy catalog)
- **Months 3-6**: Phase 6 (Remove wrapper files)

## Dependencies

- Assessment document: `artifacts/assessments/2025-11-12_1520_ui-directory-legacy-files-audit.md`
- V2 checkpoint catalog must be stable and tested
- All UI apps must be functional before wrapper removal

## Notes

- Wrapper files serve backward compatibility but cause confusion
- Some files require careful migration due to active usage
- Documentation is critical to prevent future confusion
- Staged approach minimizes risk
- Each phase should be validated before proceeding
