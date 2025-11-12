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

- **STATUS:** Not Started / In Progress / Completed
- **CURRENT STEP:** [Current Phase, Task # - Task Name]
- **LAST COMPLETED TASK:** [Description of last completed task]
- **NEXT TASK:** [Description of the immediate next task]

### Implementation Outline (Checklist)

#### **Phase 1: [Phase 1 Title] (Week [Number])**
1. [ ] **Task 1.1: [Task 1.1 Title]**
   - [ ] [Sub-task 1.1.1 description]
   - [ ] [Sub-task 1.1.2 description]
   - [ ] [Sub-task 1.1.3 description]

2. [ ] **Task 1.2: [Task 1.2 Title]**
   - [ ] [Sub-task 1.2.1 description]
   - [ ] [Sub-task 1.2.2 description]

#### **Phase 2: [Phase 2 Title] (Week [Number])**
3. [ ] **Task 2.1: [Task 2.1 Title]**
   - [ ] [Sub-task 2.1.1 description]
   - [ ] [Sub-task 2.1.2 description]

4. [ ] **Task 2.2: [Task 2.2 Title]**
   - [ ] [Sub-task 2.2.1 description]
   - [ ] [Sub-task 2.2.2 description]

*(Add more Phases and Tasks as needed)*

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] [Architectural Principle 1 (e.g., Modular Design)]
- [ ] [Data Model Requirement (e.g., Pydantic V2 Integration)]
- [ ] [Configuration Method (e.g., YAML-Driven)]
- [ ] [State Management Strategy]

### **Integration Points**
- [ ] [Integration with System X]
- [ ] [API Endpoint Definition]
- [ ] [Use of Existing Utility/Library]

### **Quality Assurance**
- [ ] [Unit Test Coverage Goal (e.g., > 90%)]
- [ ] [Integration Test Requirement]
- [ ] [Performance Test Requirement]
- [ ] [UI/UX Test Requirement]

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] [Key Feature 1 Works as Expected]
- [ ] [Key Feature 2 is Fully Implemented]
- [ ] [Performance Metric is Met (e.g., <X ms latency)]
- [ ] [User-Facing Outcome is Achieved]

### **Technical Requirements**
- [ ] [Code Quality Standard is Met (e.g., Documented, type-hinted)]
- [ ] [Resource Usage is Within Limits (e.g., <X GB memory)]
- [ ] [Compatibility with System Y is Confirmed]
- [ ] [Maintainability Goal is Met]

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW / MEDIUM / HIGH
### **Active Mitigation Strategies**:
1. [Mitigation Strategy 1 (e.g., Incremental Development)]
2. [Mitigation Strategy 2 (e.g., Comprehensive Testing)]
3. [Mitigation Strategy 3 (e.g., Regular Code Quality Checks)]

### **Fallback Options**:
1. [Fallback Option 1 if Risk A occurs (e.g., Simplified version of a feature)]
2. [Fallback Option 2 if Risk B occurs (e.g., CPU-only mode)]
3. [Fallback Option 3 if Risk C occurs (e.g., Phased Rollout)]

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

**TASK:** [Description of the immediate next task]

**OBJECTIVE:** [Clear, concise goal of the task]

**APPROACH:**
1. [Step 1 to execute the task]
2. [Step 2 to execute the task]
3. [Step 3 to execute the task]

**SUCCESS CRITERIA:**
- [Measurable outcome 1 that defines task completion]
- [Measurable outcome 2 that defines task completion]

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
## Objective

Clean up legacy, deprecated, and obsolete files in the `ui/` directory to eliminate confusion for AI agents and users. This plan addresses 8 identified files through a phased approach that minimizes risk while ensuring clear migration paths.

## Approach

The implementation follows a risk-stratified, phased approach:
1. **Immediate**: Remove obsolete files with zero dependencies (low risk)
2. **Short-term**: Add deprecation notices and migrate simple utilities (medium risk)
3. **Long-term**: Migrate complex services and remove wrappers (high risk, requires careful planning)

Each phase includes validation steps to ensure no regressions.

## Implementation Steps

### Phase 1: Remove Obsolete Files (Week 1) - Low Risk

**Goal**: Remove files with no dependencies that cause confusion.

#### Step 1.1: Remove `ui/_visualization/comparison.py`
- **Action**: Delete the file
- **Validation**:
  - Verify no imports exist: `grep -r "_visualization" .`
  - Confirm `ui/visualization/comparison.py` provides all needed functionality
- **Files to modify**: None (deletion only)
- **Risk**: Very Low - No dependencies found

#### Step 1.2: Remove `ui/test_viewer.py`
- **Action**: Delete the file
- **Validation**:
  - Verify not referenced in `run_ui.py`
  - Confirm `ui/evaluation/app.py` is the production version
  - Check no imports: `grep -r "test_viewer" .`
- **Files to modify**: None (deletion only)
- **Risk**: Very Low - Test file not in production use

#### Step 1.3: Remove `ui/apps/unified_ocr_app/backup/app_monolithic_backup_2025-10-21.py`
- **Action**: Delete the backup file
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

#### Step 4.3: Remove `ui/utils/command_builder.py`
- **Action**: Delete the deprecated file
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

#### Step 5.8: Remove legacy catalog (Month 3)
- **Action**:
  - Delete `ui/apps/inference/services/checkpoint_catalog.py`
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

#### Step 6.4: Remove wrapper files
- **Action**:
  - Delete `ui/command_builder.py`
  - Delete `ui/inference_ui.py`
  - Delete `ui/evaluation_viewer.py`
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
