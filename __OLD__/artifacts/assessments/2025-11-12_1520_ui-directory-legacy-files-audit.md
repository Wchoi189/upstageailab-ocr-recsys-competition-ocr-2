---
title: "UI Directory Legacy Files Audit"
author: "ai-agent"
date: "2025-11-12"
timestamp: "2025-11-12 15:20 KST"
status: "draft"
tags: ["legacy", "deprecation", "ui", "cleanup", "refactoring"]
---

## Progress Tracker
*(Required for iterative assessments, debugging sessions, or incremental work)*

- **STATUS:** Not Started / In Progress / Completed
- **CURRENT STEP:** [Current phase or task being worked on]
- **LAST COMPLETED TASK:** [Description of last completed task]
- **NEXT TASK:** [Description of the immediate next task]

### Assessment Checklist
- [ ] Initial assessment complete
- [ ] Analysis phase complete
- [ ] Recommendations documented
- [ ] Review and validation complete

---

## 1. Summary

This assessment identifies legacy, deprecated, and obsolete files in the `ui/` directory that should be considered for removal. These files continue to receive updates, causing confusion for AI agents and users about which files are the source of truth.

## 2. Assessment

## Findings

### High Priority - Backward Compatibility Wrappers

These files are thin wrappers that delegate to newer modular implementations. They exist only for backward compatibility but continue to be referenced and updated, causing confusion.

#### 1. `ui/command_builder.py`
- **Status**: Legacy wrapper
- **Purpose**: Backward compatibility wrapper delegating to `ui.apps.command_builder`
- **Replacement**: `ui/apps/command_builder/app.py`
- **Evidence**: File contains only a thin wrapper that imports and calls `ui.apps.command_builder.main`
- **Risk**: Low - Only used as entry point via `streamlit run ui/command_builder.py`
- **Recommendation**: Keep for now but add deprecation notice. Update all documentation to reference `ui/apps/command_builder/app.py` directly.

#### 2. `ui/inference_ui.py`
- **Status**: Legacy wrapper
- **Purpose**: Backward compatibility wrapper delegating to `ui.apps.inference`
- **Replacement**: `ui/apps/inference/app.py`
- **Evidence**: File explicitly states it's a "thin wrapper" for backward compatibility
- **Risk**: Low - Only used as entry point via `streamlit run ui/inference_ui.py`
- **Recommendation**: Keep for now but add deprecation notice. Update all documentation to reference `ui/apps/inference/app.py` directly.

#### 3. `ui/evaluation_viewer.py`
- **Status**: Legacy wrapper
- **Purpose**: Backward compatibility wrapper delegating to `ui.evaluation`
- **Replacement**: `ui/evaluation/app.py`
- **Evidence**: File comment states "This is now a wrapper around the modular ui.evaluation package"
- **Risk**: Low - Only used as entry point via `run_ui.py evaluation_viewer`
- **Recommendation**: Keep for now but add deprecation notice. Update all documentation to reference `ui/evaluation/app.py` directly.

### Medium Priority - Deprecated Utility Modules

#### 4. `ui/utils/command_builder.py`
- **Status**: Deprecated with warning
- **Purpose**: Backward compatibility wrapper for refactored command builder
- **Replacement**: `ui.utils.command` package
- **Evidence**: File contains explicit `DeprecationWarning` stating it will be removed in a future version
- **Current Usage**: Still imported in `scripts/demos/demo_ui.py` and referenced in documentation
- **Risk**: Medium - Still actively imported in some places
- **Recommendation**:
  1. Update `scripts/demos/demo_ui.py` to use `ui.utils.command` directly
  2. Update all documentation references
  3. Remove after migration period

### High Priority - Duplicate/Obsolete Files

#### 5. `ui/_visualization/comparison.py`
- **Status**: Obsolete duplicate
- **Purpose**: Appears to be legacy version of visualization comparison
- **Replacement**: `ui/visualization/comparison.py`
- **Evidence**:
  - Underscore prefix suggests private/legacy
  - No imports found in codebase
  - `ui/visualization/comparison.py` exists with similar functionality but more features
- **Risk**: Low - Not imported anywhere
- **Recommendation**: **REMOVE** - No dependencies found

#### 6. `ui/test_viewer.py`
- **Status**: Test file in production directory
- **Purpose**: Appears to be a test/development version of evaluation viewer
- **Replacement**: `ui/evaluation/app.py` (production version)
- **Evidence**:
  - File name suggests test file
  - Contains similar functionality to `ui/evaluation/app.py` but less complete
  - Not referenced in `run_ui.py` or any production code
- **Risk**: Low - Not used in production
- **Recommendation**: **REMOVE** - Test files should not be in production directories

#### 7. `ui/apps/unified_ocr_app/backup/app_monolithic_backup_2025-10-21.py`
- **Status**: Backup file
- **Purpose**: Backup of monolithic app before refactoring
- **Replacement**: N/A (historical backup)
- **Evidence**:
  - Located in `backup/` directory
  - Contains debug logging code
  - Dated filename suggests it's a historical backup
- **Risk**: Low - Not imported
- **Recommendation**: **REMOVE** - Backup files should not be in version control. Move to archive or delete.

### Medium Priority - Legacy Service with Deprecation Notice

#### 8. `ui/apps/inference/services/checkpoint_catalog.py`
- **Status**: Legacy V1 adapter (marked as deprecated)
- **Purpose**: Backward-compatible catalog building that delegates to V2 system
- **Replacement**: `ui/apps/inference/services/checkpoint/catalog.py` (V2)
- **Evidence**:
  - File header explicitly states "Legacy checkpoint catalog service (V1)"
  - Contains deprecation notice: "This module is maintained for backward compatibility only"
  - Internally uses V2 catalog system via feature flag
  - Still actively imported in multiple places:
    - `ui/apps/inference/app.py`
    - `ui/apps/unified_ocr_app/services/inference_service.py`
    - `ui/apps/inference/models/checkpoint.py`
    - `tests/integration/test_checkpoint_fixes.py`
- **Risk**: High - Still actively used in production code
- **Recommendation**:
  1. Create migration plan to update all imports to V2 catalog
  2. Update imports in all dependent files
  3. Add deprecation warnings to all public functions
  4. Remove after migration period (3-6 months)

## Impact Analysis

### Confusion Sources

1. **Multiple Entry Points**: Three wrapper files (`command_builder.py`, `inference_ui.py`, `evaluation_viewer.py`) create confusion about which file is the "real" implementation
2. **Duplicate Functionality**: `_visualization/comparison.py` and `visualization/comparison.py` have overlapping functionality
3. **Test Files in Production**: `test_viewer.py` suggests it might be a production file
4. **Backup Files in Repo**: Backup files create noise and confusion
5. **Deprecated but Active**: `checkpoint_catalog.py` is marked deprecated but still widely used

### Update Frequency Risk

Files that continue to receive updates:
- `ui/command_builder.py` - May receive updates if developers don't realize it's a wrapper
- `ui/inference_ui.py` - May receive updates if developers don't realize it's a wrapper
- `ui/evaluation_viewer.py` - May receive updates if developers don't realize it's a wrapper
- `ui/apps/inference/services/checkpoint_catalog.py` - Still actively maintained and updated

## Recommendations

### Immediate Actions (Low Risk)

1. **Remove obsolete files**:
   - `ui/_visualization/comparison.py` (no dependencies)
   - `ui/test_viewer.py` (test file in production)
   - `ui/apps/unified_ocr_app/backup/app_monolithic_backup_2025-10-21.py` (backup file)

### Short-term Actions (Medium Risk)

2. **Add deprecation notices** to wrapper files:
   - `ui/command_builder.py`
   - `ui/inference_ui.py`
   - `ui/evaluation_viewer.py`
   - Add clear comments and warnings about their wrapper status

3. **Update documentation**:
   - Update `ui/README.md` to clearly indicate which files are wrappers
   - Update all references to point to actual implementations
   - Add migration guide for deprecated utilities

4. **Migrate from deprecated utilities**:
   - Update `scripts/demos/demo_ui.py` to use `ui.utils.command` instead of `ui.utils.command_builder`
   - Remove `ui/utils/command_builder.py` after migration

### Long-term Actions (High Risk)

5. **Migrate from legacy checkpoint catalog**:
   - Create migration plan for `checkpoint_catalog.py`
   - Update all imports to use V2 catalog directly
   - Add deprecation warnings to all public functions
   - Schedule removal after 3-6 month migration period

6. **Consider removing wrapper files** (after documentation update):
   - Once all documentation and scripts reference actual implementations
   - Update `run_ui.py` to call actual implementations directly
   - Remove wrapper files

## Migration Strategy

### Phase 1: Documentation and Warnings (Week 1)
- Add deprecation notices to all wrapper files
- Update `ui/README.md` with clear architecture diagram
- Document which files are wrappers vs. actual implementations

### Phase 2: Remove Obsolete Files (Week 1)
- Remove `ui/_visualization/comparison.py`
- Remove `ui/test_viewer.py`
- Remove `ui/apps/unified_ocr_app/backup/app_monolithic_backup_2025-10-21.py`

### Phase 3: Migrate Utilities (Week 2)
- Update `scripts/demos/demo_ui.py`
- Remove `ui/utils/command_builder.py`

### Phase 4: Migrate Checkpoint Catalog (Months 1-3)
- Update all imports to V2 catalog
- Add deprecation warnings
- Monitor usage
- Remove after migration period

### Phase 5: Remove Wrappers (Months 3-6)
- Update `run_ui.py` to call implementations directly
- Update all documentation
- Remove wrapper files

## Success Criteria

- [ ] All obsolete files removed
- [ ] All wrapper files have clear deprecation notices
- [ ] All documentation updated to reference actual implementations
- [ ] All imports migrated from deprecated utilities
- [ ] Checkpoint catalog migration complete
- [ ] No confusion about which files are source of truth
- [ ] AI agents can clearly identify active vs. legacy files

## Notes

- Wrapper files serve a purpose for backward compatibility but should be clearly marked
- Some files (like `checkpoint_catalog.py`) require careful migration due to active usage
- Test and backup files should never be in production directories
- Documentation is critical to prevent future confusion

## 3. Recommendations

See detailed recommendations in the sections above, organized by priority and risk level.
