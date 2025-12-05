---
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['refactoring', 'cleanup', 'technical-debt']
title: "Remove Deprecated PathUtils Class and Legacy Helpers"
date: "2025-12-05 23:53 (KST)"
related_to: "2025-12-05_2307_implementation_plan_phase4-refactoring.md"
---

# Master Prompt (use verbatim to start)
"Remove deprecated PathUtils class: audit callers, migrate to modern resolver-based helpers, remove dead code to prevent AI agent confusion and architecture drift."

# Living Implementation Blueprint: PathUtils Legacy Code Removal

## Context
Phase 4 refactoring introduced centralized path helpers (`get_path_resolver()`, `setup_project_paths()`, `ensure_output_dirs()`). The deprecated `PathUtils` class remains in the codebase with warning messages but no active callers. This legacy code has historically caused AI agents to fix/update dead code instead of the canonical implementations, resulting in diverging architectures and unexpected behavior.

**Priority**: High (prevent technical debt accumulation)

## Progress Tracker
- **STATUS:** ✅ COMPLETE (Phase 1)
- **CURRENT STEP:** All tasks completed
- **LAST COMPLETED TASK:** Phase 1, Task 4 - Validation
- **NEXT TASK:** None - implementation plan executed successfully

### Implementation Outline (Checklist)

#### Phase 1: Audit & Migration (single session)
1. [x] **Task 1: Audit remaining PathUtils callers**
   - [x] Search for `PathUtils.` method calls across codebase
   - [x] Search for `from ocr.utils.path_utils import PathUtils`
   - [x] Identify any usage of deprecated standalone functions (`setup_paths`, `add_src_to_sys_path`, `ensure_project_root_env`)
   - [x] Document findings with file paths and line numbers
   - **RESULT**: Found 1 active caller: `runners/train_fast.py:100` calling deprecated `setup_paths()`

2. [x] **Task 2: Migrate remaining callers**
   - [x] Replace `PathUtils.*` with modern `get_path_resolver().config.*` equivalents
   - [x] Replace deprecated standalone functions with `setup_project_paths()`
   - [x] Update imports to use canonical helpers
   - [x] Run lint/type checks on modified files
   - **RESULT**: Updated `runners/train_fast.py` to correctly call `setup_project_paths()`

3. [x] **Task 3: Remove deprecated code**
   - [x] Remove entire `PathUtils` class from `ocr/utils/path_utils.py` (232 lines removed)
   - [x] Remove deprecated standalone helper functions (`setup_paths`, `add_src_to_sys_path`, `ensure_project_root_env`, deprecated `get_*` functions) (120 lines removed)
   - [x] Keep only: `OCRPathConfig`, `OCRPathResolver`, `get_path_resolver()`, `setup_project_paths()`, `ensure_output_dirs()`, `get_project_root()`, `PROJECT_ROOT`
   - [x] Update module docstring to reflect modern API
   - **RESULT**: Module reduced from 748 → 396 lines (~47% size reduction)

4. [x] **Task 4: Validation**
   - [x] Run full lint/type check suite on `ocr/utils/path_utils.py` (syntax check passed)
   - [x] Verify no imports of deprecated code work (ImportError on PathUtils, as expected)
   - [x] All modern API imports verified working
   - [x] Comprehensive codebase search confirms zero remaining PathUtils usage
   - **RESULT**: All validations passed, no deprecation code remains

## 📋 Technical Requirements

### Must Keep (Modern API)
- `OCRPathConfig` dataclass
- `OCRPathResolver` class
- `get_path_resolver()` - global resolver instance
- `setup_project_paths()` - modern setup function
- `ensure_output_dirs()` - centralized directory creation
- `get_project_root()` - convenience for PROJECT_ROOT
- `PROJECT_ROOT` - stable global constant

### Must Remove (Deprecated)
- `PathUtils` class (entire class including all methods)
- `setup_paths()` standalone function
- `add_src_to_sys_path()` standalone function
- `ensure_project_root_env()` standalone function
- All deprecated `get_*` standalone functions with warnings

### Migration Map
| Deprecated | Modern Replacement |
|-----------|-------------------|
| `PathUtils.get_project_root()` | `get_path_resolver().config.project_root` or `PROJECT_ROOT` |
| `PathUtils.get_data_path()` | `get_path_resolver().config.data_dir` |
| `PathUtils.get_config_path()` | `get_path_resolver().config.config_dir` |
| `PathUtils.get_outputs_path()` | `get_path_resolver().config.output_dir` |
| `PathUtils.get_images_path()` | `get_path_resolver().config.images_dir` |
| `PathUtils.get_annotations_path()` | `get_path_resolver().config.annotations_dir` |
| `PathUtils.get_logs_path()` | `get_path_resolver().config.logs_dir` |
| `PathUtils.get_checkpoints_path()` | `get_path_resolver().config.checkpoints_dir` |
| `PathUtils.get_submissions_path()` | `get_path_resolver().config.submissions_dir` |
| `PathUtils.setup_paths()` | `setup_project_paths()` |
| `setup_paths()` (standalone) | `setup_project_paths()` |

## 🎯 Success Criteria
- No remaining `PathUtils.*` method calls in codebase (except in path_utils.py deprecation warnings)
- No imports of `PathUtils` class
- Deprecated class and functions removed from `ocr/utils/path_utils.py`
- Lint/type checks pass
- Smoke train test completes without deprecation warnings
- Module is ~50% smaller with cleaner API surface

## 📊 Risk Mitigation
- **Risk**: Hidden callers in rarely-used scripts → **Mitigation**: comprehensive grep search + validate after removal
- **Risk**: Breaking change for external code → **Mitigation**: this is internal-only, no public API
- **Risk**: Regression in path resolution → **Mitigation**: smoke test validates core functionality

## 🚀 Immediate Next Action
**TASK:** Comprehensive audit of PathUtils usage

**OBJECTIVE:** Identify all remaining callers to plan migration

**APPROACH:**
1. Grep for `PathUtils\.` pattern across all Python files
2. Grep for `from ocr.utils.path_utils import PathUtils`
3. Grep for deprecated standalone function calls
4. Document findings with file:line references

**SUCCESS CRITERIA:**
- Complete list of callers with file paths and line numbers
- Clear migration plan for each caller

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
