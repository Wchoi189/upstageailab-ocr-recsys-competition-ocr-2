---
title: "Path Utils Migration Feasibility Assessment and Implementation Plan"
author: "ai-agent"
date: "2025-11-09"
status: "draft"
tags: []
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

## 2. Assessment

## 3. Recommendations
## Executive Summary

The project currently has **308 instances** of brittle path resolution code across **76 files**, using inconsistent patterns like `Path(__file__).parent.parent`, `sys.path.insert()`, and hardcoded `PROJECT_ROOT` variables. The `ocr/utils/path_utils.py` module provides a centralized `OCRPathResolver` solution, but it's only used in **2 scripts** (`process_manager.py`, `cache_manager.py`).

**Feasibility Assessment: HIGHLY FEASIBLE**

The migration to `path_utils` is highly feasible because:
1. ✅ `path_utils` already exists and is functional
2. ✅ It handles project root detection automatically
3. ✅ It provides a consistent API for all path operations
4. ✅ Some scripts already use it successfully
5. ⚠️  Requires careful handling of initial Path import for path_utils itself

## Current State Analysis

### Brittle Path Patterns Found

**Pattern 1: Manual Path Calculation (Most Common)**
```python
PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root = Path(__file__).resolve().parents[3]
```
**Found in:** 38+ files in `scripts/` directory

**Pattern 2: Manual sys.path Manipulation**
```python
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```
**Found in:** 30+ files

**Pattern 3: Hardcoded Relative Paths**
```python
config_path = project_root / "configs" / "tools" / "seroost_config.json"
ui_path = project_root / "ui" / "command_builder.py"
```
**Found in:** 20+ files

**Pattern 4: Bootstrap Pattern (Agent Tools)**
```python
def _load_bootstrap():
    current_dir = Path(__file__).resolve().parent
    for directory in (current_dir, *tuple(current_dir.parents)):
        candidate = directory / "_bootstrap.py"
        if candidate.exists():
            # Load bootstrap module
```
**Found in:** 15+ files in `agent_tools/`

### Current path_utils Usage

**Already Using path_utils:**
- `scripts/process_manager.py` - Uses `setup_project_paths()` and `get_path_resolver()`
- `scripts/cache_manager.py` - Uses `setup_project_paths()` (but still has manual sys.path code)
- `scripts/agent_tools/utilities/get_context.py` - Uses bootstrap pattern that calls `setup_project_paths()`

**path_utils API Available:**
```python
from ocr.utils.path_utils import (
    setup_project_paths,      # Setup sys.path and environment
    get_path_resolver,         # Get global resolver instance
    OCRPathResolver,            # Resolver class
    OCRPathConfig,              # Config dataclass
)
```

## Critical Consideration: Path Import Order

**Issue:** `path_utils.py` itself uses `from pathlib import Path`, so scripts need to import `Path` first before importing `path_utils`.

**Solution Pattern:**
```python
from pathlib import Path  # Required for path_utils to work
from ocr.utils.path_utils import setup_project_paths, get_path_resolver

# Setup paths (handles sys.path and environment)
setup_project_paths()

# Get resolver for path operations
resolver = get_path_resolver()
project_root = resolver.config.project_root
```

## Migration Strategy

### Phase 1: Enhance path_utils (1-2 hours)

**Action 1.1: Add convenience function for scripts**
Add a simple `setup_paths()` function that handles everything:
```python
def setup_paths() -> OCRPathResolver:
    """One-line setup for scripts. Handles sys.path and returns resolver."""
    resolver = setup_project_paths()
    return resolver
```

**Action 1.2: Add script-specific helpers**
Add helper methods for common script operations:
```python
def get_script_project_root(script_path: Path | str) -> Path:
    """Get project root from script location."""
    # Implementation using path_utils detection
```

**Action 1.3: Document bootstrap pattern**
If bootstrap pattern is needed for agent_tools, document it or create a helper.

### Phase 2: Create Migration Script (2-3 hours)

**Action 2.1: Create automated migration tool**
Script to:
- Find all brittle path patterns
- Replace with path_utils equivalents
- Add necessary imports
- Test that imports work

**Action 2.2: Manual review required**
Some patterns may need manual adjustment:
- Bootstrap pattern in agent_tools
- Special cases with Hydra config paths
- Shell scripts (bash path resolution)

### Phase 3: Migrate Scripts by Category (8-10 hours)

**Category 1: Simple Scripts (30 files) - 3 hours**
Scripts with simple `Path(__file__).parent.parent` patterns:
- Replace with `setup_project_paths()` + `get_path_resolver()`
- Remove manual sys.path code
- Update path references

**Category 2: Agent Tools Scripts (15 files) - 3 hours**
Scripts using bootstrap pattern:
- Option A: Replace bootstrap with direct `setup_project_paths()` call
- Option B: Keep bootstrap but make it use path_utils internally
- Update all path references

**Category 3: Complex Scripts (10 files) - 2 hours**
Scripts with multiple path operations:
- Refactor to use resolver methods
- Update all hardcoded paths
- Test thoroughly

**Category 4: Runners and UI (5 files) - 2 hours**
- `runners/train.py`, `runners/test.py`, `runners/predict.py`
- `run_ui.py`
- Update Hydra config paths if needed

### Phase 4: Update Documentation (1 hour)

**Action 4.1: Update system.md**
Add path_utils usage instructions to `docs/agents/system.md`

**Action 4.2: Create migration guide**
Document the migration process and patterns

**Action 4.3: Update code examples**
Update all documentation examples to use path_utils

### Phase 5: Testing and Validation (2-3 hours)

**Action 5.1: Test all migrated scripts**
- Run each script to verify it works
- Check that imports resolve correctly
- Verify paths are correct

**Action 5.2: Create validation script**
Script to check for remaining brittle patterns:
```python
# Check for remaining Path(__file__).parent patterns
# Check for remaining sys.path.insert patterns
# Verify all scripts use path_utils
```

## Implementation Plan

### Step 1: Enhance path_utils (Priority: High)

**File:** `ocr/utils/path_utils.py`

**Changes:**
1. Add `setup_paths()` convenience function
2. Add `get_script_project_root()` helper
3. Improve documentation with examples
4. Add type hints where missing

**Code:**
```python
def setup_paths() -> OCRPathResolver:
    """
    One-line setup for scripts.
    
    Usage:
        from pathlib import Path
        from ocr.utils.path_utils import setup_paths
        
        resolver = setup_paths()
        project_root = resolver.config.project_root
    """
    return setup_project_paths()
```

### Step 2: Create Migration Template

**Standard Pattern for All Scripts:**
```python
#!/usr/bin/env python3
"""Script description."""

from pathlib import Path  # Required before path_utils import
from ocr.utils.path_utils import setup_paths, get_path_resolver

# Setup paths (one line)
resolver = setup_paths()
project_root = resolver.config.project_root

# Rest of script uses resolver.config.* for paths
```

### Step 3: Migration by File Type

**Type A: Simple Scripts**
**Before:**
```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

**After:**
```python
from pathlib import Path
from ocr.utils.path_utils import setup_paths

resolver = setup_paths()
project_root = resolver.config.project_root
```

**Type B: Scripts with Path Operations**
**Before:**
```python
project_root = Path(__file__).parent.parent
config_path = project_root / "configs" / "tools" / "config.yaml"
```

**After:**
```python
from pathlib import Path
from ocr.utils.path_utils import setup_paths, get_path_resolver

resolver = setup_paths()
config_path = resolver.config.config_dir / "tools" / "config.yaml"
```

**Type C: Agent Tools (Bootstrap Pattern)**
**Option 1: Replace Bootstrap**
**Before:**
```python
def _load_bootstrap():
    # Complex bootstrap logic
    return bootstrap_module
```

**After:**
```python
from pathlib import Path
from ocr.utils.path_utils import setup_paths

resolver = setup_paths()
```

**Option 2: Keep Bootstrap, Use path_utils**
Modify bootstrap to use path_utils internally.

### Step 4: Special Cases

**Case 1: Hydra Config Paths**
Hydra requires relative paths from script location or absolute paths.
**Solution:** Use resolver to get absolute config path:
```python
resolver = setup_paths()
config_path = resolver.config.config_dir

@hydra.main(config_path=str(config_path), config_name="train")
def train(config):
    pass
```

**Case 2: Shell Scripts**
Bash scripts need different approach:
```bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
```
Keep as-is or create Python wrapper.

**Case 3: Test Files**
Tests may need special handling:
```python
# In conftest.py or test setup
from ocr.utils.path_utils import setup_paths
setup_paths()
```

## Migration Checklist

### Pre-Migration
- [ ] Backup current codebase
- [ ] Enhance path_utils with convenience functions
- [ ] Create migration script/tool
- [ ] Document migration patterns

### Migration
- [ ] Migrate simple scripts (Category 1)
- [ ] Migrate agent_tools scripts (Category 2)
- [ ] Migrate complex scripts (Category 3)
- [ ] Migrate runners and UI (Category 4)
- [ ] Update all documentation

### Post-Migration
- [ ] Test all migrated scripts
- [ ] Run validation script to check for remaining patterns
- [ ] Update CI/CD if needed
- [ ] Create migration summary report

## Benefits

1. **Consistency**: All scripts use same path resolution mechanism
2. **Maintainability**: Single source of truth for paths
3. **Reliability**: Automatic project root detection
4. **Flexibility**: Easy to override paths via environment variables
5. **Testability**: Easier to test with different path configurations

## Risks and Mitigation

**Risk 1: Import Order Issues**
- **Mitigation**: Always import `Path` before `path_utils`, document clearly

**Risk 2: Breaking Changes**
- **Mitigation**: Test each script after migration, keep backup

**Risk 3: Special Cases Not Covered**
- **Mitigation**: Document special cases, create helpers for common patterns

**Risk 4: Performance Impact**
- **Mitigation**: path_utils uses caching, minimal performance impact

## Success Criteria

- [ ] Zero instances of `Path(__file__).parent.parent` in scripts
- [ ] Zero instances of manual `sys.path.insert()` in scripts
- [ ] All scripts use `setup_paths()` or `setup_project_paths()`
- [ ] All path operations use `resolver.config.*` or `resolver.resolve_relative_path()`
- [ ] All scripts tested and working
- [ ] Documentation updated

## Estimated Timeline

- **Phase 1 (Enhance path_utils):** 1-2 hours
- **Phase 2 (Migration tool):** 2-3 hours
- **Phase 3 (Migrate scripts):** 8-10 hours
- **Phase 4 (Documentation):** 1 hour
- **Phase 5 (Testing):** 2-3 hours

**Total: 14-19 hours**

## Next Steps

1. Review and approve this plan
2. Enhance path_utils with convenience functions
3. Create migration script/tool
4. Begin migration with simple scripts (Category 1)
5. Progressively migrate more complex scripts
6. Test and validate all changes
