---
type: phase_completion_plan
pulse_id: hydra-refactor-2026-01-22
created: 2026-01-25
status: ready_for_next_session
---

# Phase 3/4 Enhanced Implementation Plan

## Current Status

| Metric                | Current | Target | Notes                          |
| --------------------- | ------- | ------ | ------------------------------ |
| Broken Hydra Targets  | 0 ✅     | 0      | Complete - FQN pattern working |
| Broken Python Imports | 51      | 0      | Categorized by priority        |
| Detection Pipeline    | ✅       | ✅      | Verified working               |
| Recognition Pipeline  | ⏳       | ✅      | Config OK, runtime pending     |
| Target Linter         | ✅       | ✅      | Automation in place            |
| Dependency Wrapper    | ✅       | ✅      | Central module created         |

## New Guardrails Implemented

### 1. Hydra Target Linter ✅
**File:** `scripts/audit/hydra_target_linter.py`

**Purpose:** Prevent regression to shallow Hydra target paths

**Usage:**
```bash
# Check for violations
uv run python scripts/audit/hydra_target_linter.py

# Output: 2 violations found (function targets, acceptable)
```

**Current Findings:**
- 2 non-class targets detected (functions, not classes)
- These are acceptable: `recognition_collate_fn`, `create_background_removal_transform`
- All class targets use FQN ✅

### 2. Dependency Check Module ✅
**File:** `ocr/core/utils/dependency_check.py`

**Purpose:** Centralized optional dependency management

**Pattern:**
```python
from ocr.core.utils.dependency_check import safe_import, require_dependency

# Optional dependency
tiktoken = safe_import('tiktoken')
if tiktoken:
    # Use tiktoken

# Required dependency  
boto3 = require_dependency('boto3')  # Raises ImportError if missing
```

**Current Status:**
- tiktoken: Not installed (optional for token counting)
- boto3: Not installed (optional for AWS scripts)
- deep_translator: Not installed (optional for i18n)

## Remaining 51 Import Resolution

### Priority 1: Infrastructure (9 imports)
**Target:** Fix in next session

**Pattern:** Update to use `ocr.core.infrastructure` paths

**Files:**
- ocr_agent.py, coordinator_agent.py, linting_agent.py
- orchestrator imports
- communication layer imports

**Action:** Enhance `batch_fix_imports.py` with infrastructure patterns

### Priority 2: External Dependencies (5 imports)
**Target:** Wrap with safe_import()

**Files:**
- grok_client.py, openai_client.py (tiktoken)
- batch_pseudo_labels_aws.py, prepare_test_dataset.py (boto3)
- translate_readme.py (deep_translator)

**Action:**
```python
# Before
import tiktoken

# After
from ocr.core.utils.dependency_check import safe_import
tiktoken = safe_import('tiktoken')
if tiktoken is None:
    # Fallback or skip functionality
```

### Priority 3: Script Imports (30 imports)
**Target:** Fix after infrastructure

**Categories:**
- demos/ (12): Lower priority, not critical path
- troubleshooting/ (8): Debug scripts
- validation/ (5): One-off analysis
- performance/ (3): Benchmarking
- other scripts/ (2): Utilities

**Action:** Batch update using enhanced fixer script

### Priority 4: UI Modules (5 imports)
**Target:** Defer to UI package refactor

**Strategy:** Lazy import in UI __init__ to prevent blocking training

**Files:**
- ui.apps.inference.services.checkpoint.types
- ui.utils.config_parser
- ui.utils.command
- ui.utils.inference.engine

**Pattern:**
```python
# In UI __init__.py
def __getattr__(name):
    if name == "ConfigParser":
        from .utils.config_parser import ConfigParser
        return ConfigParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

## Execution Sequence (Next Session)

### Step 1: Verify Current State (5 min)
```bash
# Check no regressions
uv run python scripts/audit/hydra_target_linter.py
uv run python scripts/audit/master_audit.py | head -20

# Verify detection pipeline still works
uv run python -c "from ocr.core.models.architecture import OCRModel; print('✅ OCRModel')"
```

### Step 2: Fix Infrastructure Imports (15 min)
```python
# Add to batch_fix_imports.py
IMPORT_FIXES.update({
    # Infrastructure communication
    'from ocr.communication.': 'from ocr.core.infrastructure.communication.',
    'import ocr.communication.': 'import ocr.core.infrastructure.communication.',
    
    # Inference imports
    'from ocr.core.inference.orchestrator': 'from ocr.pipelines.orchestrator',
    'from ocr.core.inference.': 'from ocr.pipelines.inference.',
})

# Run fixes
uv run python scripts/audit/batch_fix_imports.py --infrastructure
```

**Expected:** -9 imports (51 → 42)

### Step 3: Wrap External Dependencies (10 min)
Update 5 files to use `dependency_check.safe_import()`:
- grok_client.py
- openai_client.py
- batch_pseudo_labels_aws.py
- prepare_test_dataset.py
- translate_readme.py

**Expected:** -5 imports (42 → 37)

### Step 4: Test Recognition Pipeline (10 min)
```bash
uv run python runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True
```

**Success Criteria:**
- Config loads ✅
- Model instantiates ✅
- Vocab injection works
- Training starts (1 batch)

### Step 5: Fix Script Imports (20 min)
```bash
# Update batch fixer with script patterns
uv run python scripts/audit/batch_fix_imports.py --scripts --dry-run
uv run python scripts/audit/batch_fix_imports.py --scripts
```

**Expected:** -30 imports (37 → 7)

### Step 6: Final Validation (10 min)
```bash
# Full audit
uv run python scripts/audit/master_audit.py

# Both pipelines
uv run python runners/train.py experiment=det_resnet50_v1 +trainer.fast_dev_run=True
uv run python runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True

# Update tracking
uv run compass pulse-status
```

**Expected:** 7 or fewer imports remaining (all UI, defer-able)

## Success Metrics

### Phase 3 Complete (Hydra Alignment)
- [x] All Hydra targets use FQN ✅
- [x] Linter prevents regression ✅
- [x] Detection pipeline validated ✅
- [ ] Recognition pipeline validated
- [ ] Config validation pre-commit hook

### Phase 4 In Progress (Integration)
- [x] Detection pipeline end-to-end ✅
- [ ] Recognition pipeline end-to-end
- [ ] Infrastructure imports fixed (9 → 0)
- [ ] External deps wrapped (5 → 0)
- [ ] Regression tests added
- [ ] Documentation updated

## Risk Mitigation

| Risk                              | Mitigation                                      |
| --------------------------------- | ----------------------------------------------- |
| Infrastructure fixes break runtime| Test after each batch, rollback if needed       |
| External dep wrapper incomplete   | Use try/except in wrapper, graceful degradation |
| Script fixes introduce new issues | Scripts are non-critical, can defer             |
| Recognition pipeline blocked      | Separate issue from imports, investigate config |
| UI lazy loading causes problems   | Defer to UI package, doesn't block training     |

## Files to Modify (Next Session)

**Automation (1):**
- scripts/audit/batch_fix_imports.py (+30 lines for infrastructure patterns)

**Infrastructure (9):**
- ocr/core/infrastructure/agents/ocr_agent.py
- ocr/core/infrastructure/agents/coordinator_agent.py
- ocr/core/infrastructure/agents/linting_agent.py
- ocr/pipelines/engine.py
- scripts/performance/benchmark_pipeline.py
- (4 more from audit results)

**External Deps (5):**
- ocr/core/infrastructure/agents/llm/grok_client.py
- ocr/core/infrastructure/agents/llm/openai_client.py
- runners/batch_pseudo_labels_aws.py
- scripts/cloud/prepare_test_dataset.py
- scripts/documentation/translate_readme.py

**Scripts (30):**
- scripts/demos/*.py (12 files)
- scripts/troubleshooting/*.py (8 files)
- scripts/validation/*.py (5 files)
- scripts/performance/*.py (3 files)
- scripts/data/*.py (2 files)

## Architectural Decisions Finalized

### 1. Lazy Loading Preserved ✅
- All domain `__init__.py` files use lazy loading
- Prevents circular dependencies
- Required for clean architecture

### 2. FQN for Hydra Targets ✅
- All `_target_` paths include filename
- Bypasses `__init__.py` lazy loading
- Enforced by linter

### 3. Interface-First Base Classes ✅
- All base classes in `ocr.core.interfaces`
- Models: `ocr.core.interfaces.models`
- Losses: `ocr.core.interfaces.losses`

### 4. Centralized Dependency Management ✅
- Optional deps wrapped in `dependency_check`
- Graceful degradation if not installed
- Clear error messages with install instructions

## Next Session Checklist

- [ ] Run hydra_target_linter.py
- [ ] Update batch_fix_imports.py with infrastructure patterns
- [ ] Fix 9 infrastructure imports
- [ ] Wrap 5 external dependencies
- [ ] Test recognition pipeline
- [ ] Fix 30 script imports
- [ ] Run final audit
- [ ] Update tracking docs
- [ ] Document patterns in AgentQMS

**Estimated Time:** 70 minutes
**Expected Outcome:** 7 or fewer imports remaining, both pipelines working
