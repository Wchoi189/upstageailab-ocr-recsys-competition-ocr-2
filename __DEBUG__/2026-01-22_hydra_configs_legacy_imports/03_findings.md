# Findings and Solutions: Debugging Session Results

**Date**: 2026-01-22
**Session**: Hydra configs and legacy imports debugging
**Status**: ✅ Root causes identified, solutions documented

---

## Executive Summary

Successfully identified and documented solutions for two critical issues blocking the OCR training pipeline:

1. **Ghost Code Phenomenon** → Solution: Editable install + runtime assertions
2. **Hydra Recursive Instantiation** → Solution: `_recursive_=False` in factory calls

**Additional Outcome**: Created systematic tooling and workflows for future refactoring efforts.

---

## Critical Findings

### Finding 1: Ghost Code Phenomenon

**Severity**: CRITICAL
**Impact**: All code modifications ignored by runtime
**Status**: ✅ Root cause identified, solution documented

#### Problem Description

The runtime environment was loading the `ocr` package from `site-packages` instead of the workspace directory, making all code changes invisible to the execution environment.

#### Evidence

1. **Print statements not appearing**:
   ```python
   # Added to code
   print("DEBUG: This should appear")
   # Not visible in logs
   ```

2. **Runtime errors not triggering**:
   ```python
   # Added to code
   raise RuntimeError("This should crash")
   # Execution continues normally
   ```

3. **File renames not causing failures**:
   ```bash
   mv architecture.py architecture_bak.py
   # Import still works - proves code loaded from elsewhere
   ```

4. **Stack traces showing phantom code**:
   - Traces reference line numbers that don't exist in workspace files
   - Code content doesn't match what's on disk

#### Root Cause

Package installed in **standard mode** (`pip install .`) instead of **editable mode** (`pip install -e .`).

**Mechanism**:
- Standard install copies code to `site-packages`
- Python imports from `site-packages` by default
- Workspace changes don't affect installed copy
- Runtime and development environments diverge

#### Solution

**Immediate Fix**:
```bash
# Reinstall in editable mode
pip install -e .
# or with uv
uv pip install -e .
```

**Preventive Measures**:

1. **Runtime Assertion** (add to `runners/train.py`):
   ```python
   import ocr
   import sys

   assert "site-packages" not in ocr.__file__, (
       "🛑 STOP: You are editing code but running an installed package copy!\n"
       f"   Current location: {ocr.__file__}\n"
       "   Fix: uv pip install -e ."
   )
   ```

2. **Pre-Flight Validation** (`scripts/audit/preflight.sh`):
   ```bash
   #!/bin/bash
   PYTHON_ORIGIN=$(uv run python -c "import ocr; print(ocr.__file__)")

   if [[ "$PYTHON_ORIGIN" == *"site-packages"* ]]; then
       echo "❌ ERROR: Ghost Code Detected"
       echo "Fixing: Re-installing in editable mode..."
       uv pip install -e .
   fi
   ```

3. **Documentation Update**:
   - Add to setup instructions
   - Include in troubleshooting guide
   - Create onboarding checklist

---

### Finding 2: Hydra Recursive Instantiation Trap

**Severity**: HIGH
**Impact**: Recognition pipeline completely blocked
**Status**: ✅ Root cause identified, solution implemented

#### Problem Description

Hydra's default `_recursive_=True` behavior conflicts with factory patterns that expect raw configuration objects for late-binding instantiation.

#### Evidence

**Error Message**:
```
Error in call to target 'torch.optim.adam.Adam':
TypeError("Adam.__init__() missing 1 required positional argument: 'params'")
full_key: model.architectures.cfg.optimizer
```

**Execution Flow**:
```python
# Code expects this:
get_model_by_cfg(cfg)
  → Create model
  → Get model.parameters()
  → Instantiate optimizer with params

# Hydra does this (with _recursive_=True):
hydra.utils.instantiate(architectures, cfg=config)
  → Sees cfg.optimizer
  → Tries to instantiate optimizer immediately
  → Fails: no model.parameters() yet
```

#### Root Cause

**Design Conflict**:
- **Codebase**: Factory pattern with late-binding
- **Hydra**: Eager instantiation by default

**Specific Issue**:
- `PARSeq` architecture expects `cfg` as a dict
- Uses `cfg.optimizer` to create optimizer AFTER model is built
- Hydra's `_recursive_=True` tries to instantiate optimizer BEFORE passing cfg to PARSeq
- `torch.optim.Adam` requires `params` at init, but model doesn't exist yet

#### Solution

**Immediate Fix**:
```python
# In ocr/core/models/__init__.py
def get_model_by_cfg(config: DictConfig) -> nn.Module:
    architectures = config.model.architectures

    # CRITICAL: Disable recursive instantiation for factories
    model = hydra.utils.instantiate(
        architectures,
        cfg=config,
        _recursive_=False  # ← This prevents premature optimizer instantiation
    )

    return model
```

**Preventive Measures**:

1. **AST-Grep Lint Rule** (`rules/hydra-recursion.yaml`):
   ```yaml
   id: missing-recursive-false
   language: python
   rule:
     pattern: hydra.utils.instantiate($CONF, $$$)
     not:
       pattern: hydra.utils.instantiate($CONF, _recursive_=False, $$$)
   message: "Potential Recursive Instantiation Trap! Ensure _recursive_=False for model factories."
   severity: warning
   ```

2. **Code Review Checklist**:
   - [ ] All factory calls use `_recursive_=False`
   - [ ] Late-binding patterns documented
   - [ ] Hydra behavior explicitly controlled

3. **Documentation**:
   - Add to Hydra standards
   - Include in architecture guidelines
   - Create troubleshooting entry

---

### Finding 3: Systematic Import Breakage

**Severity**: HIGH
**Impact**: 53 broken imports across codebase
**Status**: ⏳ Tooling created, systematic fixes in progress

#### Problem Description

Large-scale refactoring (domain separation, module reorganization) left numerous broken import paths throughout the codebase.

#### Statistics

**Broken Imports by Category**:
- Training pipeline: 4 critical imports
- Utilities: 8 imports
- Inference pipeline: 6 imports
- External dependencies: 3 imports (not installed)
- Scripts/demos: 32 imports (non-critical)

**Broken Hydra Targets**:
- Initial: 18 targets
- After cleanup: 13 targets
- Primary config: `configs/data/datasets/craft.yaml`

#### Root Cause

**Incomplete Refactoring**:
- Modules moved without updating import statements
- Hydra configs not updated after module reorganization
- No systematic validation during refactoring

**Specific Issues**:
1. **Import path changes**:
   - `ocr.core.metrics` → `ocr.domains.detection.metrics`
   - `ocr.detection.*` → `ocr.domains.detection.*`

2. **Module relocations**:
   - Evaluation modules moved to domain-specific locations
   - Utility modules reorganized
   - Lightning modules restructured

3. **Interpolation variables**:
   - `${data.dataset_path}` not defined
   - Inconsistent naming (`${dataset_path}` vs `${data.dataset_path}`)

#### Solution

**Systematic Approach**:

1. **Audit** (`scripts/audit/master_audit.py`):
   ```bash
   python scripts/audit/master_audit.py > audit_results.txt
   ```

2. **Auto-Heal** (`scripts/audit/auto_align_hydra.py`):
   ```bash
   # Dry run first
   python scripts/audit/auto_align_hydra.py --dry-run

   # Apply fixes
   python scripts/audit/auto_align_hydra.py
   ```

3. **Manual Fixes** (for complex cases):
   - Use ADT `intelligent-search` to find new locations
   - Use `yq` for surgical YAML updates
   - Use `ast-grep` for structural code changes

4. **Verification**:
   ```bash
   # Re-run audit
   python scripts/audit/master_audit.py

   # Should show reduced count
   ```

**Tooling Created**:
- Migration guard script
- Auto-alignment script
- Fix manifest generator
- Pre-flight validation

---

## Solutions Implemented

### 1. Environment Validation

**Script**: `scripts/audit/migration_guard.py`

**Features**:
- Checks editable install status
- Validates Hydra targets
- Checks for recursive instantiation traps
- Generates actionable error messages

**Usage**:
```bash
python scripts/audit/migration_guard.py
```

### 2. Automated Hydra Healing

**Script**: `scripts/audit/auto_align_hydra.py`

**Features**:
- Uses runtime reflection to find true module locations
- Updates YAML configs automatically
- Supports dry-run mode
- Batch processing with verification

**Usage**:
```bash
# See what would change
python scripts/audit/auto_align_hydra.py --dry-run

# Apply fixes
python scripts/audit/auto_align_hydra.py
```

### 3. Systematic Audit

**Script**: `scripts/audit/master_audit.py`

**Features**:
- Scans all Python files for broken imports
- Scans all YAML configs for broken targets
- Generates structured kill list
- Tracks progress across iterations

**Usage**:
```bash
python scripts/audit/master_audit.py > audit_results.txt
```

---

## Testing Recommendations

### Detection Pipeline Testing

**Priority**: HIGH (primary domain)

**Test Command**:
```bash
python runners/train.py experiment=det_resnet50_v1 +trainer.fast_dev_run=True
```

**Success Criteria**:
- ✅ No import errors
- ✅ Model builds successfully
- ✅ Dataset loads
- ✅ Training loop starts
- ✅ At least 1 batch completes

**Known Issues**:
- Maps caching warning (non-critical)
- Some utility imports still broken (non-blocking for training)

### Recognition Pipeline Testing

**Priority**: MEDIUM (new implementation, single epoch completed)

**Test Command**:
```bash
python runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True
```

**Success Criteria**:
- ✅ No Hydra instantiation errors
- ✅ Vocab size injection works
- ✅ Model builds successfully
- ✅ Training loop starts

**Known Issues**:
- Only completed single epoch (needs more validation)
- Newer implementation (less battle-tested)

---

## Artifact Outputs

### Analysis Documents

**Location**: `artifacts/analysis_outputs/`

- `debugging_pain_points.md` - Detailed pain point analysis
- `broken_targets.json` - Structured list of broken Hydra targets
- `hydra_interpolation_map.md` - Interpolation variable mapping
- `master_audit.md` - Audit methodology
- `backward_compatibility_shims_technical_assessment.md` - Shim antipatterns

### Implementation Guides

**Location**: `artifacts/implementation_guides/`

- `migration_guard_implementation.md` - Pre-execution validation
- `auto_align_hydra_script.md` - Automated Hydra fixing

### Tool Guides

**Location**: `artifacts/tool_guides/`

- `yq_mastery_guide.md` - Advanced YAML manipulation
- `adt_usage_patterns.md` - AST-Grep and ADT usage

### AI Guidance

**Location**: `artifacts/ai_guidance/`

- `instruction_patterns.md` - AI agent instruction strategies

### Refactoring Patterns

**Location**: `artifacts/refactoring_patterns/`

- `shim_antipatterns_guide.md` - Why shims become toxic

---

## Recommendations

### Immediate Actions

1. **Fix environment**:
   ```bash
   uv pip install -e .
   ```

2. **Run pre-flight check**:
   ```bash
   bash scripts/audit/preflight.sh
   ```

3. **Test detection pipeline**:
   ```bash
   python runners/train.py experiment=det_resnet50_v1 +trainer.fast_dev_run=True
   ```

### Short-Term Actions

1. **Systematic import fixes**:
   - Process broken imports in batches
   - Use auto-alignment for Hydra configs
   - Verify after each batch

2. **Add safeguards**:
   - Runtime assertions in entry points
   - AST-grep lint rules
   - Pre-commit hooks

3. **Update documentation**:
   - Setup instructions
   - Troubleshooting guide
   - Hydra standards

### Long-Term Actions

1. **Prevent future issues**:
   - CI/CD integration for validation
   - Automated refactoring workflows
   - Regular audit runs

2. **Improve tooling**:
   - Enhance auto-alignment script
   - Add more lint rules
   - Create refactoring templates

3. **Knowledge transfer**:
   - Document patterns
   - Create training materials
   - Share lessons learned

---

## Success Metrics

| Metric                 | Before | After | Target |
| ---------------------- | ------ | ----- | ------ |
| Broken Python imports  | 53     | 49    | 0      |
| Broken Hydra targets   | 18     | 13    | 0      |
| Detection pipeline     | ❌      | ⏳     | ✅      |
| Recognition pipeline   | ❌      | ⏳     | ✅      |
| Environment validation | ❌      | ✅     | ✅      |
| Automated tooling      | ❌      | ✅     | ✅      |

---

## References

- Initial analysis: `01_initial_analysis.md`
- Investigation process: `02_investigation.md`
- Session README: `README.md`
- Artifacts directory: `artifacts/`
- Logs directory: `logs/`
