# Investigation Process: Systematic Debugging Approach

**Date**: 2026-01-22
**Session**: Hydra configs and legacy imports debugging
**Focus**: Detection pipeline (primary), Recognition pipeline (secondary)

---

## Investigation Strategy

### Phase 1: Environment Validation

**Objective**: Confirm runtime environment state

**Actions**:
1. Attempted to run detection training with fast_dev_run
2. Observed import failures
3. Tested code modifications (print statements, runtime errors)
4. Discovered "ghost code" phenomenon

**Key Discovery**: Package installed in standard mode, not editable

### Phase 2: Systematic Audit

**Objective**: Quantify the scope of broken imports and configs

**Tool**: `scripts/audit/master_audit.py`

**Methodology**:
- Scan all Python files for broken imports
- Scan all YAML configs for broken Hydra targets
- Check for known problematic paths
- Generate structured kill list

**Results**:
- Initial scan: 53 broken imports, 18 broken Hydra targets
- After cleanup: 53 broken imports, 13 broken Hydra targets
- Pattern identified: Interpolation variables not resolving

### Phase 3: Root Cause Analysis

**Objective**: Identify underlying causes, not just symptoms

**Approach**:
1. Trace import failure chains
2. Analyze Hydra instantiation behavior
3. Compare workspace code vs. runtime code
4. Test hypotheses with targeted fixes

**Findings**:
1. **Ghost Code**: Non-editable install causing code/runtime mismatch
2. **Recursive Instantiation**: Hydra default behavior conflicting with factory pattern
3. **Incomplete Refactoring**: Module reorganization left broken paths

---

## Debugging Techniques Applied

### 1. Import Chain Tracing

**Method**: Follow stack traces to identify cascading failures

**Example**:
```
orchestrator.py:101
  → DetectionPLModule import
    → module.py:11
      → WandbProblemLogger import
        → wandb_loggers.py:13
          → CLEvalMetric import
            → cleval_metric.py:15
              → logging utils import (FAILS)
```

**Insight**: Single broken import at the end of chain blocks entire pipeline

### 2. Code Modification Testing

**Method**: Add observable changes to verify code execution

**Tests Performed**:
- Added print statements → Not visible in output
- Added raise RuntimeError → Not triggered
- Renamed critical files → Imports still work

**Conclusion**: Runtime not using workspace code

### 3. Audit-Driven Discovery

**Method**: Automated scanning to find all issues systematically

**Audit Cycles**:
1. **First run**: Baseline (53 imports, 18 targets)
2. **After config removal**: (53 imports, 13 targets)
3. **After import fixes**: (49 imports, 13 targets)

**Pattern**: Hydra targets more persistent than import errors

### 4. Hypothesis Testing

**Hypothesis 1: Ghost Code**
- Test: Rename `architecture.py` → `architecture_bak.py`
- Expected: ModuleNotFoundError
- Actual: No error
- Conclusion: CONFIRMED - Code running from site-packages

**Hypothesis 2: Recursive Instantiation**
- Test: Add `_recursive_=False` to instantiate call
- Expected: Optimizer error disappears
- Actual: Error disappears
- Conclusion: CONFIRMED - Hydra instantiating too early

**Hypothesis 3: Refactoring Incomplete**
- Test: Run master audit
- Expected: Multiple broken imports
- Actual: 53 broken imports found
- Conclusion: CONFIRMED - Systematic cleanup needed

---

## Tools and Scripts Used

### Master Audit Script

**Location**: `scripts/audit/master_audit.py`

**Capabilities**:
- Scans Python files for broken imports
- Scans YAML configs for broken Hydra targets
- Checks for known problematic paths
- Generates structured report

**Usage**:
```bash
python scripts/audit/master_audit.py
```

**Output Format**:
```
🚨 BROKEN IMPORTS (53):
  [File] path/to/file.py:line
    --> Import: module.path ['Symbol']
    --> Error: Module or attribute not found

🚨 BROKEN HYDRA TARGETS (13):
  [Config] path/to/config.yaml
    --> Key: config.key.path
    --> Target: ${interpolation}.Class
    --> Error: Module not found
```

### Pre-Flight Check (Planned)

**Purpose**: Validate environment before execution

**Components**:
1. Check editable install status
2. Validate Hydra targets
3. Check for recursive instantiation traps
4. Verify critical imports

**Reference**: `artifacts/implementation_guides/migration_guard_implementation.md`

---

## Iterative Debugging Process

### Iteration 1: Initial Discovery

**Action**: Run detection training
**Result**: ModuleNotFoundError for logging utils
**Response**: Check import path, discover it doesn't exist
**Learning**: Import paths need systematic audit

### Iteration 2: First Fix Attempt

**Action**: Fix logging import path
**Result**: New ModuleNotFoundError for metrics utils
**Response**: Fix metrics import
**Learning**: Cascading failures require systematic approach

### Iteration 3: Systematic Audit

**Action**: Run master_audit.py
**Result**: 53 broken imports, 18 broken targets
**Response**: Realize scope is too large for manual fixes
**Learning**: Need automated healing approach

### Iteration 4: Ghost Code Discovery

**Action**: Add print statements to debug
**Result**: Print statements don't appear
**Response**: Test with file rename
**Learning**: Runtime using different code than workspace

### Iteration 5: Recognition Pipeline Test

**Action**: Try recognition training
**Result**: Optimizer instantiation error
**Response**: Analyze Hydra behavior
**Learning**: Recursive instantiation conflicting with design

---

## Pain Points Encountered

### 1. Ghost Code Phenomenon (Severity: CRITICAL)

**Impact**: All code changes ignored by runtime

**Symptoms**:
- Print statements invisible
- Runtime errors not triggered
- File renames not causing failures
- Stack traces showing non-existent code

**Root Cause**: Standard install instead of editable install

**Resolution**: Reinstall with `pip install -e .` (or `uv pip install -e .`)

### 2. Hydra Recursive Instantiation (Severity: HIGH)

**Impact**: Recognition pipeline completely blocked

**Symptoms**:
- `TypeError: Adam.__init__() missing 'params'`
- Error occurs before model.__init__ is called
- No amount of model code changes fix it

**Root Cause**: Hydra defaults to `_recursive_=True`, instantiates optimizer before model exists

**Resolution**: Set `_recursive_=False` in factory calls

### 3. Cascading Import Failures (Severity: HIGH)

**Impact**: Single broken import blocks entire domain

**Symptoms**:
- Long import chains
- Error at end of chain blocks everything
- Difficult to identify root cause

**Root Cause**: Incomplete refactoring, modules moved without updating imports

**Resolution**: Systematic import alignment using audit + ADT

### 4. Interpolation Variable Resolution (Severity: MEDIUM)

**Impact**: 13 Hydra targets broken

**Symptoms**:
- `${data.dataset_path}` not resolving
- Module not found errors for interpolated paths
- Inconsistent variable naming (`${dataset_path}` vs `${data.dataset_path}`)

**Root Cause**: Hydra interpolation variables not defined or misnamed

**Resolution**: Define variables in base configs or use absolute paths

---

## Lessons Learned

### 1. Environment Hygiene is Critical

**Lesson**: Always use editable installs for development

**Implementation**:
- Add runtime assertion in train.py
- Create pre-flight validation script
- Document in setup instructions

### 2. Hydra Requires Explicit Control

**Lesson**: Factory patterns need `_recursive_=False`

**Implementation**:
- Add to all factory instantiation calls
- Create AST-grep lint rule to catch violations
- Document in Hydra standards

### 3. Refactoring Needs Systematic Validation

**Lesson**: Manual fixes don't scale to 50+ broken imports

**Implementation**:
- Use automated audit tools
- Generate fix manifests
- Apply fixes in batches with verification

### 4. Debugging Needs Observable Signals

**Lesson**: Silent failures waste time

**Implementation**:
- Add logging at critical points
- Use assertions to fail fast
- Create validation layers instead of shims

---

## Debugging Workflow Recommendations

### For Future Sessions

1. **Start with validation**:
   ```bash
   python scripts/audit/migration_guard.py
   ```

2. **Run systematic audit**:
   ```bash
   python scripts/audit/master_audit.py > audit_results.txt
   ```

3. **Generate fix manifest**:
   ```bash
   python scripts/audit/generate_fix_manifest.py audit_results.txt
   ```

4. **Apply fixes in batches**:
   - Process 10 items at a time
   - Verify after each batch
   - Re-run audit to confirm progress

5. **Validate environment**:
   ```bash
   python -c "import ocr; assert 'site-packages' not in ocr.__file__"
   ```

### Tool Integration

**Recommended Stack**:
- `master_audit.py` - Systematic scanning
- `yq` - YAML manipulation
- `ast-grep` - Structural code analysis
- ADT `intelligent-search` - Symbol location
- `migration_guard.py` - Pre-execution validation

**Workflow**:
```bash
# 1. Validate environment
python scripts/audit/migration_guard.py

# 2. Audit current state
python scripts/audit/master_audit.py > audit.txt

# 3. Auto-heal Hydra configs
python scripts/audit/auto_align_hydra.py

# 4. Verify fixes
python scripts/audit/master_audit.py
```

---

## References

- Initial analysis: `01_initial_analysis.md`
- Pain points detail: `artifacts/analysis_outputs/debugging_pain_points.md`
- Broken targets: `artifacts/analysis_outputs/broken_targets.json`
- Tool guides: `artifacts/tool_guides/`
- Implementation guides: `artifacts/implementation_guides/`
