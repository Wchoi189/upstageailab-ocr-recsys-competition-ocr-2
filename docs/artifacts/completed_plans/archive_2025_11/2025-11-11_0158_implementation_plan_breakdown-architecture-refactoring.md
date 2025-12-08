---
title: "Breakdown Architecture Refactoring"
date: "2025-12-06 18:09 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---









# Implementation Plan Breakdown – Architecture Refactoring

## Executive Summary

This document breaks down the architecture audit findings into **5 executable implementation plans** that can be executed by unsupervised workers. Each plan is designed to be:
- **Self-contained** with minimal cross-dependencies
- **Validatable** without runtime dependencies (syntax/import checks)
- **Atomic** with clear rollback procedures
- **Risk-assessed** with mitigation strategies

## Plan Breakdown Overview

| Plan ID | Name | Risk | Dependencies | Parallelizable | Est. Complexity |
|---------|------|------|---------------|----------------|----------------|
| **PLAN-001** | Core Training Stabilization | **CRITICAL** | None | No | High |
| **PLAN-002** | Polygon Validation Consolidation | **HIGH** | PLAN-001 | After PLAN-001 | Medium |
| **PLAN-003** | Import-Time Optimization | **MEDIUM** | None | Yes | Low |
| **PLAN-004** | Inference Service Consolidation | **VERY HIGH** | None | Yes | Very High |
| **PLAN-005** | Legacy Cleanup & Config Consolidation | **LOW** | None | Yes | Low |

## Dependency Graph

```
PLAN-001 (Core Training) ──┐
                            ├──> PLAN-002 (Polygon Validation)
PLAN-003 (Imports) ─────────┼──> (Independent)
PLAN-004 (Inference) ───────┼──> (Independent)
PLAN-005 (Cleanup) ─────────┘──> (Independent)
```

**Execution Order:**
1. **PLAN-001** (must complete first - unblocks training)
2. **PLAN-002** (depends on PLAN-001)
3. **PLAN-003, PLAN-004, PLAN-005** (can run in parallel after PLAN-001)

## Risk Assessment Matrix

### PLAN-001: Core Training Stabilization
- **Risk Level:** CRITICAL ⚠️⚠️⚠️
- **Impact if Broken:** Training pipeline completely non-functional
- **Recovery:** Requires immediate rollback, may corrupt checkpoints
- **Mitigation:**
  - Atomic commits per change
  - Comprehensive syntax validation
  - Unit test stubs (no runtime needed)
  - Clear rollback instructions

### PLAN-002: Polygon Validation Consolidation
- **Risk Level:** HIGH ⚠️⚠️
- **Impact if Broken:** Data pipeline failures, training instability
- **Recovery:** Medium complexity rollback
- **Mitigation:**
  - Shared utility with backward compatibility
  - Gradual migration path
  - Validation script checks

### PLAN-003: Import-Time Optimization
- **Risk Level:** MEDIUM ⚠️
- **Impact if Broken:** Import errors, slower startup
- **Recovery:** Easy rollback (mostly additive changes)
- **Mitigation:**
  - Lazy imports are backward compatible
  - Can be disabled via feature flags
  - Low risk of breaking existing code

### PLAN-004: Inference Service Consolidation
- **Risk Level:** VERY HIGH ⚠️⚠️⚠️⚠️
- **Impact if Broken:** UI completely non-functional
- **Recovery:** Complex rollback, may require UI state migration
- **Mitigation:**
  - Feature flag for gradual rollout
  - Maintain old path as fallback
  - Extensive validation checks
  - Consider splitting into sub-plans

### PLAN-005: Legacy Cleanup
- **Risk Level:** LOW ⚠️
- **Impact if Broken:** Minimal (mostly deletion)
- **Recovery:** Easy (git restore)
- **Mitigation:**
  - Archive before deletion
  - Verify no active references
  - Low risk, high reward

## Success Criteria for Unsupervised Execution

### Validation Without Runtime Dependencies

Each plan must include:
1. **Syntax Validation:** `python -m py_compile <file>` for all changed files
2. **Import Validation:** `python -c "import <module>"` (may fail on missing deps, but should not fail on syntax)
3. **Type Checking:** `mypy --ignore-missing-imports <file>` (if available)
4. **Static Analysis:** `ruff check <file>` (if available)
5. **Documentation:** All changes must include docstrings/comments

### Rollback Procedures

Each plan must provide:
1. **Git commands** to revert specific changes
2. **File-level rollback** instructions
3. **Config rollback** steps (if configs changed)
4. **Verification** that rollback succeeded

### Atomic Change Strategy

- **One logical change per commit** (not one file per commit)
- **Feature flags** where possible for gradual rollout
- **Backward compatibility** maintained during transition
- **Clear commit messages** explaining what and why

## Detailed Plan Specifications

*(Each plan will be expanded in separate implementation plan documents)*

### PLAN-001: Core Training Stabilization
**Files to Modify:**
- `ocr/models/head/db_head.py` (step function fix)
- `ocr/models/loss/dice_loss.py` (input clamping)
- `ocr/lightning_modules/ocr_pl.py` (remove redundant CPU detaches)
- `configs/dataloaders/*.yaml` (CUDA-safe presets)
- `configs/hardware/*.yaml` (batch size defaults)

**Key Changes:**
1. Replace `torch.reciprocal(1 + torch.exp(-k(x-y)))` with `torch.sigmoid(k * (x - y))`
2. Add input clamping before step function
3. Remove `.detach().cpu()` validation calls from forward pass
4. Update hardware presets for 12GB GPUs

**Validation:**
- Syntax check all files
- Verify no `torch.reciprocal` + `torch.exp` pattern remains
- Check that sigmoid is used correctly
- Verify config YAML syntax

### PLAN-002: Polygon Validation Consolidation
**Files to Modify:**
- `ocr/utils/polygon_utils.py` (create shared validators)
- `ocr/datasets/base.py` (use shared validators)
- `ocr/datasets/db_collate_fn.py` (use shared validators)
- `ocr/lightning_modules/callbacks/wandb_image_logging.py` (use shared validators)

**Key Changes:**
1. Extract polygon validation logic into `ocr.utils.polygon_validators`
2. Replace duplicate implementations with shared utility calls
3. Ensure backward compatibility during migration

**Validation:**
- Verify all three locations use shared utility
- Check that function signatures match
- Ensure no duplicate validation logic remains

### PLAN-003: Import-Time Optimization
**Files to Modify:**
- `runners/train.py` (lazy wandb import)
- `ocr/lightning_modules/callbacks/*.py` (lazy wandb)
- `configs/callbacks/default.yaml` (conditional callbacks)
- `ui/apps/*/services/*.py` (lazy streamlit)
- `pyproject.toml` (optional dependencies)

**Key Changes:**
1. Move wandb imports inside functions guarded by config checks
2. Make streamlit imports conditional
3. Add optional dependency groups
4. Update callback defaults to respect wandb config

**Validation:**
- Verify imports are inside functions, not module scope
- Check that config guards are present
- Ensure backward compatibility

### PLAN-004: Inference Service Consolidation
**Files to Modify:**
- `ui/utils/inference/engine.py` (add singleton caching)
- `ui/apps/inference/services/inference_runner.py` (use cached engine)
- `ui/apps/unified_ocr_app/services/inference_service.py` (use cached engine)
- `ui/utils/inference/__init__.py` (export caching API)

**Key Changes:**
1. Add checkpoint caching to InferenceEngine
2. Create shared engine instance
3. Eliminate tempfile duplication
4. Stream numpy arrays directly

**Validation:**
- Verify engine caching logic
- Check that tempfile usage is eliminated
- Ensure backward compatibility

### PLAN-005: Legacy Cleanup
**Files to Remove:**
- `.backup/scripts-backup-*/` (archive first)
- `scripts/debug_cuda.sh` (if Python version exists)
- Duplicate config files

**Key Changes:**
1. Archive backup directories
2. Remove duplicate scripts
3. Consolidate config presets
4. Update documentation references

**Validation:**
- Verify no active references to deleted files
- Check that git history preserves deleted files
- Ensure documentation is updated

## Execution Strategy for Unsupervised Workers

### Pre-Execution Checklist
- [ ] Read entire plan document
- [ ] Understand risk level and mitigation strategies
- [ ] Verify git branch is correct
- [ ] Ensure backup/rollback procedures are clear
- [ ] Confirm validation commands work

### During Execution
1. **One change at a time** - don't batch unrelated changes
2. **Validate after each change** - run syntax/import checks
3. **Commit frequently** - atomic commits with clear messages
4. **Document decisions** - add comments explaining why
5. **Test incrementally** - verify each change doesn't break syntax

### Post-Execution Validation
1. **Syntax Check:** All modified files compile
2. **Import Check:** All imports resolve (syntax-wise)
3. **Static Analysis:** Run linters if available
4. **Documentation:** Verify docstrings/comments added
5. **Git Status:** Clean working directory, all changes committed

### Rollback Procedure (if needed)
1. Identify the problematic commit(s)
2. Use `git revert` or `git reset` as specified in plan
3. Verify rollback with syntax checks
4. Document what went wrong for future reference

## Parallelization Strategy

### Can Run in Parallel (after PLAN-001):
- **PLAN-003** (Import optimization) - Independent, low risk
- **PLAN-005** (Legacy cleanup) - Independent, low risk
- **PLAN-004** (Inference) - Independent but high risk, consider sequential

### Must Run Sequentially:
- **PLAN-001** → **PLAN-002** (dependency on shared utilities)

### Recommended Execution Order:
1. **PLAN-001** (critical, blocks everything)
2. **PLAN-002** (depends on PLAN-001)
3. **PLAN-003** + **PLAN-005** (parallel, low risk)
4. **PLAN-004** (last, highest risk, needs careful testing)

## Risk Mitigation for Aggressive Unsupervised Refactoring

### 1. Feature Flags
- Add config flags to enable/disable new code paths
- Keep old paths as fallback
- Gradual rollout capability

### 2. Comprehensive Validation
- Syntax checks (no runtime needed)
- Import resolution checks
- Static analysis where possible
- Documentation completeness

### 3. Atomic Commits
- One logical change per commit
- Clear commit messages
- Easy to identify and revert problematic changes

### 4. Backward Compatibility
- Maintain old APIs during transition
- Deprecation warnings before removal
- Gradual migration path

### 5. Documentation
- Inline comments explaining changes
- Docstrings for new functions
- Updated architecture docs
- Migration guides if needed

### 6. Testing Strategy (for workers without runtime)
- Syntax validation scripts
- Import resolution checks
- Static type checking
- Code review checklists

## Success Metrics

### Immediate (Post-Execution):
- ✅ All files pass syntax validation
- ✅ All imports resolve (syntax-wise)
- ✅ No duplicate code remains
- ✅ Documentation updated

### Short-term (After Integration):
- ✅ Training pipeline runs without CUDA errors
- ✅ Import time reduced by 30%+
- ✅ Inference latency improved
- ✅ Code duplication reduced

### Long-term (After Testing):
- ✅ No NaN/Inf gradients in training
- ✅ Stable training runs (200+ steps)
- ✅ UI responsive and functional
- ✅ Maintainability improved

## Recommendations

### How Many Plans?
**Answer: 5 separate plans** (as outlined above)

**Rationale:**
- Each addresses a distinct concern (training, data, imports, UI, cleanup)
- Different risk levels require different execution strategies
- Parallelization opportunities exist
- Easier to track progress and rollback if needed

### Parallelization?
**Answer: Limited parallelization**

**Can run in parallel:**
- PLAN-003 (Imports) + PLAN-005 (Cleanup) - both low risk, independent
- After PLAN-001 completes: PLAN-002, PLAN-003, PLAN-005 can proceed

**Should run sequentially:**
- PLAN-001 must complete first (blocks training)
- PLAN-002 depends on PLAN-001 (uses shared utilities)
- PLAN-004 should run last (highest risk, needs careful validation)

### Risk Level?
**Answer: EXTREMELY HIGH** ⚠️⚠️⚠️⚠️

**Why:**
- Core training changes (PLAN-001) can break entire pipeline
- Inference consolidation (PLAN-004) can break UI completely
- Accumulated changes increase risk exponentially
- No runtime testing means issues discovered late

**Mitigation:**
- Feature flags for all major changes
- Maintain backward compatibility
- Atomic commits for easy rollback
- Comprehensive validation scripts
- Clear rollback procedures

### Increasing Success Likelihood

**For Unsupervised Execution:**

1. **Detailed Step-by-Step Instructions**
   - Exact file paths and line numbers
   - Before/after code examples
   - Clear success criteria for each step

2. **Validation Scripts**
   - Syntax checking automation
   - Import resolution verification
   - Pattern matching (e.g., verify sigmoid usage)

3. **Feature Flags**
   - Enable new code paths via config
   - Keep old paths as fallback
   - Gradual rollout capability

4. **Comprehensive Documentation**
   - Inline comments explaining why
   - Docstrings for all new functions
   - Migration notes for breaking changes

5. **Atomic Changes**
   - One logical change per commit
   - Easy to identify and revert
   - Clear commit messages

6. **Backward Compatibility**
   - Maintain old APIs during transition
   - Deprecation warnings
   - Gradual migration path

7. **Pre-commit Validation**
   - Syntax checks
   - Import resolution
   - Static analysis
   - Documentation completeness

## Next Steps

1. **Create detailed implementation plans** for each PLAN-00X
   - Expand each plan with step-by-step instructions
   - Include code examples and validation steps
   - Provide rollback procedures

2. **Generate validation scripts**
   - Syntax checking automation
   - Import resolution verification
   - Pattern matching validation

3. **Prepare feature flags**
   - Config options for new code paths
   - Backward compatibility switches
   - Gradual rollout controls

4. **Set up execution tracking**
   - Progress checklists
   - Validation results logging
   - Rollback decision points

5. **Assign plans to workers**
   - Match risk tolerance to plan risk level
   - Provide clear instructions and validation tools
   - Establish communication channels for questions

---

**Status:** Draft - Awaiting review and expansion into detailed plans
**Last Updated:** 2025-11-11 01:58 KST
