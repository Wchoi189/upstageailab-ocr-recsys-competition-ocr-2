---
type: handover
pulse_id: hydra-refactor-2026-01-22
session_date: 2026-01-25
status: paused
reason: architectural_decision_required
---

# Session Handover: Hydra Refactor 2026-01-25

## Session Summary

**Objective:** Resolve 81 broken imports and 12 broken Hydra targets  
**Progress:** Partial - encountered architectural conflict  
**Status:** Paused - requires planning

## What Was Accomplished

### Artifacts Created
1. **Roadmap**: [ocr-hydra-refactor-roadmap.md](../vault/milestones/ocr-hydra-refactor-roadmap.md)
   - 4-phase plan with detailed tasks
   - Success criteria and verification commands
   - Risk mitigation strategies

2. **Tracking Document**: [hydra-refactor-progress-tracking.md](hydra-refactor-progress-tracking.md)
   - Phase-by-phase task lists
   - Metrics tracking table
   - Daily progress log

3. **Batch Fix Script**: `scripts/audit/batch_fix_imports.py`
   - Automated import replacement
   - 17 successful fixes applied

### Fixes Applied
- ✅ Fixed 17 imports (ocr.agents → ocr.core.infrastructure.agents)
- ✅ Fixed ocr.detection → ocr.domains.detection
- ✅ Fixed base class imports (BaseEncoder, BaseDecoder, etc.)
- ✅ Created loss module __init__.py

### Analysis Completed
- Pattern categorization: 42 ocr.core issues, 5 missing deps, etc.
- Import flow analysis via batch script
- Hydra target accessibility investigation

## Critical Issue Discovered

### The Lazy Loading vs Hydra Conflict

**Problem:**
- Hydra's `_target_` requires explicit imports in `__init__.py`
- Explicit imports trigger circular dependencies
- Lazy loading (`__getattr__`) prevents Hydra from finding classes

**Evidence:**
```
Initial audit: 81 broken imports, 12 broken Hydra targets
After batch fixes: 74 broken imports (-7)
After __init__ fixes: 90 broken imports (+16), 15 broken targets (+3)
```

**Root Cause:**
Lazy loading architecture conflicts with Hydra's instantiation mechanism.

## Architectural Decision Required

### Option 1: Full Module Paths in Hydra Configs
**Approach:** Use complete paths like `ocr.domains.detection.models.heads.craft_head.CraftHead`  
**Pros:** Bypasses __init__, no circular imports  
**Cons:** Verbose configs, harder to refactor  
**Implementation:** Update all YAML _target_ paths

### Option 2: TYPE_CHECKING Guards
**Approach:** Import for type hints only, lazy load at runtime  
**Pros:** Maintains lazy loading, adds type safety  
**Cons:** Hydra still can't instantiate  
**Implementation:** Won't solve Hydra issue

### Option 3: Eager Loading with Import Order Fix
**Approach:** Explicit imports but fix circular dependencies  
**Pros:** Hydra works, simpler configs  
**Cons:** Requires careful import ordering  
**Implementation:** Refactor base classes, use forward refs

### Option 4: Hydra Custom Resolvers
**Approach:** Create resolvers that handle lazy loading  
**Pros:** Keeps lazy loading, Hydra-compatible  
**Cons:** Complex, requires custom Hydra setup  
**Implementation:** New resolver infrastructure

## Recommended Next Steps

### Immediate (Next Session)
1. **Revert problematic __init__ changes:**
   ```bash
   git checkout ocr/domains/recognition/models/__init__.py
   git checkout ocr/domains/detection/models/{heads,decoders,encoders}/__init__.py
   ```

2. **Try Option 1 (Full Paths) on 1-2 configs:**
   - Update `configs/model/architectures/craft.yaml`
   - Test with: `uv run python runners/train.py experiment=det_resnet50_v1 +trainer.fast_dev_run=True`

3. **Document findings:**
   - If Option 1 works → create batch YAML update script
   - If fails → investigate Option 3 or 4

### Short Term (This Week)
- Complete Option 1 for all 12-15 Hydra targets
- Fix remaining 74 broken imports via batch script
- Run full audit to verify no new issues

### Medium Term (Next Week)
- Test both pipelines end-to-end
- Add pre-commit hooks for import validation
- Document pattern in AgentQMS standards

## Files to Review

### Modified Files (This Session)
```
scripts/audit/batch_fix_imports.py                    # NEW - working well
ocr/domains/detection/models/loss/__init__.py         # NEW - needs verification
ocr/domains/detection/models/heads/__init__.py        # MODIFIED - revert needed
ocr/domains/detection/models/decoders/__init__.py     # MODIFIED - revert needed  
ocr/domains/detection/models/encoders/__init__.py     # MODIFIED - revert needed
ocr/domains/recognition/models/__init__.py            # MODIFIED - revert needed
+ 17 files with import fixes (via batch script)       # KEEP
```

### Key References
- [Debug Session](__DEBUG__/2026-01-22_hydra_configs_legacy_imports/README.md)
- [Master Audit](../../scripts/audit/master_audit.py)
- [Auto-Align Hydra](../../scripts/audit/auto_align_hydra.py)

## Metrics Snapshot

| Metric                 | Session Start | Session End | Target |
| ---------------------- | ------------- | ----------- | ------ |
| Broken Python imports  | 81            | 90          | 0      |
| Broken Hydra targets   | 12            | 15          | 0      |
| Detection pipeline     | ⏳             | ⏳           | ✅      |
| Recognition pipeline   | ⏳             | ⏳           | ✅      |

## Context for Next Agent

1. **Don't repeat eager loading mistake:** Lazy loading is there for a reason
2. **Hydra needs full paths:** Short paths require __init__ imports
3. **Test incrementally:** One config at a time, validate pipeline
4. **Use batch_fix_imports.py:** It works well for simple replacements
5. **Check circular deps:** Before any __init__ changes, trace import chains

## Questions to Resolve

1. Why was lazy loading chosen originally? (Check git history/docs)
2. Are there examples of Hydra working with lazy loading elsewhere?
3. Can we use Hydra's OmegaConf for dynamic resolution?
4. Would moving base classes to separate module help?

## Session Health

- **Token Usage:** ~50k (moderate)
- **Progress:** 20% - planning complete, implementation blocked
- **Blockers:** Architectural decision needed
- **Risk:** Medium - wrong choice could cascade issues

## Compass Status

```bash
$ uv run compass pulse-status
🔥 Active Pulse
   ID: hydra-refactor-2026-01-22
   Objective: Resolve all Hydra and import issues toward zero errors...
   Milestone: ocr-domain-refactor
   Artifacts: 1
   Rules: 4
   Token Burden: low
```

---

**Handover Complete** - Ready for next session with clear decision point.
