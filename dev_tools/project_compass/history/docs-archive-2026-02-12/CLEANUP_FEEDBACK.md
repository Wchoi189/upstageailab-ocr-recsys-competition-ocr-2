# Nuclear Cleanup - Feedback & Recommendations

**Date**: 2026-02-12
**Status**: Ready for execution
**Confidence**: HIGH - Clear path forward

---

## TL;DR Recommendations

### Immediate Actions (User Decisions Required)

1. **✅ Remove MCP Server** - Complete removal, zero legacy
2. **✅ Remove Spec-Kit** - You said it's overkill, purge it
3. **✅ Remove Snapshots** - Git commits provide same value, adds complexity
4. **✅ Use Nested Staging** - Keep `project_compass/pulse_staging/`, delete top-level
5. **✅ Delete Router** - Only routes MCP calls, not needed after MCP removal

### Execution Plan

**Timeline**: 6-8 hours (optimized from initial 10-hour estimate)
**Risk**: Medium (well-defined scope, good backups)
**Breaking Changes**: YES - Major version bump to v3.0.0

---

## Investigation Results

### 1. Staging Directory Mystery - SOLVED ✅

**Finding**: TWO `pulse_staging/` directories with DIFFERENT content

**Top-Level** (`/workspaces/dev_tools/project_compass/pulse_staging/`):
- Contains: audit/, findings/, recommendations/, tests/, INDEX.md, CONTINUATION_PROMPT.md
- **This is CURRENT ACTIVE PULSE artifacts**
- Used by current session

**Nested** (`/workspaces/dev_tools/project_compass/project_compass/pulse_staging/`):
- Contains: constitution.md, specification.md, implementation_plan.md, tasks.md
- **These are spec-kit generated files** (being removed anyway)

**Why Two Exist**:
- Code expects: `{compass_dir}/pulse_staging/` → `/workspaces/dev_tools/project_compass/project_compass/pulse_staging/`
- But `.vessel` is at: `/workspaces/dev_tools/project_compass/.vessel/`
- Path logic in `core.py`:
  ```python
  self.compass_dir = self.project_root / "project_compass"
  self.staging_dir = self.compass_dir / "pulse_staging"
  ```
- But actual usage puts vessel at project_root level

**CRITICAL ISSUE FOUND**: Path configuration is INCONSISTENT

**Resolution Strategy**:
1. Move ACTIVE artifacts from top-level to nested (correct location per code)
2. Delete top-level `pulse_staging/` entirely
3. Update `.vessel` location OR update path logic for consistency

**Recommendation**: Move `.vessel` INSIDE `project_compass/` for consistency:
```
project_compass/
├── .vessel/
├── pulse_staging/
├── vault/
└── history/
```

### 2. Snapshots Investigation - REMOVE ❌

**Current Usage**:
```bash
snapshots/recognition-audit-post-refactor/
├── 20260202_201246_test_script_snapshot/
└── context-bundling-audit/
```

**Function**: Mid-pulse checkpointing (pulse stays active)

**Analysis**:
- **Git commits**: Provide version control with better tools (diff, revert, history)
- **Pulse export**: Provides complete archiving when work is done
- **Snapshot adds**: Extra directory, extra code, extra complexity

**Overlap with Git**:
- Snapshot timestamps files → Git commits timestamp
- Snapshot preserves state → Git commits preserve state
- Snapshot allows rollback → Git reset/revert allows rollback

**Value Proposition**: Weak - Git already solves this problem

**Recommendation**: **REMOVE** - Delete `snapshots/`, remove `create_snapshot()` function

**Migration**: Archive existing snapshots to `history/snapshots-archive/` before removal

### 3. Router Module - DELETE ✅

**File**: `project_compass/src/router.py` (91 lines)

**Purpose**: Routes MCP meta-tool calls to specific tool handlers

**Function**:
```python
route_pulse("init", args) → {"tool_name": "pulse_init", "arguments": args}
route_spec("specify", args) → {"tool_name": "spec_specify", "arguments": args}
```

**Used By**: `mcp_server.py` ONLY

**Conclusion**: Once MCP server is deleted, router has zero purpose

**Recommendation**: **DELETE** completely

### 4. External MCP Usage - NONE FOUND ✅

**Search Result**: No external MCP calls found outside project_compass

**Implication**: Safe to remove MCP server without breaking external systems

**Confidence**: HIGH

---

## Architecture Simplification Impact

### Before Cleanup (Current Toxic State)

```
┌─────────────────────────────────────────────┐
│           User / AI Agent                   │
└─────────────────────────────────────────────┘
               │
       ┌───────┼───────┬──────────────┐
       │       │       │              │
   ┌───▼───┐ ┌▼──────┐│      ┌───────▼──────┐
   │Skills │ │  CLI  ││      │ Direct MCP    │
   │(new)  │ │(old)  ││      │ (unified)     │
   └───┬───┘ └┬──────┘│      └───────┬──────┘
       │      │       │              │
       │      │  ┌────▼──────────────▼─────┐
       │      │  │   MCP Server (519 lines)│
       │      │  │   router.py (91 lines)  │
       └──────┼──┤   compass_meta_pulse     │
              │  │   compass_meta_spec      │
              │  └────────┬─────────────────┘
              │           │
          ┌───▼───────────▼───┐
          │  Python Core      │
          │  PulseManager     │
          │  pulse_exporter   │
          │  rule_injector    │
          └───────────────────┘
```

**Layers**: 4 (User → Skills/CLI/MCP → MCP Server → Core)
**Lines of Code**: ~2,000+
**Confusion**: HIGH (3 entry points, unclear primary path)

### After Cleanup (Target Clean State)

```
┌─────────────────────────────────────────────┐
│           User / AI Agent                   │
└─────────────────────────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
   ┌───▼───┐       ┌───▼────┐
   │Skills │       │  CLI   │
   │(main) │       │(auto)  │
   └───┬───┘       └───┬────┘
       │               │
       └───────┬───────┘
               │
          ┌────▼────────────────┐
          │  Python Core        │
          │  PulseManager       │
          │  pulse_exporter     │
          │  rule_injector      │
          └─────────────────────┘
```

**Layers**: 2 (User → Skills/CLI → Core)
**Lines of Code**: ~800 (60% reduction)
**Clarity**: HIGH (skills primary, CLI for automation)

**Benefits**:
- 🎯 Single clear path (Skills)
- 📉 60% less code to maintain
- 🚀 Faster execution (no MCP layer)
- 🧹 Zero legacy confusion
- 📝 Simpler documentation

---

## Detailed Removal Checklist

### Phase 1: Backup & Preparation (30 min)

- [ ] Git commit current state
- [ ] Git tag: `v2.1.0-pre-nuclear-cleanup`
- [ ] Archive snapshots: `mv snapshots/ history/snapshots-archive-2026-02-12/`
- [ ] Document current vessel location: `.vessel/` at project root
- [ ] List all files in both `pulse_staging/` dirs

### Phase 2: File Deletions (30 min)

**Delete Completely**:
- [ ] `project_compass/mcp_server.py` (519 lines)
- [ ] `project_compass/src/router.py` (91 lines)
- [ ] `snapshots/` directory (after archiving)
- [ ] `design/spec-kit-integration.md` (obsolete)

**Partial Deletions** (cli.py):
- [ ] Lines 140-180: spec-* subparsers (4 commands)
- [ ] Lines 329-508: spec-* command handlers
- [ ] Lines 329-336: spec imports (if any)

**Cleanup** (pulse_exporter.py):
- [ ] Remove `create_snapshot()` function
- [ ] Remove snapshot imports
- [ ] Update docstrings

### Phase 3: Directory Consolidation (45 min)

**Issue**: `.vessel` at project root, but code expects nested structure

**Option A (Recommended)**: Move vessel inside project_compass
```bash
mv /workspaces/dev_tools/project_compass/.vessel \
   /workspaces/dev_tools/project_compass/project_compass/.vessel
```

**Option B**: Update all path logic in `core.py`

**Staging Consolidation**:
1. Move current artifacts from top-level to nested:
   ```bash
   cp -r /workspaces/dev_tools/project_compass/pulse_staging/artifacts/* \
         /workspaces/dev_tools/project_compass/project_compass/pulse_staging/artifacts/
   ```
2. Verify vessel_state.json references are correct
3. Delete top-level: `rm -rf /workspaces/dev_tools/project_compass/pulse_staging/`

### Phase 4: Skills Updates (90 min)

Update all 7 skills from MCP calls to CLI calls:

**Template Change**:
```yaml
# OLD (MCP)
Call MCP tool:
mcp__unified__compass_meta_pulse(kind="init", ...)

# NEW (CLI)
Execute CLI:
!`uv run compass pulse-init --id [id] --obj "[obj]" --milestone [milestone]`
```

**Files to Update**:
- [ ] `skills/compass-start/SKILL.md` - Change init call
- [ ] `skills/compass-status/SKILL.md` - Change status call
- [ ] `skills/compass-register/SKILL.md` - Change sync call
- [ ] `skills/compass-finish/SKILL.md` - Change export call
- [ ] `skills/compass-resume/SKILL.md` - Update references
- [ ] `skills/compass-help/SKILL.md` - Remove MCP documentation
- [ ] `skills/audit-run/SKILL.md` - Update references

### Phase 5: Documentation Updates (120 min)

#### AGENTS.md (Complete Rewrite)
- [ ] Section 1: Update "Core Concepts" - add Skills
- [ ] Section 3: Remove MCP Resources
- [ ] Section 3: Replace CLI section with Skills section
- [ ] Section 5: Update Pulse Lifecycle diagram
- [ ] Section 8: Delete Spec Kit Integration entirely
- [ ] New section: "Skills vs CLI" guidance

#### AGENTS.yaml (Restructure)
- [ ] Update header: v3.0, ~40 tokens
- [ ] Add `primary_interface.skills` section
- [ ] Move CLI to `automation_interface.cli`
- [ ] Remove spec-* commands
- [ ] Add deprecation notes

#### CHANGELOG.md (New Entry)
- [ ] Create v3.0.0 section
- [ ] Document BREAKING CHANGES
- [ ] List all removals
- [ ] List all additions
- [ ] Write migration guide

#### Skills Docs Updates
- [ ] `skills/README.md` - Remove MCP references
- [ ] `skills/QUICKSTART.md` - Update all examples
- [ ] `SKILLS_IMPLEMENTATION.md` - Note MCP removal

### Phase 6: Testing & Validation (90 min)

**Unit Tests** (if they exist):
- [ ] Update test mocks (no MCP server)
- [ ] Remove spec-kit tests
- [ ] Run test suite

**Integration Tests**:
- [ ] Test: `/compass-start` → creates pulse
- [ ] Test: Create artifact in staging
- [ ] Test: `/compass-register` → registers artifact
- [ ] Test: `/compass-status` → shows state
- [ ] Test: `/compass-finish` → exports to history
- [ ] Test: Directive injection works
- [ ] Test: CLI direct usage works
- [ ] Test: No broken imports

**Edge Cases**:
- [ ] Start pulse when one exists (should fail)
- [ ] Register nonexistent file (should fail)
- [ ] Export with unregistered files (should block)

### Phase 7: Cleanup Verification (30 min)

**Code Search for Orphaned References**:
```bash
# Search for MCP references
grep -r "mcp_server" project_compass/
grep -r "compass_meta" project_compass/
grep -r "router.py" project_compass/

# Search for spec-kit references
grep -r "spec-constitution" project_compass/
grep -r "spec_constitution" project_compass/

# Search for snapshot references
grep -r "create_snapshot" project_compass/
grep -r "snapshot_meta" project_compass/
```

**Expected Result**: Zero matches (except in CHANGELOG.md)

---

## Artifact Type Cleanup

### Current Types (state_schema.py)
```python
class ArtifactType(str, Enum):
    DESIGN = "design"
    RESEARCH = "research"
    WALKTHROUGH = "walkthrough"
    IMPLEMENTATION_PLAN = "implementation_plan"
    BUG_REPORT = "bug_report"
    AUDIT = "audit"
    # Spec-kit additions:
    SPECIFICATION = "specification"        # REMOVE
    REQUIREMENTS = "requirements"          # REMOVE
    ARCHITECTURE = "architecture"          # REMOVE
```

**Action**: Remove 3 spec-kit types from enum

**Reasoning**: Without spec-kit commands, these types are unused

---

## Path Configuration Fix

### Current Issue
```
project_compass/
├── .vessel/                    # ← Actual location (WRONG per code)
├── project_compass/
│   ├── .vessel/                # ← Expected by paths.py (doesn't exist)
│   ├── pulse_staging/          # ← Expected by paths.py
│   └── src/
└── pulse_staging/              # ← Actual staging (WRONG per code)
```

**Inconsistency**: Vessel and staging at different levels

### Recommended Fix
```
project_compass/                # ← Project root
├── project_compass/            # ← Compass directory
│   ├── .vessel/                # ← Move here
│   ├── pulse_staging/          # ← Correct location
│   ├── vault/
│   ├── history/
│   └── src/
├── skills/
└── AGENTS.md
```

**Implementation**:
```bash
# Move vessel
mv /workspaces/dev_tools/project_compass/.vessel \
   /workspaces/dev_tools/project_compass/project_compass/.vessel

# Merge staging
cp -r /workspaces/dev_tools/project_compass/pulse_staging/artifacts/* \
      /workspaces/dev_tools/project_compass/project_compass/pulse_staging/artifacts/

# Delete old staging
rm -rf /workspaces/dev_tools/project_compass/pulse_staging/

# Update vessel_state.json if paths are absolute (unlikely)
```

---

## Risk Mitigation

### Backup Strategy
1. **Git Tag**: `v2.1.0-pre-nuclear-cleanup`
2. **Snapshot Archive**: Move `snapshots/` to `history/` before deletion
3. **Staging Backup**: Copy both staging dirs before consolidation
4. **Rollback Plan**: `git reset --hard v2.1.0-pre-nuclear-cleanup`

### Testing Strategy
1. **Test After Each Phase**: Don't batch changes
2. **Keep CLI Working**: Test direct CLI usage throughout
3. **Skill Testing**: Test one skill completely before updating others
4. **No Force Push**: Keep refactor in branch initially

### Communication
- Document what was removed in CHANGELOG
- Note breaking changes clearly
- Provide migration examples

---

## Timeline Estimate (Revised)

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| 1 | Backup & Prep | 30min | 0.5h |
| 2 | File Deletions | 30min | 1h |
| 3 | Directory Consolidation | 45min | 1.75h |
| 4 | Skills Updates | 90min | 3.25h |
| 5 | Documentation | 120min | 5.25h |
| 6 | Testing | 90min | 6.75h |
| 7 | Verification | 30min | 7.25h |

**Total**: ~7-8 hours (vs original 10-hour estimate)

**Optimization**: Phases can be done incrementally over multiple sessions

---

## Success Metrics

After cleanup, verify:
- ✅ **Code Reduction**: ~60% less code (1200 lines removed)
- ✅ **Layer Reduction**: 4 layers → 2 layers
- ✅ **File Count**: 2 major files deleted, multiple cleanups
- ✅ **Documentation**: All docs reference skills as primary
- ✅ **Testing**: Complete workflow tested end-to-end
- ✅ **Zero Legacy**: No MCP references, no spec-kit traces
- ✅ **Path Consistency**: Single vessel location, single staging location

---

## Recommended Execution Order

### Session 1 (2-3 hours): Preparation & Deletions
1. Backup (git tag)
2. Archive snapshots
3. Delete MCP server, router, spec-kit code
4. Test CLI still works

### Session 2 (2-3 hours): Directory Consolidation & Skills
1. Fix path inconsistency (move vessel)
2. Consolidate staging directories
3. Update 7 skills (MCP → CLI)
4. Test skills work

### Session 3 (2-3 hours): Documentation & Testing
1. Rewrite AGENTS.md, AGENTS.yaml
2. Write CHANGELOG v3.0 entry
3. Complete integration testing
4. Verification sweep

**Advantage**: Can stop/resume between sessions, each session delivers value

---

## Final Recommendations

### Immediate Approval Needed

**Question 1**: Proceed with nuclear cleanup as specified?
- Remove MCP server ✓
- Remove spec-kit ✓
- Remove snapshots ✓
- Consolidate staging ✓
- Delete router ✓

**Question 2**: Timeline acceptable? (3 sessions, ~7-8 hours total)

**Question 3**: Path consolidation approach?
- Move `.vessel` inside `project_compass/` (Recommended)
- OR update path logic in `core.py`

### What You'll Get

**Single, Clean Architecture**:
- Skills as primary interface
- CLI for automation
- Zero legacy code
- Clear documentation
- 60% less code to maintain

**Breaking Changes**:
- MCP tools gone (but no external usage found)
- Spec-kit gone (you said it was overkill)
- Snapshots gone (git commits better)

**Risk Level**: Medium (well-scoped, good backups, clear rollback)

---

**Ready to proceed?** Confirm decisions and I'll execute the nuclear cleanup.
