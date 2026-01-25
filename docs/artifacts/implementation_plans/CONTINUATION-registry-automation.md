# Continuation Prompt: Begin Registry Automation Implementation

## Task: Execute ADS v2.0 Registry Automation - Phase 0 & Phase 1 Foundation

**Goal**: Archive legacy standards and build the intelligent registry compiler with strict validation, cycle detection, and visual graphs

---

## Context

**Implementation Plan**: [docs/artifacts/implementation_plans/registry-automation-plan.md](docs/artifacts/implementation_plans/registry-automation-plan.md)  
**Quick Start Guide**: [docs/artifacts/implementation_plans/QUICKSTART-registry-automation.md](docs/artifacts/implementation_plans/QUICKSTART-registry-automation.md)  
**Updates Summary**: [docs/artifacts/implementation_plans/registry-automation-UPDATES.md](docs/artifacts/implementation_plans/registry-automation-UPDATES.md)  
**Specification**: [specs/001-registry-automation/spec.md](specs/001-registry-automation/spec.md)

**Current State**:
- 440-line manually-maintained `registry.yaml` (error-prone)
- 71 existing standards across 4 tiers
- Agent context burden: ~1000 tokens per query
- Manual synchronization causing drift

**Target State**:
- 100% auto-generated registry from distributed ADS v2.0 headers
- Agent context: ~200 tokens per query (80% reduction)
- Strict enforcement (zero untracked files)
- Cycle detection (prevents circular dependencies)
- Visual architecture graphs

---

## Milestones

### Phase 0: Nuclear Archive ✅ READY TO EXECUTE
**Duration**: 1-2 hours  
**Target**: Archive all 71 legacy standards, create backup, establish clean workspace

**Commands**:
```bash
# Archive all standards (REVERSIBLE)
mkdir -p AgentQMS/standards/_archive
mv AgentQMS/standards/tier1-sst/*.yaml AgentQMS/standards/_archive/
mv AgentQMS/standards/tier2-framework/**/*.yaml AgentQMS/standards/_archive/
mv AgentQMS/standards/tier3-agents/**/*.yaml AgentQMS/standards/_archive/ 2>/dev/null || true
mv AgentQMS/standards/tier4-workflows/**/*.yaml AgentQMS/standards/_archive/ 2>/dev/null || true

# Backup registry
cp AgentQMS/standards/registry.yaml AgentQMS/standards/_archive/registry.yaml.backup

# Verify clean workspace
find AgentQMS/standards/tier* -name "*.yaml" 2>/dev/null || echo "✅ Workspace clean"

# Git commit for rollback safety
git add -A
git commit -m "Phase 0: Archive legacy standards for v2.0 rebuild"
```

**Acceptance**:
- [ ] All 71 YAML files in `_archive/` directory
- [ ] Registry backup at `_archive/registry.yaml.backup`
- [ ] No `.yaml` files in `tier*` directories (except schemas/templates)
- [ ] Git commit created for instant rollback

---

### Phase 1: Foundation & Intelligent Compiler ⚡ START HERE
**Duration**: Week 1-2  
**Target**: Build v2.0 schema and enhanced registry compiler

#### Task 1.1: Create ADS v2.0 Header Schema (4 hours)
**File**: `AgentQMS/standards/schemas/ads-header.json`

**Key Fields**:
- `ads_version: "2.0"` (required)
- `id: "naming-conventions"` (required, kebab-case, for dependencies)
- `dependencies: ["other-standard-id"]` (optional, enables graph)
- `fuzzy_threshold: 80` (optional, 0-100, for typo-tolerant search)
- `triggers: {task_id: {keywords, path_patterns}}` (required)

**Reference**: See [registry-automation-plan.md](docs/artifacts/implementation_plans/registry-automation-plan.md) Task 1.1 for full schema implementation

**Tests**:
- Valid header with all fields → passes
- Missing `id` field → rejects
- Invalid `id` format (uppercase/spaces) → rejects
- Circular dependencies (caught by compiler, not schema)

---

#### Task 1.2: Build Intelligent Registry Compiler (18 hours)
**File**: `AgentQMS/tools/sync_registry.py`

**Core Features**:
1. **Strict Mode**: Fail if any `.yaml` lacks valid ADS header (SC-011)
2. **Cycle Detection**: DFS algorithm on dependency graph
3. **Visual Graph**: Generate `architecture_map.dot` (GraphViz)
4. **Pulse Delta**: Print semantic diff ([+]/[-]/[Δ] tasks)
5. **Atomic Write**: Temp file → validate → rename

**Usage**:
```bash
uv run python AgentQMS/tools/sync_registry.py --dry-run  # Validate only
uv run python AgentQMS/tools/sync_registry.py  # Compile + generate graph
dot -Tpng AgentQMS/standards/architecture_map.dot -o architecture.png
```

**Reference**: See [registry-automation-plan.md](docs/artifacts/implementation_plans/registry-automation-plan.md) Task 1.2 for implementation (note: example code may need adjustments)

**Tests**:
- Empty archive → minimal registry
- 5 valid headers → correct task mappings
- File with missing header + strict mode → fails compilation
- Circular dependency A→B→A → detected and blocked
- DOT graph → valid GraphViz syntax

---

#### Task 1.3: Unit Tests (6 hours)
**File**: `AgentQMS/tests/test_sync_registry.py`

**Coverage**:
- Schema validation (valid/invalid headers)
- Cycle detection (A→B→A, A→B→C→A)
- Strict mode enforcement
- DOT graph generation
- Pulse delta accuracy
- Atomic write behavior

**Target**: 95%+ code coverage

---

## Acceptance Criteria (Phase 0 + Phase 1)

- [ ] **Phase 0 Complete**: All 71 files archived, backup created, git commit made
- [ ] **Schema Valid**: `ads-header.json` passes JSON Schema Draft 7 validation
- [ ] **Compiler Functional**: Dry-run succeeds on empty archive
- [ ] **Strict Mode Works**: Compilation fails when non-archived file lacks header
- [ ] **Cycles Detected**: A→B→A dependency caught with clear error
- [ ] **Graph Generated**: `architecture_map.dot` renders in GraphViz
- [ ] **Pulse Delta Shows**: Added/removed/modified tasks displayed correctly
- [ ] **Tests Pass**: 95%+ coverage, all assertions pass
- [ ] **Documentation**: Inline comments explain complex logic

---

## Critical Decision Point (End of Week 2)

**Decision**: Is the v2.0 compiler stable and ready for pilot migration?

**Success Criteria**:
- ✅ Dry-run compiles without crashes
- ✅ Cycle detection catches test case (A→B→A)
- ✅ DOT graph visualizes empty registry
- ✅ Unit tests achieve 95%+ coverage

**If YES** → Proceed to Phase 2 (High-Performance Resolver)  
**If NO** → Debug compiler issues, extend Phase 1 by 1 week

---

## Resources

### Implementation References
- **Main Plan**: [registry-automation-plan.md](docs/artifacts/implementation_plans/registry-automation-plan.md) - Complete 6-week plan with all phases
- **Quick Start**: [QUICKSTART-registry-automation.md](docs/artifacts/implementation_plans/QUICKSTART-registry-automation.md) - Commands and checklists
- **Updates**: [registry-automation-UPDATES.md](docs/artifacts/implementation_plans/registry-automation-UPDATES.md) - v2.0 enhancements explained
- **Specification**: [specs/001-registry-automation/spec.md](specs/001-registry-automation/spec.md) - Requirements and user stories

### Architecture Context
- **Philosophy**: [AgentQMS/standards/tier1-sst/ai-native-architecture.md](AgentQMS/standards/tier1-sst/ai-native-architecture.md) - Registry-driven principles
- **Current Registry**: [AgentQMS/standards/registry.yaml](AgentQMS/standards/registry.yaml) - 440 lines to replace
- **Draft Research**: [__DEBUG__/2026-01-25_standards-update/additional-updates/draft-research.md](__DEBUG__/2026-01-25_standards-update/additional-updates/draft-research.md) - Original conversation

### Dependencies to Install
```bash
# Add to pyproject.toml
uv add jsonschema rapidfuzz pyyaml
```

---

## Rollback Procedure (If Needed)

If critical issues arise during Phase 0 or Phase 1:

```bash
# Restore from archive
git revert HEAD  # Undo Phase 0 commit
cp AgentQMS/standards/_archive/registry.yaml.backup AgentQMS/standards/registry.yaml
mv AgentQMS/standards/_archive/*.yaml AgentQMS/standards/tier*/  # Manual restore by tier

# Document issues
# Create bug report in docs/artifacts/bug_reports/
# Schedule post-mortem review
```

---

## Next Steps After Phase 1

Once Phase 1 is complete and stable:

1. **Review Architecture Graph**: Open `architecture.png` and verify visual makes sense
2. **Test Strict Mode**: Try creating a file without ADS header, verify compilation fails
3. **Proceed to Phase 2**: Build high-performance resolver with fuzzy matching and caching
4. **Document Learnings**: Update plan with any adjustments made during implementation

---

**Status**: Ready to begin Phase 0 (archive operation)  
**First Action**: Execute Phase 0 commands, verify workspace clean, create git commit  
**Estimated Time to Phase 1 Complete**: 2 weeks from start  
**Critical Milestone**: End of Week 2 - Compiler functional, tests passing

**Recommendation**: Reference the [Quick Start Guide](docs/artifacts/implementation_plans/QUICKSTART-registry-automation.md) for detailed command sequences and success checklists throughout implementation.
