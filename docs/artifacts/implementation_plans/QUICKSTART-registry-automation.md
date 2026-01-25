# Quick Start: Registry Automation Implementation

**Implementation Plan**: [registry-automation-plan.md](./registry-automation-plan.md)  
**Updates Summary**: [registry-automation-UPDATES.md](./registry-automation-UPDATES.md)  
**Specification**: [specs/001-registry-automation/spec.md](../../../specs/001-registry-automation/spec.md)  
**Duration**: 6 weeks  
**Status**: Ready to Start  
**Version**: ADS v2.0 (Enhanced with strict mode, cycle detection, fuzzy matching)

## Overview

Transform the 440-line manually-maintained `registry.yaml` into an auto-generated artifact using the "nuclear approach" - archive all existing standards and rebuild with 100% v2.0 compliance. This eliminates synchronization errors, reduces agent context burden by 80%, enables safe standards refactoring, and adds intelligent features like cycle detection and fuzzy search.

**Key Enhancements in v2.0**:
- 🔒 Strict mode enforcement (zero untracked files)
- 🔄 Circular dependency detection
- 🎨 Visual architecture graphs (DOT format)
- 🔍 Fuzzy keyword matching (error-tolerant)
- ⚡ Binary caching (sub-10ms queries)
- 🤖 ML-aided migration tool

## Quick Commands Reference

```bash
# Phase 0: Nuclear Archive (REVERSIBLE - safe rollback)
mkdir -p AgentQMS/standards/_archive
mv AgentQMS/standards/tier*/*.yaml AgentQMS/standards/_archive/
cp AgentQMS/standards/registry.yaml AgentQMS/standards/_archive/registry.yaml.backup

# Phase 1: Create v2.0 schema and intelligent compiler
# (See Task 1.1, 1.2, 1.3 in plan for implementation)
uv run python AgentQMS/tools/sync_registry.py --dry-run  # Validate before write
uv run python AgentQMS/tools/sync_registry.py  # Compile with cycle detection
dot -Tpng AgentQMS/standards/architecture_map.dot -o architecture.png  # Visualize

# Phase 2: Build high-performance resolution tool
uv run python AgentQMS/tools/resolve_standards.py --task config_files
uv run python AgentQMS/tools/resolve_standards.py --path ocr/models/vgg.py
uv run python AgentQMS/tools/resolve_standards.py --keywords "hydra configuration"
uv run python AgentQMS/tools/resolve_standards.py --query "hidra config" --fuzzy  # Typo-tolerant
cat $(uv run python AgentQMS/tools/resolve_standards.py --task config_files --paths-only)  # Agent piping

# Phase 3: ML-aided migration
uv run python AgentQMS/tools/suggest_header.py _archive/naming-conventions.yaml  # Preview
uv run python AgentQMS/tools/suggest_header.py _archive/naming-conventions.yaml --apply  # Promote
uv run python AgentQMS/tools/migrate_to_ads_headers.py --limit 5  # Pilot batch

# Phase 4: Sync registry (with enhancements)
uv run python AgentQMS/tools/sync_registry.py  # Shows Pulse Delta + generates graph

# Phase 5: Validation & Parity Check
pytest AgentQMS/tests/test_sync_registry.py -v
pytest AgentQMS/tests/test_resolve_standards.py -v
uv run python AgentQMS/tools/verify_parity.py  # Must show 100% before rollout
```

## Success Checklist

### Phase 0: Nuclear Archive
- [ ] All 71 standards archived to `_archive/`
- [ ] Registry backed up to `_archive/registry.yaml.backup`
- [ ] Workspace clean (no files in tier* directories)
- [ ] Git commit created for safe rollback

### Phase 1: Foundation & Schema
- [ ] ADS v2.0 schema created with `id`, `dependencies`, `fuzzy_threshold` fields
- [ ] Schema validation passes (valid/invalid test cases)
- [ ] Registry compiler with strict mode functional
- [ ] Cycle detection prevents circular dependencies
- [ ] DOT graph generation works (`architecture_map.dot`)
- [ ] Semantic diff (Pulse Delta) displays correctly
- [ ] Unit tests pass (95%+ coverage)

### Phase 2: High-Performance Resolver
- [ ] Resolution tool returns correct standards (by task/path/keywords)
- [ ] Fuzzy matching works (typo-tolerant queries)
- [ ] Binary caching achieves sub-10ms queries
- [ ] Dependency expansion includes Tier 1 laws automatically
- [ ] Agent context loading integrated with resolver
- [ ] Token count reduced by 80% (1000 → 200 tokens measured)
- [ ] `--paths-only` flag enables shell piping

### Phase 3: ML-Aided Migration
- [ ] `suggest_header.py` tool built and tested
- [ ] Legacy trigger recovery from old registry works
- [ ] Tier inference from path is accurate
- [ ] 5 pilot files migrated successfully (Tier 1 priority)
- [ ] Registry compiles from pilot files without errors
- [ ] Visual graph shows pilot file dependencies

### Phase 4: Integration & Automation
- [ ] CLI commands functional (sync-registry, resolve-standards)
- [ ] Pre-commit hooks block invalid files
- [ ] CI pipeline validates on push
- [ ] Agent prompts updated with v2.0 requirements

### Phase 5: Full Migration & Validation
- [ ] All 71 files migrated (tier-by-tier: T1 → T2 → T3 → T4)
- [ ] **Parity verification: 100%** (zero data loss from legacy)
- [ ] Full test suite passes
- [ ] Performance benchmarks met (<5s sync, <10ms cached queries)
- [ ] Zero untracked files (SC-011 strict enforcement)
- [ ] Zero circular dependencies detected
- [ ] Documentation complete
- [ ] Registry frozen, monitoring active

## Key Deliverables by Phase

### Phase 0 (Pre-Week 1)
- All 71 standards archived to `_archive/`
- Backup created: `_archive/registry.yaml.backup`
- Clean workspace ready for v2.0 rebuild

### Phase 1 (Week 1-2)
- `AgentQMS/standards/schemas/ads-header.json` - v2.0 JSON schema
- `AgentQMS/tools/sync_registry.py` - Intelligent compiler (strict mode, cycles, graphs, diffs)
- `AgentQMS/tests/test_sync_registry.py` - Unit tests (95%+ coverage)
- `architecture_map.dot` - Visual dependency graph

### Phase 2 (Week 2-3)
- `AgentQMS/tools/resolve_standards.py` - High-performance resolver (fuzzy, cache, dependencies)
- Modified context loading to use resolver
- Token count benchmarks (80% reduction measured)
- `.ads_cache.pickle` - Binary cache for sub-10ms queries

### Phase 3 (Week 3-4)
- `AgentQMS/tools/suggest_header.py` - ML-aided migration tool
- `AgentQMS/tools/verify_parity.py` - Parity verification tool
- 5 pilot files with ADS v2.0 headers
- Pilot validation complete (registry compiles, graph renders)

### Phase 4 (Week 4-5)
- CLI commands (`aqms sync-registry`, `aqms resolve-standards`, `aqms suggest-header`)
- `.pre-commit-hooks.yaml` + validation hook
- `.github/workflows/validate-standards.yml`
- Updated agent prompts with v2.0 ADS requirements

### Phase 5 (Week 5-6)
- All 71 standards with ADS headers
- Complete test suite (unit + integration + performance)
- Full documentation
- Production deployment

## Critical Decision Points

### Week 2: After Pilot Migration
**Decision**: Are generated ADS headers accurate?
- If YES → Proceed to full migration
- If NO → Refine migration logic, retry pilot

### Week 4: After Tier 1 Migration
**Decision**: Is registry compilation stable?
- If YES → Continue tier-by-tier
- If NO → Pause, fix compiler issues

### Week 5: Before Full Rollout
**Decision**: Are all success metrics met?
- Registry sync <5s ✓
- Query latency <100ms ✓
- Token reduction 80% ✓
- All tests pass ✓
- If YES → Deploy to production
- If NO → Extended validation phase

## Rollback Procedure

If critical issues arise:

1. **Immediate Rollback**:
   ```bash
   git revert <migration-commit>
   git restore AgentQMS/standards/registry.yaml
   ```

2. **Restore Old Registry**:
   ```bash
   cp backups/registry.yaml.backup AgentQMS/standards/registry.yaml
   ```

3. **Disable Automation**:
   - Comment out pre-commit hooks
   - Disable CI validation workflow
   - Revert agent prompt changes

4. **Document Issues**:
   - Log errors encountered
   - Identify root cause
   - Create bug reports
   - Schedule post-mortem

## Getting Started

1. **Read the Full Plan**: [registry-automation-plan.md](./registry-automation-plan.md)
2. **Read the Spec**: [specs/001-registry-automation/spec.md](../../../specs/001-registry-automation/spec.md)
3. **Review Draft Research**: [__DEBUG__/2026-01-25_standards-update/additional-updates/draft-research.md](../../../__DEBUG__/2026-01-25_standards-update/additional-updates/draft-research.md)
4. **Start Phase 1**: Create `ads-header.json` schema (Task 1.1)

## Questions?

- Architecture questions → Review [ai-native-architecture.md](../../../AgentQMS/standards/tier1-sst/ai-native-architecture.md)
- Implementation questions → See detailed task descriptions in plan
- Migration questions → See Task 3.1 in plan
- Validation questions → See Phase 4 and 5 in plan

---

**Ready to Start**: Phase 1, Task 1.1 - Create ADS Header JSON Schema  
**Estimated First Milestone**: Week 2 - Registry compiler functional
