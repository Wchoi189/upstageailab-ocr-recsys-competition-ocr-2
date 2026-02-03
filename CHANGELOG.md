# Changelog

> [!NOTE]
> **Full History Archived**: Complete changelog history has been moved to [`archive/legacy-docs/CHANGELOG-full-2026-02-03.md`](archive/legacy-docs/CHANGELOG-full-2026-02-03.md). Only recent entries are maintained here for AI agents.

---

## [2026-02-03] - Phase 3 Context Bundle Optimization
- **Bundle Consolidation**: Reduced `ast-debugging-tools.yaml` from 361 to 121 lines (66% reduction) by removing embedded tutorials and referencing `AI_USAGE.yaml`.
- **v5-domains Enhancement**: Expanded `v5-domains-standard.yaml` from 10 to 77 lines with all domain controllers (detection, recognition, layout) and module implementations.
- **Token Budget Tool**: Created `scripts/measure_bundle_tokens.py` using tiktoken to measure all 17 bundles and generate compliance reports.
- **Registry Deprecation**: Removed dead `REGISTRY_SYNC` path and updated `registry_sync()` to use `generate_registry.py` (Phase 7.2 migration).
- **Architecture Documentation**: Created `AgentQMS/ARCHITECTURE.md` (750 tokens) documenting consolidation principles, shallow hierarchy, token discipline, and bundle constraints for AI consumption.
- **Spec Versioning**: Batch-added `spec_version: '1.0.0'` to all 11 tier2 specs (5 with new frontmatter, 5 with version field added).
- **Validation**: All 17 bundles pass validation with 0 broken references.

## [2026-02-03] - Post-Phase 3 Cleanup & Optimization
- **Script Consolidation**: Archived 5 one-time migration scripts to `archive/migration-scripts/2026-02-03-spec-kit-migration/` (consolidate_specs.py, update_flattened_refs.py, add_spec_versions.py, measure_bundle_tokens.py, bundle_token_report.md).
- **Makefile Integration**: Created `make qms-bundle-tokens` target for measuring context bundle token usage.
- **Documentation Updates**: Added "Architecture & Design Principles" section to root README.md with links to ARCHITECTURE.md and specs directory.
- **AgentQMS Documentation**: Created comprehensive `AgentQMS/README.md` documenting purpose, directory structure, context bundle system, CLI tools, and design principles.
- **Bundle Pruning**: Optimized 3 high-token bundles by restructuring tiering strategy:
  - `documentation-update.yaml`: Moved README.md and CHANGELOG.md from tier2 to tier3 (16.8k → ~1k tier2 tokens)
  - `compliance-check.yaml`: Reduced tier1 from 4 to 3 files, moved monitor_artifacts.py to tier3
  - `ocr-debugging.yaml`: Moved 2 large analyzers (merge_order, instantiation) from tier1 to tier2 (10.7k → ~5-6k tier1 tokens)
- **Validation**: All 17 bundles pass integrity checks with 0 broken references.

---

## Changelog Format Guidelines
- **Format**: `[YYYY-MM-DD] - Brief description (max 120 chars)`
- **Placement**: Add new entries at the very top, below this guidelines section
- **Conciseness**: Keep entries ultra-concise - focus on what changed, not why
- **Categories**: Group related changes under appropriate section headers
