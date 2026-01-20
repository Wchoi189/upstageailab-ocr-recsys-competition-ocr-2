# Changelog

All notable changes to AgentQMS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-01-20

### 🚨 BREAKING CHANGES - Nuclear Refactor

This release represents a complete architectural overhaul of AgentQMS, aggressively removing legacy systems to prevent split-brain syndrome and configuration drift.

#### Removed (Legacy System Deletion)

**Deleted Tool Scripts:**
- `AgentQMS/tools/core/artifact_workflow.py` - Replaced by `qms artifact` subcommand
- `AgentQMS/tools/compliance/validate_artifacts.py` - Replaced by `qms validate` subcommand
- `AgentQMS/tools/compliance/monitor_artifacts.py` - Replaced by `qms monitor` subcommand
- `AgentQMS/tools/utilities/agent_feedback.py` - Replaced by `qms feedback` subcommand
- `AgentQMS/tools/compliance/documentation_quality_monitor.py` - Replaced by `qms quality` subcommand

**Deleted Archived Files:**
- `AgentQMS/standards/.archive/INDEX.yaml` - Consolidated into registry.yaml
- `AgentQMS/standards/.archive/standards-router.yaml` - Consolidated into registry.yaml
- `AgentQMS/standards/.archive/` - Entire directory removed

**Removed Makefile Commands:**
- All legacy Makefile artifact commands removed from AGENTS.yaml
- Only `qms` CLI commands supported going forward

### Added

**New Unified CLI:**
- `AgentQMS/bin/qms` - Single entry point for all AgentQMS operations
  - `qms artifact` - Create, validate, and manage artifacts
  - `qms validate` - Run validation checks
  - `qms monitor` - Compliance monitoring
  - `qms feedback` - Report issues
  - `qms quality` - Documentation quality checks
  - `qms generate-config` - Path-aware configuration generation

**New Configuration System:**
- `AgentQMS/standards/registry.yaml` (419 lines) - Unified standards registry
  - Consolidates INDEX.yaml and standards-router.yaml
  - Supports both keyword and path_pattern triggers
  - Enables path-aware discovery (85% token reduction)

**Documentation:**
- `AgentQMS/README.md` - Comprehensive single source of truth
  - Quick start guide
  - Command reference
  - Architecture overview
  - Performance metrics
- `AgentQMS/MIGRATION_GUIDE.md` - Legacy to v0.3.0 migration guide
- `AgentQMS/DEPRECATION_PLAN.md` - Marked as COMPLETED (nuclear refactor)
- `NUCLEAR_REFACTOR_HANDOVER.md` - Detailed execution plan documentation

**Monitoring Tools:**
- `AgentQMS/bin/monitor-token-usage.py` - Token usage analysis tool
  - Reports 85.6% average token reduction
  - Shows path-aware discovery impact

**Global CLI Access:**
- Created symlink `/usr/local/bin/qms` for system-wide access
- `qms --version` returns "AgentQMS CLI v0.3.0"

### Changed

**Configuration Files:**
- `AgentQMS/.agentqms/settings.yaml`
  - Removed ALL legacy tool mappings
  - Only `qms` CLI entry remains
  - Added comprehensive subcommand documentation

- `AgentQMS/AGENTS.yaml`
  - Updated `resources.readme` to point to new README.md
  - Changed `resources.standards_index` to `resources.standards_registry`
  - Removed all legacy Makefile command references
  - Added `qms` CLI command examples
  - Added performance notes (85% token reduction)

- `AgentQMS/bin/Makefile`
  - Fixed `plugins-snapshot` target to use system Python with graceful error handling
  - Resolves CI workflow failures

**Enhanced ConfigLoader:**
- `AgentQMS/tools/utils/config_loader.py`
  - Added `resolve_active_standards()` - Path-aware standard resolution
  - Added `generate_effective_config()` - Dynamic context injection
  - Implements fnmatch pattern matching for path_patterns

**OCR Domain Compatibility Shims (Updated with Deprecation Warnings):**
- `ocr/data/datasets/db_collate_fn.py` - Added v0.4.0 removal warning
- `ocr/core/utils/geometry_utils.py` - Added v0.4.0 removal warning
- `ocr/core/utils/polygon_utils.py` - Added v0.4.0 removal warning
- `ocr/core/inference/engine.py` - Added v0.4.0 removal warning
- `ocr/core/evaluation/evaluator.py` - Added v0.4.0 removal warning
- `ocr/core/lightning/ocr_pl.py` - Restored from deprecated state
- `experiment_manager/src/etk/compass.py` - Added v0.4.0 removal warning

All shims now emit `DeprecationWarning` with clear migration paths.

### Fixed

**CI Workflow Failures:**
- Fixed "AgentQMS Compliance" workflow - Makefile plugins-snapshot error
- Fixed "Tests" workflow - 16 import errors from OCR domain refactor
- Fixed "Validate Artifacts" workflow - Same Makefile error
- All CI workflows now pass successfully

### Performance

**Token Usage Optimization:**
- **87.5% reduction** in standards loaded per session (24 → 3 average)
- **85.6% reduction** in total context tokens (~12,500 → ~1,700)
- Path-aware discovery matches current directory to relevant standards only
- Eliminates static loading of all 24 standards files

### Migration Notes

**For AI Agents:**
- Update all tool invocations to use `qms` CLI
- Remove references to legacy Python scripts
- Use `qms generate-config --path $(pwd)` for path-aware context

**For Developers:**
- Legacy tools no longer available - use `qms` CLI exclusively
- Update OCR imports to use new domain paths before v0.4.0
- Refer to `AgentQMS/README.md` for complete command reference

**Breaking Change Timeline:**
- **v0.3.0 (NOW):** Legacy tools deleted, compatibility shims emit warnings
- **v0.4.0 (FUTURE):** Compatibility shims will be removed entirely

### Validation

- ✅ All 69 file paths in registry.yaml validated (100% valid)
- ✅ qms CLI tested and functional
- ✅ Legacy files confirmed deleted
- ✅ Documentation updated and consistent
- ✅ CI workflows passing
- ✅ Path-aware discovery verified

---

## [0.2.0] - Previous Release

### Initial Implementation
- Original AgentQMS framework with separate tool scripts
- INDEX.yaml and standards-router.yaml for discovery
- Makefile-based artifact workflows
- Static context loading (all standards always loaded)

[0.3.0]: https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/compare/v0.2.0...v0.3.0
