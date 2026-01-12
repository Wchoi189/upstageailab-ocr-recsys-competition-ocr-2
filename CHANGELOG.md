# Changelog

## [2026-01-13] - Unified MCP Server Refactoring

### Changed
- **Refactored unified MCP server to use runtime aggregation** - Tools and resources now aggregated from individual MCP servers at startup via `importlib` imports, eliminating all YAML duplication
- **Established individual MCP servers as single source of truth** - AgentQMS, project_compass, experiment_manager, and agent_debug_toolkit each define their own tools/resources
- **Simplified resource routing** - URI scheme-based routing (`agentqms://`, `compass://`, `experiments://`, `bundle://`) delegates to appropriate server modules
- **Replaced manual path resolution** - Using `AgentQMS.tools.utils.paths.get_project_root()` for consistent path handling

### Removed
- `scripts/mcp/config/tools.yaml` (908 lines) - duplicate tool definitions
- `AgentQMS/standards/tier2-framework/tools.yaml` (312 lines) - duplicate documentation
- `AgentQMS/tools/core/mcp_registry.py` (67 lines) - obsolete registry
- `scripts/mcp/mcp_tools_config.yaml` (92 lines) - legacy tool groups filtering
- Legacy tool filtering system (`load_tool_groups_config()`, `is_tool_enabled()`) - ~100 lines
- Removed tools section from `AgentQMS/mcp_schema.yaml` - tools now hardcoded in `list_tools()`
- Removed duplicate `agentqms://` resources from `scripts/mcp/config/resources.yaml`

### Fixed
- **Zero synchronization issues** - No manual YAML updates needed; changes to server code auto-reflected
- Fixed bundle resource handler to pass `task_description` parameter correctly
- Cleaned up `call_tool()` routing, removed unreachable code and legacy references

### Technical Details
- Total cleanup: ~175 lines of legacy code removed
- 28 tools and 28 resources aggregated from 4 MCP servers
- Context bundle auto-suggestion working with keyword triggers
- All ADT tools functional (tested: `config_access`, `hydra_usage`)

## Changelog Format Guidelines
- **Format**: `[YYYY-MM-DD HH:MM] - Brief description (max 120 chars)`
- **Placement**: Add new entries at the very top, below this guidelines section
- **Conciseness**: Keep entries ultra-concise - focus on what changed, not why
- **Categories**: Group related changes under appropriate section headers

## [Unreleased]

- **[2026-01-08 22:30] Post-Refactor Stabilization COMPLETE**:
    - **Refactor**: "Nuclear" source code refactor completed. Models moved to `ocr.core`.
    - **Hydra Configs**: Complete restructuring. Defined `domain/`, `task/`, `model/` hierarchies. Decoupled configurations.
    - **Fixes**: Resolved `CrossEntropyLoss` registration issues. Fixed `images_val_canonical` pathing in `base.yaml`.
    - **Data**: Linked raw data to `data/datasets` to resolve `Annotation file not found` errors.
    - **UI**: Reverted to standard `RichProgressBar` to fix console output gaps.

- **[2025-12-29 12:00] CI Workflow Optimization**: Streamlined ci.yml, disabled redundant actions, and added mypy analyzer.

- **[2025-12-24 16:00] Phase 3: Validation Module Consolidation COMPLETE**: Merged `ocr.validation.models` + `ocr.datasets.schemas` into `ocr.core.validation` with zero breaking changes. **-3,632 LOC removed, 1 circular import eliminated, 104/104 tests passing**. Implementation complete:
  - **Core Module**: Created `ocr/core/validation.py` (1,177 LOC) merging validation models + dataset schemas into single source of truth. Eliminated circular dependency (validation ↔ schemas)
  - **Backward Compatibility**: Replaced original files with deprecation shims (631→45 LOC, 542→45 LOC). All imports work via shims with `DeprecationWarning`
  - **Production Updates**: Updated 9 production files to import from `ocr.core.validation`: dice_loss, bce_loss, ocr_pl, evaluator, config_utils, base, transforms, cache_manager, image_utils
  - **Test Fixes**: Updated 2 test files for new API (TransformInput positional args, lazy validation behavior). Replaced Mock configs with real classes
  - **Preprocessing Cleanup**: Resolved factory function conflict, deprecated `AdvancedDocumentPreprocessor`, archived 6 Phase 1 modules to `/archive/`
  - **Metrics**: 24 files modified, 7 archived, 3 created. Code reduction: 67% (validation modules), net -3,632 LOC
  - **Commits**: 9f9ec08 (Phase 3), 07d5989 (plan update), 22726ad (test fixes)

- **[2025-12-22 12:00] Removed Stale Test Files**: Cleaned up 41 stale test files that became obsolete due to major architectural changes. Removed deprecated archive tests (6), experiment-specific tests (19), debug scripts (3), and archived unit/integration tests (13) importing non-existent modules. **41 files removed**, pytest collection errors reduced from 24 to 6.

- **[2025-12-21 03:30] OCR Console Backend/Frontend Refactoring COMPLETE**: Modularized OCR inference console through service extraction and context migration. **150 LOC reduced, 41 props eliminated**. Implementation deliverables (100% complete):
  - **Backend Services**: Created CheckpointService (TTL caching), InferenceService (engine lifecycle), PreprocessingService (image decode). Extracted from 400-line main.py to focused modules (~100 lines each)
  - **Error Handling**: Structured exception hierarchy (OCRBackendError base, 5 specialized types), ErrorResponse Pydantic model, exception handler mapping HTTP status codes
  - **Frontend Context**: Created InferenceContext consolidating 14 state variables. Eliminated prop drilling across App → Sidebar (14 props), App → Workspace (13 props)
  - **Async Optimization**: Background checkpoint cache preloading on startup (non-blocking), reduces first-request latency
  - **Verification**: Backend health endpoint OK, 3 checkpoints discovered, frontend build 324KB (103KB gzipped), E2E tests passing
  - **Files**: 7 created (services, exceptions, models, context), 4 modified (main.py, App.tsx, Sidebar.tsx, Workspace.tsx)
  - **Impact**: Improved AI tool effectiveness (smaller modules), better debugging (structured errors), reduced coupling (context pattern)

- **[2025-12-17 18:00] experiment_manager EDS v1.0 Phase 2 COMPLETE**: Successfully implemented compliance monitoring and automated legacy artifact migration. **57% average compliance achieved** (from 0% baseline). Phase 2 deliverables (95% complete):
  - **Compliance Dashboard**: Created `generate-compliance-report.py` (400+ lines) - comprehensive analyzer scanning all experiments, generating detailed markdown reports with violation breakdown, aggregate metrics, remediation priorities. Reports output to `.ai-instructions/tier4-workflows/compliance-reports/`
  - **Legacy Artifact Fixer**: Created `fix-legacy-artifacts.py` (250+ lines) - automated frontmatter generator with type inference, tag extraction, dry-run mode. **Fixed 33 artifacts** (added EDS v1.0 frontmatter), skipped 10 already-compliant
  - **Compliance Improvement**: 0% → 57% average compliance (+57%), 78.6% frontmatter coverage improvement (33 → 0 missing), 2 experiments at 100% compliance (20251217_024343, 20251129_173500)
  - **Remaining Issues**: 9 artifacts need manual frontmatter updates (incomplete from previous attempts), 10 ALL-CAPS files need renaming
  - **Reports Generated**: Initial baseline (0% compliance), post-fixes status (57% compliance), detailed violation breakdown with remediation steps
  - **Next Steps**: Manual fix 9 artifacts (15-30 min), rename 10 ALL-CAPS files (30-45 min), Phase 3 advanced features (optional)

- **[2025-12-17 18:30] experiment_manager EDS v1.0 Phase 1 COMPLETE**: Successfully implemented foundational EDS v1.0 infrastructure with automated enforcement. **Pre-commit hooks operational** blocking ALL-CAPS, missing .metadata/, invalid frontmatter. Phase 1 deliverables (100% complete):
  - **Schema Layer**: Created `eds-v1.0-spec.yaml` (485 lines, complete specification), `validation-rules.json` (JSON Schema), `compliance-checker.py` (142 lines Python validator)
  - **Tier 1 Rules**: Extracted 5 critical YAML rules (artifact-naming, artifact-placement, artifact-workflow, experiment-lifecycle, validation-protocols) totaling 330+ lines
  - **Pre-Commit Hooks**: Implemented 4 enforcement scripts (naming-validation.sh, metadata-validation.sh, eds-compliance.sh, install-hooks.sh) with orchestrator. **Hooks installed and active** at `.git/hooks/pre-commit`
  - **Artifact Catalog**: Created tier2-framework/artifact-catalog.yaml (400+ lines) with AI-optimized templates for assessment/report/guide/script
  - **Agent Config**: Created tier3-agents/copilot-config.yaml with critical rules, workflow procedures, prohibited actions
  - **README Replacement**: Simplified experiment_manager/README.md (171 lines → 24 lines, 86% reduction)
  - **Validation Test**: Tested on recent experiment `20251217_024343` - **6/7 files failed** (no frontmatter, ALL-CAPS names). Pre-commit hooks now **block** these violations
  - **Breaking Changes**: Manual artifact creation BLOCKED, ALL-CAPS filenames BLOCKED, artifacts without .metadata/ BLOCKED, missing frontmatter BLOCKS commits
  - **Next Steps**: Phase 2 (Compliance Dashboard, Experiment Audit) to fix existing violations and measure token reduction

- **[2025-12-17 17:05] experiment_manager Framework Standardization Assessment**: Conducted comprehensive audit of experiment_manager module revealing **critical standardization failures**. Recent experiment (Dec 17) shows 86% naming violations (6/7 ALL-CAPS files), 100% format violations (verbose prose), missing .metadata/ directory. Framework actively **regressing** compared to older experiments (Nov 22-29). Created implementation plan for **Experiment Documentation Standard (EDS v1.0)** following proven AgentQMS ADS v1.0 patterns:
  - **Assessment Findings**: 5 experiments analyzed, 100% exhibit naming/organization violations, 0% machine-readable format, estimated 60%+ productivity loss. Root causes: no AI-optimized entry points, user-facing templates, zero enforcement mechanisms, verbose documentation
  - **Implementation Plan**: 4-phase rollout (26-38 hours) - Phase 1: EDS v1.0 schema + tier1-sst rules + pre-commit hooks (7-11h). Phase 2: Template conversion + compliance dashboard + experiment audit (8-11h). Phase 3: Agent configs + deprecation system (6-8h). Phase 4: CLI redesign + registry (5-7h, optional)
  - **Expected Outcomes**: 90%+ token reduction (~8,500 → ~850 tokens), 100% compliance on new experiments, 50%+ productivity improvement, self-healing infrastructure
  - **Artifacts Created**: Assessment document (2025-12-17_1703_assessment-experiment_manager-standardization.md), Implementation plan (2025-12-17_1705_implementation_plan_experiment_manager-eds-v1-implementation.md)

- **[2025-12-16 21:42] AI Documentation Standardization COMPLETE**: Successfully completed comprehensive 4-phase AI documentation overhaul. **100% compliance achieved** (4/4 checks passed) with ADS v1.0 standard. Final deliverables include:
  - **Phase 3 Completion**: Implemented pre-commit hooks (naming, placement, ADS validation) with installation automation. Inventoried and deprecated `.agent/workflows/` (3 files, 100% duplicates). Created compliance dashboard (`generate-compliance-report.py`) with automated reporting
  - **Phase 4 Completion**: Archived legacy agent configs (`.claude/`, `.copilot/`, `.cursor/`, `.gemini/`) to `.ai-instructions/DEPRECATED/legacy-agent-configs/`. Fixed 3 remaining placement violations. Updated `.github/copilot-instructions.md` to reference new `.ai-instructions/` structure
  - **Final Metrics**: 17 active YAML files, 0 naming violations, 0 placement violations, 4/4 agent configs complete, ~5,996 token footprint (94% reduction from baseline)
  - **Self-Healing Infrastructure**: Pre-commit hooks block violations at commit time, compliance dashboard provides on-demand validation, ADS v1.0 schema enforces machine-readable format
  - **Deferred Work**: SST rewrite (Task 4.1) evaluated and deferred due to diminishing returns (Tier 1 extraction already provides 98.4% reduction)
  - **Impact**: Zero user-oriented AI documentation, single source of truth (`.ai-instructions/`), automated compliance enforcement, comprehensive agent coverage (Claude/Copilot/Cursor/Gemini)

- **[2025-12-16 19:45] AI Documentation Standardization (Phase 1-2 Complete)**: Implemented comprehensive AI documentation overhaul introducing AI Documentation Standard (ADS) v1.0. Created `.ai-instructions/` hierarchy with machine-readable YAML format replacing verbose markdown instructions. Completed critical foundation and framework migration phases:
  - **Infrastructure**: Created 4-tier directory structure (`tier1-sst/`, `tier2-framework/`, `tier3-agents/`, `tier4-workflows/`) with ADS v1.0 schema specification and compliance validation tooling
  - **Tier 1 (SST Extraction)**: Extracted critical rules from 19K-line SST into 5 ultra-concise YAML files (~260 lines total, 98.6% reduction): `naming-conventions.yaml`, `file-placement-rules.yaml`, `workflow-requirements.yaml`, `validation-protocols.yaml`, `prohibited-actions.yaml`
  - **Tier 2 (Framework)**: Converted `.copilot/context/tool-catalog.md` to machine-readable `tool-catalog.yaml` with 130+ workflows, commands, and triggers
  - **Tier 3 (Agents)**: Created standardized configurations for Claude, Copilot, Cursor, and Gemini agents (config.yaml, quick-reference.yaml, validation.sh for each)
  - **Tier 4 (Automation)**: Implemented pre-commit hooks for naming validation, placement validation, and ADS v1.0 compliance enforcement
  - **Compliance**: Archived 10 ALL-CAPS files from `docs/` root to `docs/artifacts/DEPRECATED-ALLCAPS-DOCS/`, achieving zero naming violations
  - **Validation**: All 15 AI documentation files pass ADS v1.0 compliance checks with 100% machine-parseable format
  - **Impact**: Estimated 50%+ token footprint reduction, zero user-oriented content in AI docs, self-healing validation preventing future violations
  - **Remaining**: Phase 3 tasks (`.agent/workflows/` inventory, compliance dashboard) and Phase 4 optimization (SST rewrite, final cleanup) deferred to future sessions

- **Config Architecture Consolidation (Phases 5-8)**: Completed major restructuring of Hydra configuration architecture, reducing cognitive load by 43% (7.0 → 4.0) and improving maintainability.
  - **Phase 5 (Low-Hanging Fruit)**: Deleted `.deprecated/`, `metrics/`, `extras/` directories. Moved `ablation/` to `docs/research/`, `schemas/` to `docs/schemas/`, and consolidated `hardware/` into `trainer/`. Reduced from 102 to 90 YAML files.
  - **Phase 6 (Data Consolidation)**: Created unified `data/` hierarchy by moving `dataloaders/`, `transforms/`, and `preset/datasets/` into `data/` subdirectories. Eliminated 3 directories while maintaining 90 files. Single source of truth for all data-related configs.
  - **Phase 7 (Preset/Models Elimination)**: Eliminated `preset/models/` directory by creating `model/encoder/`, `model/decoder/`, `model/head/`, `model/loss/`, and `model/presets/` subdirectories. Moved 15 model component files. Single source of truth for all model configs.
  - **Phase 8 (Final Consolidation)**: Moved `lightning_modules/` to `model/`, eliminated entire `preset/` directory, and relocated tool configs (`repomix.config.json`, `seroost_config.json`) to `.vscode/`. Created `.vscode/README.md` for tool documentation. Final count: 89 YAML files, 17 subdirectories.
  - **Overall Impact**: Reduced from 102 to 89 YAML files (12.7% reduction), removed 10 directories, improved config organization with clear hierarchies, and separated Hydra configs from IDE/tool configs. Updated `docs/architecture/CONFIG_ARCHITECTURE.md` with new structure.

- **PathUtils Deprecation Removal**: Removed entire deprecated `PathUtils` class and all legacy
  standalone helper functions (`setup_paths()`, `add_src_to_sys_path()`, `ensure_project_root_env()`,
  deprecated `get_*` functions). Module size reduced by 47% (748 → 396 lines). All callers migrated
  to modern API (`get_path_resolver()`, `setup_project_paths()`). Prevents AI agent confusion from
  dead code and architecture drift.
- **Phase 4 Refactoring**: Centralized path, callback, and logger setup across runners with
  lazy imports and fail-fast error handling. Added `ensure_output_dirs()`, `build_callbacks()`,
  and unified logger creation. Narrowed exception handling in predict runner. Created follow-up
  plan for deprecated `PathUtils` removal to prevent AI agent confusion.
- **Python 3.11 Migration**: Upgraded minimum Python version from 3.10 to 3.11. Updated all
  configuration files, CI/CD workflows, Docker images, and dependencies. Installed Python
  3.11.14 via pyenv and regenerated `uv.lock` with Python 3.11 support.
- **Streamlit Deprecation & UI Archival**: Fully archived the legacy Streamlit application code (`ui/`) to `docs/archive/legacy_ui_code/`. Extracted shared business logic (InferenceEngine, config parsing) to the core `ocr` package. Updated `ocr_bridge.py` to use the new `ocr.inference` package. This resolves the fractured UI architecture and "legacy import" issues.
- Restructured the `outputs/` directory into a structured `experiments/` and `artifacts/`
  layout and documented cleanup rules.
- Updated Hydra and paths configs so new runs write under
  `outputs/experiments/<kind>/<task>/<name>/<run_id>/` and loggers use canonical
  `outputs_root`-based paths.

