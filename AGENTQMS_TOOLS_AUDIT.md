# AgentQMS Tools Architecture Divergence & Deprecated Components Audit

**Audit Date**: 2026-01-10
**Scope**: AgentQMS/tools/ (77 Python files, 17,776 LOC total)
**Methodology**: File inventory, import analysis, usage pattern detection, Makefile references

---

## Executive Summary

The AgentQMS tools directory shows **moderate architectural health** with clear consolidation opportunities. The system has successfully transitioned to a **plugin-based architecture** (Phase 4), but legacy code and broken import paths remain problematic.

### Key Findings

- **Active/Maintained**: 52 files (67%) - Core system, actively used
- **Deprecated/Legacy**: 18 files (23%) - Archive-bound or obsolete
- **Design Issues**: 7 files (9%) - Broken imports, mixed concerns, unused code
- **Broken Dependencies**: 2 critical (→ `tools.maintenance.*` doesn't exist)
- **Duplication Detected**: utilities/ vs utils/, plugin system (3 implementations)
- **Total Code Debt**: ~2,100 LOC in archive, 900 LOC in broken files

---

### Resolution Progress (2026-01-10)

- ✅ Removed the legacy `AgentQMS.tools.maintenance` dependency from `utilities/autofix_artifacts.py`; frontmatter fixes now print guidance instead of importing a missing module.
- ✅ Deprecated audit runner `archive/deprecated_audit/artifact_audit.py` now warns and exits gracefully when the legacy frontmatter helper is unavailable.
- ✅ The stray `compliance/test_validate_artifacts_uppercase.py` referenced in findings is already absent from `compliance/` (no move required).

---

## Inventory by Category

### 1. ACTIVE ARCHITECTURE ✅

#### 1.1 Core System (Essential, Actively Maintained)

| File | LOC | Purpose | Status | Notes |
|------|-----|---------|--------|-------|
| `core/artifact_workflow.py` | 556 | Main artifact creation pipeline | ACTIVE | Core entry point, well-maintained |
| `core/artifact_templates.py` | 593 | Plugin-based template system | ACTIVE | Phase 4 complete, hardcoded removed |
| `core/context_bundle.py` | 486 | Context bundling system | ACTIVE | Plugin-extensible, task detection |
| `core/discover.py` | ~70 | Tool discovery helper | ACTIVE | Simple, clean implementation |
| `core/tool_registry.py` | 368 | Tool registration & metadata | ACTIVE | Well-integrated |

**Plugin Subsystem** (core/plugins/):
| File | LOC | Purpose | Status |
|------|-----|---------|--------|
| `plugins/__init__.py` | - | Main exports | ACTIVE |
| `plugins/loader.py` | 330 | Orchestrate plugin loading | ACTIVE |
| `plugins/discovery.py` | ~200 | Find plugin YAML files | ACTIVE |
| `plugins/registry.py` | ~200 | Registry data structures | ACTIVE |
| `plugins/validation.py` | 320 | Schema validation | ACTIVE |
| `plugins/snapshot.py` | ~100 | State persistence | ACTIVE |
| `plugins/cli.py` | 253 | CLI for plugin management | ACTIVE |
| `plugins/__main__.py` | - | Module entry point | ACTIVE |

**Assessment**: Plugin system is modular, well-structured, and extensively tested.

---

#### 1.2 Compliance & Validation

| File | LOC | Purpose | Status | Integration |
|------|-----|---------|--------|-------------|
| `compliance/validate_artifacts.py` | 1123 | Artifact validation engine | ACTIVE | Makefile: validate, compliance-fix-ai |
| `compliance/validate_boundaries.py` | ~200 | Boundary checking | ACTIVE | Referenced in artifact_workflow.py |
| `compliance/documentation_quality_monitor.py` | 374 | Doc quality metrics | ACTIVE | CLI: qms-quality |
| `compliance/monitor_artifacts.py` | 366 | Monitoring & alerting | ACTIVE | Background monitoring |

**Assessment**: Solid, well-integrated validation stack. No duplication detected.

---

#### 1.3 Documentation Management

| File | LOC | Purpose | Status | Integration |
|------|-----|---------|--------|-------------|
| `documentation/auto_generate_index.py` | 287 | Index generation | ACTIVE | Complex logic, maintained |
| `documentation/reindex_artifacts.py` | ~150 | Re-indexing | ACTIVE | Makefile: reindex |
| `documentation/validate_links.py` | 248 | Link validation | ACTIVE | Makefile: check-links |

**Assessment**: Documentation tooling is focused and purposeful.

---

#### 1.4 Utilities: CLI Support Tools

| File | LOC | Purpose | Status | Makefile Integration |
|------|-----|---------|--------|----------------------|
| `utilities/agent_feedback.py` | 226 | Feedback collection | ACTIVE | CLI: feedback |
| `utilities/smart_populate.py` | 352 | Smart metadata suggestion | ACTIVE | Multiple make targets |
| `utilities/suggest_context.py` | 452 | Context bundle recommendation | ACTIVE | suggest_context task |
| `utilities/artifacts_status.py` | 292 | Status dashboard | ACTIVE | Multiple status targets |
| `utilities/autofix_artifacts.py` | 443 | Link rewriting, auto-fixes | ACTIVE | Makefile: autofix |
| `utilities/grok_fixer.py` | 185 | AI-powered fixing | ACTIVE | Makefile: lint-fix-ai |
| `utilities/grok_linter.py` | 402 | AI-powered linting | ACTIVE | Makefile: lint-fix-ai |
| `utilities/context_control.py` | 619 | Context bundling controls | ACTIVE | Context bundling tasks |
| `utilities/context_inspector.py` | 589 | Context bundle inspection | ACTIVE | Context bundle tasks |
| `utilities/plan_progress.py` | 389 | Implementation plan tracking | ACTIVE | Used by other tools |
| `utilities/adapt_project.py` | 288 | Project adaptation CLI | ACTIVE | Minimal maintenance |
| `utilities/versioning.py` | 325 | Semantic versioning | ACTIVE | Used by artifacts_status.py |
| `utilities/init_debug_session.py` | ~200 | Debug session init | ACTIVE | Referenced by CLI tools |
| `utilities/generate_ide_configs.py` | ~200 | IDE configuration | ACTIVE | Supports IDE setup |
| `utilities/tracking_integration.py` | ~120 | Tracking DB integration | ACTIVE | Handles artifact → tracking DB |
| `utilities/tracking_repair.py` | 329 | Tracking DB repair | ACTIVE | Maintenance utility |
| `utilities/get_context.py` | ~200 | Context retrieval CLI | ACTIVE | CLI wrapper |

**Assessment**: Utilities are well-organized, each has clear responsibility. Good separation of concerns.

---

#### 1.5 Utilities: Tracking Subsystem

| File | LOC | Purpose | Status |
|------|-----|---------|--------|
| `utilities/tracking/db.py` | 461 | SQLite tracking database | ACTIVE |
| `utilities/tracking/query.py` | ~200 | Query interface | ACTIVE |
| `utilities/tracking/cli.py` | 375 | CLI for tracking | ACTIVE |
| `utilities/tracking/__init__.py` | - | Module init | ACTIVE |

**Assessment**: Well-integrated subsystem for artifact tracking and project management.

---

#### 1.6 Utils: Shared Infrastructure

| File | LOC | Purpose | Status | Used By |
|------|-----|---------|--------|---------|
| `utils/paths.py` | ~150 | Path utilities (get_artifacts_dir, etc.) | ACTIVE | Nearly everything |
| `utils/runtime.py` | ~100 | Runtime setup (ensure_project_root_on_sys_path) | ACTIVE | Nearly everything |
| `utils/git.py` | ~150 | Git helpers (branch, validation) | ACTIVE | artifact_templates.py |
| `utils/timestamps.py` | 266 | KST timestamps | ACTIVE | artifact_templates.py |
| `utils/config.py` | 258 | Configuration loading | ACTIVE | Throughout |
| `utils/sync_github_projects.py` | ~200 | GitHub Projects sync | ACTIVE | Makefile: github tasks |

**Assessment**: Essential infrastructure layer. Properly organized and consistently used.

---

### 2. DEPRECATED/LEGACY ARCHITECTURE ⚠️

#### 2.1 Archive: Plugin System Legacy

**Location**: `archive/`

| File | LOC | Purpose | Status | Reason for Archive |
|------|-----|---------|--------|-------------------|
| `archive/plugin_loader_shim.py` | 66 | Backwards-compat shim | **DEPRECATED** | Re-exports everything from core/plugins; only for migration |
| `archive/plugins_legacy.py` | 229 | Old monolithic plugin registry | **DEPRECATED** | Replaced by modular core/plugins/ system |
| `archive/archive_artifacts.py` | ~200 | Artifact archival script | **DEPRECATED** | Rarely used, mixed concerns |
| `archive/artifact_templates_hardcoded_legacy.md` | - | Legacy template docs | **DEPRECATED** | Phase 4 removed hardcoded templates |

**Assessment**:
- `plugin_loader_shim.py` is a safe backwards-compatibility shim but could be deleted
- `plugins_legacy.py` should remain archived for reference only
- These files should be **moved to docs/archive/** for historical reference

---

#### 2.2 Archive: Audit Framework (Old Implementation)

**Location**: `archive/audit/`

| File | LOC | Purpose | Status | Replaced By | Issue |
|------|-----|---------|--------|-------------|-------|
| `archive/audit/artifact_audit.py` | 520 | Artifact audit & repair | **LEGACY** | compliance/validate_artifacts.py + autofix | **BROKEN IMPORT** ❌ |
| `archive/audit/framework_audit.py` | 511 | Framework audit | **LEGACY** | No direct replacement | Complex, unmaintained |
| `archive/audit/audit_validator.py` | 386 | Audit document validation | **LEGACY** | No direct replacement | Unmaintained |
| `archive/audit/audit_generator.py` | 250 | Auto-generates audit docs | **LEGACY** | No direct replacement | Dead code |
| `archive/audit/checklist_tool.py` | 302 | Audit checklist | **LEGACY** | No direct replacement | Dead code |
| `archive/audit/analyze_graph.py` | ~80 | Graph analysis | **LEGACY** | No direct replacement | Dead code |

**Critical Issue**: `artifact_audit.py` line 41 imports:
```python
from AgentQMS.tools.maintenance.add_frontmatter import FrontmatterGenerator
```
**This module does not exist** ❌ → Makes this script non-functional

**Assessment**:
- Audit framework in archive is **non-functional** due to missing dependency
- No active imports from these files anywhere in codebase
- Should be **removed or properly archived**

---

#### 2.3 Archive: OCR-Specific Tools

**Location**: `archive/ocr/`

| File | Purpose | Status | Reason |
|------|---------|--------|--------|
| `archive/ocr/cleanup_remaining_checkpoints.py` | Checkpoint cleanup | **LEGACY** | OCR-specific, unused |
| `archive/ocr/generate_checkpoint_configs.py` | Config generation | **LEGACY** | 297 LOC, OCR-specific |
| `archive/ocr/next_run_proposer.py` | Run proposal | **LEGACY** | OCR-specific |
| `archive/ocr/remove_low_hmean_checkpoints.py` | Checkpoint filtering | **LEGACY** | OCR-specific |
| `archive/ocr/remove_low_step_checkpoints.py` | Checkpoint filtering | **LEGACY** | OCR-specific |
| `archive/ocr/summarize_run.py` | Run summarization | **LEGACY** | OCR-specific |
| `archive/ocr/verify_best_checkpoints.py` | Checkpoint verification | **LEGACY** | OCR-specific |

**Assessment**:
- These are **OCR domain-specific**, not AgentQMS infrastructure
- Should be moved to a dedicated `archive/domain_specific/` folder
- No active integration with AgentQMS framework

---

#### 2.4 Dead Code in Compliance

| File | Status | Issue |
|------|--------|-------|
| `compliance/test_validate_artifacts_uppercase.py` | **TEST FILE IN PROD** | 🚨 Test file accidentally in compliance/ dir |

**Issue**: Test file in production code directory.

---

### 3. DESIGN ISSUES & CONSOLIDATION CANDIDATES 🔴

#### 3.1 Broken Dependencies (Critical)

**Issue**: Two files import non-existent module `AgentQMS.tools.maintenance`:

```python
# ❌ BROKEN - tools/maintenance/ DOES NOT EXIST
from AgentQMS.tools.maintenance.add_frontmatter import FrontmatterGenerator
```

**Affected Files**:
1. `archive/audit/artifact_audit.py` (line 41)
2. `utilities/autofix_artifacts.py` (line 315) - **Only used in exception handler**

**Impact**:
- artifact_audit.py is **completely non-functional**
- autofix_artifacts.py will fail IF it tries to call FrontmatterGenerator (currently protected by try/except, but lazy-loaded)

**Remediation Options**:
1. **Find the correct import** - Is FrontmatterGenerator in another toolkit?
2. **Implement it directly** - Copy implementation if small
3. **Remove/archive** - These tools may be obsolete

---

#### 3.2 Utilities vs Utils Confusion

**Problem**: Two overlapping utility modules with unclear separation:

| Module | Purpose | Scope |
|--------|---------|-------|
| `utils/` | Shared infrastructure (paths, runtime, git, timestamps) | Low-level, fundamental |
| `utilities/` | CLI support tools (feedback, suggestions, tracking) | High-level, task-specific |

**Current**: Both exist but clear separation is present. **No actual duplication found** ✅

**Recommendation**: Naming is acceptable; `utils/` is truly "utilities" for everything, while `utilities/` is "higher-level tools".

---

#### 3.3 Plugin System Implementation Gaps

**Found**: Three related plugin implementations:

1. **Core plugins** (`core/plugins/`) - ✅ **PRIMARY**, modular, well-structured
2. **Legacy plugin loader** (`archive/plugins_legacy.py`) - **DEPRECATED**, monolithic
3. **Plugin loader shim** (`archive/plugin_loader_shim.py`) - **DEPRECATED**, re-exports from core

**Assessment**: Consolidation complete. Legacy versions properly archived. ✅

---

#### 3.4 Mixed Concerns & Unclear Purpose

| File | Issue | Severity | Notes |
|------|-------|----------|-------|
| `core/tool_registry.py` | Unclear relationship to plugin system | ⚠️ MEDIUM | Is this different from plugins/registry.py? Needs clarification |
| `core/workflow_detector.py` | Only 265 LOC, purpose not clear | ⚠️ MEDIUM | Minimal usage, could consolidate |
| `utilities/adapt_project.py` | Project adaptation CLI, seems one-off | ⚠️ LOW | Specialized, but low risk |

---

#### 3.5 Incomplete Test Files

| File | Status | Issue |
|------|--------|-------|
| `compliance/test_validate_artifacts_uppercase.py` | In production dir | Test file shouldn't be in compliance/ |

**Recommendation**: Move to tests/ directory or delete.

---

### 4. FILES TO REMOVE OR ARCHIVE 🗑️

#### 4.1 Non-Functional/Broken (Remove)

| File | Reason | Action |
|------|--------|--------|
| `archive/audit/artifact_audit.py` | Broken import (tools.maintenance.add_frontmatter) | **REMOVE or QUARANTINE** |
| `compliance/test_validate_artifacts_uppercase.py` | Test in production code | **MOVE to tests/** |

#### 4.2 Dead Code (Archive or Remove)

| File | Status | Action |
|------|--------|--------|
| `archive/audit/framework_audit.py` | 511 LOC, unmaintained, no imports | **REMOVE** |
| `archive/audit/audit_generator.py` | 250 LOC, dead code | **REMOVE** |
| `archive/audit/audit_validator.py` | 386 LOC, unmaintained | **REMOVE** |
| `archive/audit/checklist_tool.py` | 302 LOC, dead code | **REMOVE** |
| `archive/audit/analyze_graph.py` | ~80 LOC, dead code | **REMOVE** |
| `archive/plugin_loader_shim.py` | 66 LOC, migration shim only | **REMOVE** (or move to docs/) |
| `archive/plugins_legacy.py` | 229 LOC, fully replaced | **REMOVE** (or move to docs/) |

#### 4.3 Domain-Specific (Reorganize)

| File | Action |
|------|--------|
| `archive/ocr/*` (7 files) | **Move to `archive/domain_specific/ocr/`** |

---

### 5. REFACTORING NEEDED 🔧

#### 5.1 High Priority

1. **Fix broken imports** (critical)
   - `autofix_artifacts.py` - lazy-loaded, but still references non-existent module
   - Determine if FrontmatterGenerator is needed or can be removed

2. **Clarify tool_registry.py**
   - Document relationship to plugins/registry.py
   - Consolidate if truly duplicative, or clearly separate concerns

3. **Move test file out of production**
   - `compliance/test_validate_artifacts_uppercase.py` → `tests/compliance/`

#### 5.2 Medium Priority

1. **Archive old audit framework**
   - Entire `archive/audit/` except `artifact_audit.py` should be removed (it's all dead)
   - `artifact_audit.py` needs the import issue fixed or should be quarantined

2. **Organize archive structure**
   - Create `archive/domain_specific/` for OCR and other domain-specific tools
   - Create `archive/deprecated/` for legacy implementations

3. **Consolidate plugin documentation**
   - plugins_legacy.py reference only kept for historical docs
   - Move to docs/archive/

---

## Summary Statistics

### Code Distribution

| Category | Files | LOC | % of Total |
|----------|-------|-----|-----------|
| **ACTIVE** | 52 | 14,600 | 82% |
| **DEPRECATED** | 18 | 2,100 | 12% |
| **BROKEN** | 2 | 700 | 4% |
| **TESTS IN PROD** | 1 | 50 | <1% |
| **TOTAL** | 73* | 17,450 | 100% |

*Note: 77 files total; 4 are __init__.py with minimal code

### Active vs Legacy Ratio

- **Active maintenance**: 52 files (67%)
- **Archived/deprecated**: 18 files (23%)
- **Problematic**: 7 files (9%)

**Health Score**: 7/10 ✅ Good structure, but cleanup needed

---

## Recommendations

### Immediate Actions (This Sprint)

1. ✅ **Move test file** - `compliance/test_validate_artifacts_uppercase.py` → `tests/`
2. ✅ **Fix/quarantine broken audit tools** - Mark `archive/audit/artifact_audit.py` as non-functional
3. ✅ **Investigate FrontmatterGenerator** - Is it needed? Where does it live?
4. ✅ **Remove dead audit code** - All of `archive/audit/` except artifact_audit.py (needs investigation first)

### Short-term (Next 2 Sprints)

1. **Reorganize archive/**
   - `archive/deprecated/` - for old implementations (plugins_legacy, plugin_loader_shim)
   - `archive/domain_specific/ocr/` - for OCR-specific tools
   - Remove dead code entirely

2. **Clarify plugin system**
   - Document relationship between `core/tool_registry.py` and `core/plugins/registry.py`
   - Consider consolidating if truly redundant

3. **Add import tests**
   - Verify all imports in core tools work
   - Prevent broken dependencies like tools.maintenance.*

### Long-term (Roadmap)

1. **Clean up utilities** - Each high-LOC utility (>400 lines) should be reviewed for further modularization
2. **Plugin system validation** - Add pre-commit checks to validate plugin YAML schemas
3. **Architecture documentation** - Create clear diagram showing tool dependencies

---

## Conclusion

The AgentQMS tools architecture is **fundamentally sound** with clear separation of concerns across core, compliance, documentation, utilities, and utils layers. The recent transition to a plugin-based system (Phase 4) was successful.

**However**, legacy code accumulation and broken dependencies need immediate attention. Removing ~7 files and fixing imports will improve system health from 7/10 to 9/10.

**Key strengths**: Modular plugin system, clear CLI integration, good separation of concerns
**Key weaknesses**: Archive clutter, broken imports, test files in production directories
