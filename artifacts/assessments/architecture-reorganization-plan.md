---
title: "Architecture Reorganization Plan"
author: "ai-agent"
date: "2025-11-09"
status: "draft"
tags: ["architecture", "reorganization", "documentation", "artifacts", "assessment"]
---

# Architecture Reorganization Plan

**Status:** Draft
**Date:** 2025-11-09
**Priority:** High
**Impact:** Critical - Resolves confusion about artifact/documentation placement

## Executive Summary

The current architecture has significant confusion about:
1. **Artifact location** - Multiple conflicting references (`docs/artifacts/` vs `artifacts/`)
2. **Naming conventions** - Conflicting timestamp vs semantic naming
3. **Documentation structure** - Duplicate locations and unclear organization
4. **Protocol placement** - Multiple protocol locations with outdated information

This plan proposes an **industry-grade reorganization** that:
- Establishes clear, single source of truth for all locations
- Enforces conventions via validation
- Eliminates duplicates
- Follows professional project organization standards

## Current Problems

### 1. Artifact Location Confusion

**Problem:** Multiple conflicting references to artifact location

| Location | Reference | Status | Correct? |
|----------|-----------|--------|----------|
| `docs/artifacts/` | `docs/agents/docs_governance/01_artifact_management_protocol.md` | Referenced | ❌ **WRONG** - Doesn't exist |
| `artifacts/` | `agent_qms/q-manifest.yaml` | Active | ✅ **CORRECT** - Project root |
| `artifacts/` | `docs/agents/system.md` | Active | ✅ **CORRECT** - Single source of truth |
| `artifacts/` | Actual artifacts | Active | ✅ **CORRECT** - Current location |

**Impact:** Agents may try to create artifacts in non-existent `docs/artifacts/` directory.

### 2. Naming Convention Confusion

**Problem:** Conflicting naming conventions

| Convention | Reference | Status | Correct? |
|------------|-----------|--------|----------|
| Timestamp-based | `docs/agents/docs_governance/01_artifact_management_protocol.md` | Outdated | ❌ **WRONG** - Deprecated |
| Semantic naming | `agent_qms/q-manifest.yaml` | Active | ✅ **CORRECT** - Current standard |
| Semantic naming | `docs/agents/system.md` | Active | ✅ **CORRECT** - Single source of truth |

**Impact:** Agents may use outdated timestamp-based naming.

### 3. Documentation Structure Confusion

**Problem:** Duplicate and unclear documentation locations

| Location | Purpose | Status | Issue |
|----------|---------|--------|-------|
| `docs/agents/docs_governance/` | Governance protocols | Outdated | ❌ Contains outdated info |
| `docs/agents/protocols/` | Current protocols | Active | ✅ Current location |
| `docs/maintainers/` | Human maintainer docs | Active | ✅ Correct |
| `agent_qms/` | Framework code + docs | Mixed | ⚠️ Should only contain code |

**Impact:** Agents may read outdated protocols or be confused about where to find information.

### 4. Protocol Duplication

**Problem:** Same protocols in multiple locations with different information

| Protocol | Location 1 | Location 2 | Status |
|----------|------------|------------|--------|
| Artifact Management | `docs/agents/docs_governance/01_artifact_management_protocol.md` | `docs/agents/protocols/governance.md` | ❌ Conflicting |
| Implementation Plans | `docs/agents/docs_governance/02_implementation_plan_protocol.md` | `docs/agents/protocols/governance.md` | ❌ Conflicting |

**Impact:** Agents may follow outdated protocols.

## Proposed Solution

### Industry-Grade Architecture

Following professional project organization standards:

```
project-root/
├── artifacts/                    # ✅ Artifacts (project root)
│   ├── assessments/
│   ├── implementation_plans/
│   └── ...
├── agent_qms/                    # ✅ Framework code ONLY
│   ├── q-manifest.yaml          # Single source of truth for artifact config
│   ├── schemas/
│   ├── templates/
│   └── toolbelt/
├── docs/                         # ✅ All documentation
│   ├── agents/                   # AI agent instructions
│   │   ├── system.md             # Single source of truth
│   │   ├── protocols/            # Current protocols
│   │   ├── references/           # Quick references
│   │   └── ...
│   └── maintainers/              # Human maintainer documentation
│       ├── protocols/            # Detailed protocols
│       └── ...
└── scripts/                      # ✅ Automation scripts
    └── agent_tools/
```

### Key Principles

1. **Single Source of Truth**
   - Artifact location: `agent_qms/q-manifest.yaml` (enforced by AgentQMS)
   - Agent instructions: `docs/agents/system.md`
   - Naming conventions: AgentQMS validation (enforced)

2. **Clear Separation**
   - `artifacts/` - All artifacts (project root)
   - `agent_qms/` - Framework code ONLY (no docs)
   - `docs/agents/` - AI agent instructions
   - `docs/maintainers/` - Human maintainer docs

3. **Enforcement via Validation**
   - AgentQMS validates artifact location before creation
   - AgentQMS validates naming conventions before creation
   - Scripts validate documentation placement

## Implementation Plan

### Phase 1: Fix Artifact Location References

**Actions:**
1. Update `docs/agents/docs_governance/01_artifact_management_protocol.md`
   - Change `docs/artifacts/` → `artifacts/`
   - Update naming convention: timestamp → semantic
   - Mark as deprecated or update to current standards

2. Update all references to `docs/artifacts/`
   - Search and replace across codebase
   - Update documentation

3. Verify `agent_qms/q-manifest.yaml` is correct
   - Ensure all locations are `artifacts/` (project root)

**Validation:**
- No references to `docs/artifacts/` in codebase
- All artifact locations in manifest are `artifacts/`

### Phase 2: Consolidate Protocols

**Actions:**
1. **Deprecate `docs/agents/docs_governance/`**
   - Move current protocols to `docs/agents/protocols/`
   - Mark old location as deprecated
   - Add redirects or clear deprecation notices

2. **Update `docs/agents/protocols/governance.md`**
   - Ensure it matches current AgentQMS standards
   - Reference `agent_qms/q-manifest.yaml` as source of truth
   - Remove outdated information

3. **Create protocol index**
   - Single index in `docs/agents/protocols/README.md`
   - Links to all current protocols
   - Clear deprecation notices for old locations

**Validation:**
- No duplicate protocols with conflicting information
- All protocols reference correct artifact locations

### Phase 3: Clean Up agent_qms/

**Actions:**
1. **Move documentation out of `agent_qms/`**
   - `agent_qms/adoption_and_usage_guide.md` → `docs/maintainers/agent_qms/`
   - `agent_qms/quality_management_framework.md` → `docs/maintainers/agent_qms/`
   - Keep only code in `agent_qms/`

2. **Update references**
   - Update all references to moved docs
   - Update documentation indexes

**Validation:**
- `agent_qms/` contains only code (no `.md` files except README)
- All documentation is in `docs/`

### Phase 4: Enforce via Validation

**Actions:**
1. **Enhance AgentQMS validation**
   - Validate artifact location matches manifest
   - Validate naming conventions (semantic, not timestamp)
   - Validate no artifacts in wrong locations

2. **Add documentation validation**
   - Script to check documentation placement
   - Validate no docs in `agent_qms/` (except README)
   - Validate protocol locations

3. **Add pre-commit hooks**
   - Validate artifact locations before commit
   - Validate documentation placement before commit

**Validation:**
- All validation passes
- No artifacts created in wrong locations
- No documentation in wrong locations

## Directory Structure (Final)

```
project-root/
├── artifacts/                          # ✅ Artifacts (project root)
│   ├── assessments/                    # Assessment artifacts
│   ├── implementation_plans/            # Implementation plan artifacts
│   └── ...
│
├── agent_qms/                          # ✅ Framework code ONLY
│   ├── q-manifest.yaml                 # Single source of truth for artifacts
│   ├── schemas/                        # JSON schemas
│   ├── templates/                      # Jinja2 templates
│   └── toolbelt/                       # Python code
│       ├── __init__.py
│       └── core.py
│
├── docs/                               # ✅ All documentation
│   ├── agents/                         # AI agent instructions
│   │   ├── system.md                   # Single source of truth
│   │   ├── index.md                    # Documentation map
│   │   ├── protocols/                  # Current protocols
│   │   │   ├── governance.md           # Artifact management, etc.
│   │   │   ├── development.md
│   │   │   └── ...
│   │   ├── references/                 # Quick references
│   │   │   ├── architecture.md
│   │   │   ├── commands.md
│   │   │   └── tools.md
│   │   └── ...
│   │
│   └── maintainers/                    # Human maintainer documentation
│       ├── agent_qms/                  # AgentQMS documentation
│       │   ├── adoption_and_usage_guide.md
│       │   └── quality_management_framework.md
│       ├── protocols/                 # Detailed protocols
│       │   └── governance/
│       │       └── 03_blueprint_protocol_template.md
│       └── ...
│
└── scripts/                            # ✅ Automation scripts
    └── agent_tools/
        └── ...
```

## Enforcement Strategy

### 1. AgentQMS Validation (Primary)

**Location:** `agent_qms/toolbelt/core.py`

**Validations:**
- ✅ Artifact location matches `q-manifest.yaml`
- ✅ Filename follows semantic naming (not timestamp)
- ✅ No ALL CAPS filenames (except README.md, CHANGELOG.md)
- ✅ Frontmatter matches schema
- ✅ File doesn't already exist

**Enforcement:**
- Validation happens BEFORE creation
- Creation fails if validation fails
- File removed if post-creation validation fails

### 2. Documentation Validation (Secondary)

**Location:** `scripts/agent_tools/compliance/validate_documentation.py` (new)

**Validations:**
- ✅ No `.md` files in `agent_qms/` (except README.md)
- ✅ All protocols in `docs/agents/protocols/`
- ✅ No duplicate protocols with conflicting info
- ✅ All artifact location references are `artifacts/`

**Enforcement:**
- Pre-commit hook
- CI/CD validation
- Regular audits

### 3. Single Source of Truth

**Artifact Location:**
- **Source:** `agent_qms/q-manifest.yaml`
- **Enforced by:** AgentQMS toolbelt
- **Referenced by:** `docs/agents/system.md`

**Naming Conventions:**
- **Source:** AgentQMS validation
- **Enforced by:** AgentQMS toolbelt
- **Referenced by:** `docs/agents/system.md`

**Protocols:**
- **Source:** `docs/agents/protocols/`
- **Enforced by:** Documentation validation
- **Referenced by:** `docs/agents/system.md`

## Migration Plan

### Step 1: Update References (Low Risk)

1. Update `docs/agents/docs_governance/01_artifact_management_protocol.md`
   - Change `docs/artifacts/` → `artifacts/`
   - Update naming convention section
   - Add deprecation notice

2. Search and replace all `docs/artifacts/` references
   - Update documentation
   - Update scripts
   - Update comments

### Step 2: Consolidate Protocols (Medium Risk)

1. Review `docs/agents/docs_governance/` vs `docs/agents/protocols/`
2. Merge current information into `docs/agents/protocols/governance.md`
3. Mark `docs/agents/docs_governance/` as deprecated
4. Add redirects or clear deprecation notices

### Step 3: Move Documentation (Low Risk)

1. Move docs from `agent_qms/` to `docs/maintainers/agent_qms/`
2. Update all references
3. Verify no broken links

### Step 4: Add Validation (Low Risk)

1. Enhance AgentQMS validation (already done)
2. Create documentation validation script
3. Add pre-commit hooks
4. Add CI/CD validation

## Success Criteria

1. ✅ **No confusion about artifact location**
   - All references point to `artifacts/` (project root)
   - No references to `docs/artifacts/`
   - AgentQMS enforces location

2. ✅ **No confusion about naming conventions**
   - All references use semantic naming
   - No references to timestamp-based naming
   - AgentQMS enforces naming

3. ✅ **No duplicate protocols**
   - Single protocol location: `docs/agents/protocols/`
   - No conflicting information
   - Clear deprecation notices

4. ✅ **Clear documentation structure**
   - `agent_qms/` contains only code
   - All documentation in `docs/`
   - Clear separation between agent and maintainer docs

5. ✅ **Enforcement via validation**
   - AgentQMS validates before creation
   - Documentation validation prevents mistakes
   - Pre-commit hooks catch issues early

## Risks and Mitigation

### Risk 1: Breaking Existing References

**Mitigation:**
- Search and replace all references
- Update documentation indexes
- Add redirects for deprecated locations

### Risk 2: Agents Reading Outdated Protocols

**Mitigation:**
- Clear deprecation notices
- Update `docs/agents/system.md` to reference correct locations
- Remove or archive outdated protocols

### Risk 3: Validation Too Strict

**Mitigation:**
- Start with warnings
- Gradually enforce
- Provide clear error messages

## Next Steps

1. **Review and approve this plan**
2. **Execute Phase 1: Fix artifact location references**
3. **Execute Phase 2: Consolidate protocols**
4. **Execute Phase 3: Clean up agent_qms/**
5. **Execute Phase 4: Add validation**
6. **Verify success criteria**

## References

- `agent_qms/q-manifest.yaml` - Single source of truth for artifact locations
- `docs/agents/system.md` - Single source of truth for agent instructions
- `agent_qms/toolbelt/core.py` - AgentQMS validation implementation

---

*This plan establishes a professional, industry-grade architecture that eliminates confusion and enforces conventions via validation.*

