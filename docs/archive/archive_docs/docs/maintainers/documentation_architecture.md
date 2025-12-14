# Documentation Architecture Guide

This document explains the organization and structure of the project's documentation system.

## Overview

The documentation is organized by audience and purpose, with clear separation between AI agent instructions, human maintainer documentation, and generated artifacts.

## Directory Structure

### `docs/agents/` - AI Agent Instructions

**Purpose**: Instructions, protocols, and references for AI agents working on this project.

**Contents**:
- `system.md` - Single source of truth for all AI agent rules and instructions
- `index.md` - Documentation map for agents
- `protocols/` - Development, component, configuration, and governance protocols
- `references/` - Quick reference for architecture, commands, and tools
- `tracking/` - Tracking CLI and database API documentation
- `automation/` - Tooling overview and catalogs
- `coding_protocols/` - Coding-specific protocols (e.g., Streamlit)

**Key Principle**: This is the primary documentation for AI agents. All agent-facing documentation should be here.

### `docs/maintainers/` - Human Maintainer Documentation

**Purpose**: Detailed documentation for human maintainers and developers.

**Contents**:
- Detailed guides and tutorials
- Planning documents and assessments
- Operational procedures
- Architecture deep-dives
- Process management guides

**Key Principle**: This is the authoritative source for maintainer-facing documentation. More detailed than agent docs.

### `artifacts/` (root level) - AI-Generated Artifacts

**Purpose**: AgentQMS-managed artifacts created by AI agents.

**Contents**:
- `assessments/` - Assessment artifacts (audits, evaluations)
- `implementation_plans/` - Implementation plan artifacts
- `MASTER_INDEX.md` - Master index of all artifacts
- `*/INDEX.md` - Category-specific indexes

**Key Principle**: All artifacts are managed by AgentQMS toolbelt. Never create manually.

**Management**:
- Created via: `AgentQMSToolbelt.create_artifact()`
- Indexes updated via: `python scripts/agent_tools/documentation/update_artifact_indexes.py`
- Validated via: `python scripts/agent_tools/compliance/validate_artifacts.py`

### `docs/bug_reports/` - Bug Reports

**Purpose**: AgentQMS-managed bug reports.

**Contents**:
- Bug reports with standardized format
- Bug ID format: `BUG-YYYYMMDD-###`

**Key Principle**: Managed by AgentQMS, requires bug ID generation before creation.

### `docs/quick_reference/` - Quick Reference Documentation

**Purpose**: Lightweight quick reference documents and wrappers.

**Contents**:
- `QUICK_FIXES.md` - Quick fixes log
- `process_management.md` - Quick reference wrapper pointing to maintainers version
- Other quick reference guides

**Key Principle**: These are lightweight wrappers or quick references. Authoritative content is in `maintainers/` or `agents/`.

### `docs/archive/` - Historical/Archived Content

**Purpose**: Historical documentation, deprecated content, and archived materials.

**Contents**:
- Deprecated documentation
- Historical changelogs
- Archived assessments and plans
- Legacy documentation

**Key Principle**: Read-only historical reference. Do not update or modify.

### Other Documentation Directories

- `docs/pipeline/` - Pipeline documentation
- `docs/hardware/` - Hardware-specific documentation
- `docs/setup/` - Setup and installation guides
- `docs/testing/` - Testing documentation
- `docs/troubleshooting/` - Troubleshooting guides
- `docs/sessions/` - Session logs and notes
- `docs/assets/` - Static assets (images, diagrams)

## Root-Level Documentation Files

### Standard Files (Keep at Root)
- `CHANGELOG.md` - Project changelog (standard location)
- `README.md` - Project readme (standard location)
- `index.md` - Documentation navigation hub
- `sitemap.md` - Auto-generated documentation sitemap

### Quick Reference Files
- `docs/quick_reference/process_management.md` - Quick reference wrapper pointing to `maintainers/process_management.md`

## Artifact Storage

### Current Architecture

**Single Source of Truth**: `artifacts/` (root level)

All artifacts are stored in the root-level `artifacts/` directory:
- `artifacts/assessments/` - Assessment artifacts
- `artifacts/implementation_plans/` - Implementation plan artifacts
- `artifacts/MASTER_INDEX.md` - Master index
- `artifacts/*/INDEX.md` - Category indexes

**Legacy Note**: `docs/artifacts/` was removed in 2025-11-12. All scripts now use `artifacts/` root.

### Artifact Management

**Creation**:
```python
from agent_qms.toolbelt import AgentQMSToolbelt

toolbelt = AgentQMSToolbelt()
artifact_path = toolbelt.create_artifact(
    artifact_type="assessment",  # or "implementation_plan"
    title="My Artifact",
    content="## Summary\n...",
    author="ai-agent",
    tags=["tag1", "tag2"]
)
```

**Index Updates**:
```bash
python scripts/agent_tools/documentation/update_artifact_indexes.py
```

**Validation**:
```bash
python scripts/agent_tools/compliance/validate_artifacts.py
```

## Documentation Principles

### 1. Single Source of Truth
- Each piece of information should have one authoritative location
- Use wrappers/links to point to authoritative sources
- Avoid duplication

### 2. Audience Separation
- **AI Agents** → `docs/agents/`
- **Human Maintainers** → `docs/maintainers/`
- **Quick Reference** → `docs/quick_reference/`

### 3. Clear Organization
- Group related documentation together
- Use consistent naming conventions
- Maintain clear directory structure

### 4. Discoverability
- Use indexes and sitemaps
- Provide clear navigation
- Link related documents

### 5. Maintenance
- Keep documentation up to date
- Remove obsolete content (move to archive)
- Update references when moving files

## Migration History

### 2025-11-12: Artifact Storage Consolidation
- **Removed**: `docs/artifacts/` (legacy index-only directory)
- **Consolidated**: All artifacts now in `artifacts/` root
- **Updated**: Script defaults to use `artifacts/` root
- **Result**: Single source of truth for artifacts

### 2025-11-12: Documentation Organization
- **Created**: `docs/quick_reference/` for quick reference docs
- **Moved**: `QUICK_FIXES.md` → `docs/quick_reference/`
- **Moved**: `CI_FIX_IMPLEMENTATION_PLAN.md` → `docs/maintainers/planning/`
- **Moved**: `process_management.md` → `docs/quick_reference/process_management.md` (quick reference wrapper)

## Adding New Documentation

### For AI Agents
1. Place in `docs/agents/` or appropriate subdirectory
2. Follow agent documentation conventions
3. Update `docs/agents/index.md` if needed

### For Human Maintainers
1. Place in `docs/maintainers/` or appropriate subdirectory
2. Use detailed, comprehensive format
3. Link from quick reference if needed

### For Artifacts
1. Use AgentQMS toolbelt (never create manually)
2. Artifacts go in `artifacts/` root
3. Indexes are auto-generated

### For Quick References
1. Place in `docs/quick_reference/`
2. Keep lightweight
3. Link to authoritative sources in `maintainers/` or `agents/`

## Tools and Scripts

### Documentation Management
- `scripts/agent_tools/documentation/update_artifact_indexes.py` - Update artifact indexes
- `scripts/agent_tools/documentation/generate_sitemap.py` - Generate documentation sitemap
- `scripts/agent_tools/documentation/validate_links.py` - Validate documentation links

### Artifact Management
- `scripts/agent_tools/core/artifact_workflow.py` - Legacy artifact workflow
- `scripts/agent_tools/compliance/validate_artifacts.py` - Validate artifacts
- `scripts/agent_tools/compliance/monitor_artifacts.py` - Monitor artifacts

### Discovery
- `scripts/agent_tools/core/discover.py` - Discover available tools
- `docs/agents/index.md` - Agent documentation map
- `docs/sitemap.md` - Documentation sitemap

## Best Practices

1. **Always use AgentQMS for artifacts** - Never create artifact files manually
2. **Update indexes after changes** - Run index updater after adding/removing artifacts
3. **Link, don't duplicate** - Use wrappers and links instead of copying content
4. **Keep it organized** - Place files in appropriate directories
5. **Document changes** - Update this guide when making architectural changes
6. **Validate regularly** - Run validation scripts to catch issues early

---

**Last Updated**: 2025-11-12
**Maintained By**: AI Agent + Human Maintainers
