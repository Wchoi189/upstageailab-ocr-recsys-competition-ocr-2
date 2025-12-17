---
ads_version: "1.0"
title: "Readme"
date: "2025-12-16 22:11 (KST)"
type: "reference"
category: "reference"
status: "active"
version: "1.0"
tags: ['reference', 'reference']
---

# DEPRECATED ALL-CAPS Documentation Files

## Notice

These files were moved here on 2025-12-16 as part of the AI Documentation Standardization initiative.

**Reason for Deprecation**: Violation of AgentQMS naming conventions
- ALL-CAPS filenames are prohibited (except README.md, CHANGELOG.md)
- Required format: `YYYY-MM-DD_HHMM_{TYPE}_slug.md`
- All files must be in `docs/artifacts/{TYPE}/` directories

## Files Archived

| Original File | Type | Migration Status |
|---------------|------|------------------|
| CLAUDE_HANDOFF_INDEX.md | reference | Archived - Contains outdated Phase 4 context |
| CONTINUATION_PROMPT.md | handoff | Archived - Superseded by implementation plans |
| DOCUMENTATION_CONVENTIONS.md | reference | Archived - Superseded by tier1-sst/*.yaml |
| DOCUMENTATION_EXECUTION_HANDOFF.md | implementation_plan | Archived - Session-specific context |
| DOCUMENTATION_STANDARDIZATION_PROGRESS.md | documentation | Archived - Progress now in artifacts |
| FOUNDATION_PREPARATION_COMPLETE.md | assessment | Archived - Historical completion report |
| FOUNDATION_STATUS.md | implementation_plan | Archived - Superseded by current plans |
| INFERENCE_REFACTORING_DOCUMENTATION_STATUS.md | assessment | Archived - Missing frontmatter |
| PHASE4_QUICKSTART.md | reference | Archived - Phase 4 context |
| SESSION_HANDOVER_2025-12-16.md | session-handover | Archived - Should use session_notes/ |

## Migration Path

These files are NOT being migrated to proper artifacts because:
1. **Outdated Context**: Most contain session-specific or phase-specific information now obsolete
2. **Incomplete Frontmatter**: Many lack proper AgentQMS frontmatter
3. **Superseded**: Information replaced by standardized artifacts in proper locations
4. **Historical**: Serve as archive only, not active documentation

## Retention Policy

- **Retention Period**: 30 days from deprecation date (until 2026-01-15)
- **Access**: Read-only archive
- **Deletion**: After retention period if no objections raised

## Reference to Current Standards

See:
- `.ai-instructions/tier1-sst/naming-conventions.yaml` - Naming rules
- `.ai-instructions/tier1-sst/file-placement-rules.yaml` - Placement rules
- `docs/artifacts/` - Current artifact structure

---
*Archived: 2025-12-16 by AI Documentation Standardization*
