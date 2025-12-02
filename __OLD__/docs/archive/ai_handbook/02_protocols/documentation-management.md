# Documentation Management Protocol

<!-- ai_cue:priority=high -->
<!-- ai_cue:trigger=before_creating_documentation -->

## File Placement

❌ **NEVER** place generated files in project root
✅ **ALWAYS** use proper directories:
- `/docs/sessions/YYYY-MM-DD/` - Session summaries
- `/docs/ai_handbook/08_planning/` - Plans and architecture
- `/scripts/temp/` - Temporary test scripts
- `/tests/` - Permanent test files

## Naming Convention

- Lowercase with hyphens: `session-summary-2025-10-21.md`
- Include dates in filename or header
- Descriptive but concise

## Documentation Hierarchy

### 1. Implementation Plan (PRIMARY)
- **Update existing**, never regenerate
- Single source of truth
- Must have: progress tracker, checklist, next tasks

### 2. Session Summary (SUPPLEMENTARY)
- **Rolling updates** only
- Focus on changes since last session
- Reference main plan, don't duplicate

## Before Creating ANY Document

**Ask yourself:**
1. Does similar documentation already exist?
2. Should I update existing doc instead?
3. Where specifically does this belong?

**If unsure, ASK before generating.**

## Cleanup

- Move temp files to `/scripts/temp/`
- Remove when no longer needed
- Keep root directory clean (only README.md, CLAUDE.md, core scripts)

---

**Key Principle**: Update > Create. Reference > Duplicate. Ask > Assume.
