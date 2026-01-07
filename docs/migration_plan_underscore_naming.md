# Artifact Naming Convention Migration Plan

## Objective
Standardize all artifact filenames to use **underscore separators** between structural components for consistency and simplified validation.

## Current vs. Proposed Convention

### Current (Mixed)
```
YYYY-MM-DD_HHMM_BUG_NNN_description.md      ✓ Uses underscores
YYYY-MM-DD_HHMM_assessment-description.md   ✗ Uses hyphen
YYYY-MM-DD_HHMM_design-description.md       ✗ Uses hyphen
YYYY-MM-DD_HHMM_audit-description.md        ✗ Uses hyphen
YYYY-MM-DD_HHMM_research-description.md     ✗ Uses hyphen
```

### Proposed (Unified)
```
YYYY-MM-DD_HHMM_BUG_NNN_description.md
YYYY-MM-DD_HHMM_assessment_description.md
YYYY-MM-DD_HHMM_design_description.md
YYYY-MM-DD_HHMM_audit_description.md
YYYY-MM-DD_HHMM_research_description.md
```

**Key**:
- `_` = Structural separator (between date, time, type, description)
- `-` = Word separator (within description for kebab-case)

## Migration Scope

### Files to Rename
- **Assessments**: 8 files + 1 Korean translation
- **Design Documents**: 7 files
- **Audits**: 1 file
- **Research**: 0 files (already compliant or none exist)
- **Total**: 17 files

### References to Update
- **INDEX.md files**: 3 files (assessments, audits, design_documents)
- **Cross-references**: 3 files
- **Total**: 6 files with references

## Changes Made

### 1. AgentQMS Template Updates ✅
Updated [AgentQMS/tools/core/artifact_templates.py](../AgentQMS/tools/core/artifact_templates.py):
- `assessment-{name}` → `assessment_{name}`
- `design-{name}` → `design_{name}`
- `research-{name}` → `research_{name}`
- `audit` pattern (needs checking - may be plugin-based)

### 2. Translation Tool Updates ✅
Updated [tools/translate_document.py](../tools/translate_document.py):
- Modified `normalize_artifact_name()` to use underscore separators for all artifact types
- Maintains hyphen for word separation within descriptive names

### 3. Migration Script Created ✅
[scripts/migrate_to_underscore_naming.py](../scripts/migrate_to_underscore_naming.py):
- Automatically finds all files needing migration
- Searches and updates all references in markdown files
- Renames both `.md` and `.ko.md` files
- Generates rollback script for safety

## Execution Steps

### Step 1: Review Changes
```bash
# Preview what will be migrated (dry-run)
python3 scripts/migrate_to_underscore_naming.py
# Answer "no" when prompted to see the plan without executing
```

### Step 2: Execute Migration
```bash
# Run the migration
python3 scripts/migrate_to_underscore_naming.py
# Answer "yes" when prompted to execute
```

### Step 3: Verify
```bash
# Check for any issues
git status
git diff

# Verify no broken links
grep -r "assessment-\|design-\|audit-\|research-" --include="*.md" docs/
```

### Step 4: Rollback (if needed)
```bash
# If something goes wrong, run the generated rollback script
./scripts/rollback_naming_migration.sh
```

## Benefits

1. **Consistency**: All artifact types use the same separator convention
2. **Simplification**: Single pattern to validate and parse
3. **Clarity**: Clear distinction between structural separators (_) and word separators (-)
4. **Maintainability**: Easier to write regex patterns and validation rules

## Risk Mitigation

- ✅ Automated script handles both renaming and reference updates
- ✅ Rollback script generated before execution
- ✅ All changes can be reviewed in git before committing
- ✅ No external references found (GitHub, documentation sites)

## Testing

After migration, verify:
```bash
# Test new artifact creation
cd AgentQMS/bin && make create-assessment NAME="test-migration" TITLE="Test"

# Expected filename format:
# YYYY-MM-DD_HHMM_assessment_test-migration.md
```

## Timeline

- **Preparation**: ✅ Complete
- **Execution**: Ready to run (5 minutes)
- **Verification**: 10 minutes
- **Total**: ~15 minutes

## Approval

- [ ] Review migration plan
- [ ] Execute migration script
- [ ] Verify results
- [ ] Commit changes

---

**Created**: 2026-01-07
**Status**: Ready for execution
**Rollback**: Automated script included
