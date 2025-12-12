# Automated Compliance Fix Script - User Guide

## Overview

The automated compliance fix script helps maintain artifact naming conventions and directory organization in the AgentQMS framework. This guide covers the recent bug fixes and how to use the improved script safely.

## Recent Fixes (2025-11-28)

### Critical Bugs Resolved

1. **Processing Limits** - Script now respects user-specified file limits
2. **Data Corruption** - Removed duplicate file `_1` suffix logic
3. **Incorrect Prefixes** - Fixed mapping (`implementation_plan_` not `IMPLEMENTATION_PLAN_`)
4. **Registry File Protection** - INDEX.md and similar files now skipped
5. **Validation** - Pre-execution checks prevent conflicts

## Usage

### Basic Syntax

```bash
./automated_compliance_fix.sh [OPTIONS]
```

### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--max-files N` | Process maximum N files | `--max-files 5` |
| `--dry-run` | Preview changes without applying | `--dry-run` |
| `--directory DIR` | Specify artifacts directory | `--directory docs/artifacts` |

### Examples

#### Preview changes for 3 files
```bash
./automated_compliance_fix.sh --max-files 3 --dry-run
```

**Output:**
- Shows what would be renamed/moved
- No actual changes made
- File limit respected

#### Actually fix 5 files
```bash
./automated_compliance_fix.sh --max-files 5
```

**Safety:**
- Backups created automatically
- Validation runs before changes
- Stops at specified limit

#### Preview all changes
```bash
./automated_compliance_fix.sh --dry-run
```

**Use case:** Understand full scope before committing

## Safety Features

### 1. Dry-Run Mode
- **Purpose**: Preview all changes before applying
- **Usage**: Always run with `--dry-run` first
- **Output**: Detailed list of proposed changes

### 2. Processing Limits
- **Purpose**: Control scope of changes
- **Usage**: Start with small limits (3-5 files)
- **Benefit**: Incremental fixes, easier to review

### 3. Automatic Backups
- **Location**: `backups/automated_fixes_YYYYMMDD_HHMMSS/`
- **Content**: Original files before modification
- **Restoration**: Manual copy if needed

### 4. Pre-execution Validation
- **Checks**: Duplicate targets, existing files, missing sources
- **Action**: Blocks operation if issues detected
- **Message**: Clear warning with specific issues

### 5. Registry File Protection
- **Protected**: INDEX.md, MASTER_INDEX.md, README.md, REGISTRY.md
- **Behavior**: Completely skipped from all operations
- **Reason**: System files shouldn't be auto-modified

## What Gets Fixed

### Naming Conventions

#### Timestamp Format
- **Expected**: `YYYY-MM-DD_HHMM_`
- **Example**: `2025-11-28_1400_`
- **Fixed**: Missing or malformed timestamps

#### Type Prefixes
- **implementation_plan_**: Implementation plans
- **assessment-**: Assessments
- **design-**: Design documents
- **research-**: Research documents
- **template-**: Templates
- **BUG_**: Bug reports
- **SESSION_**: Session notes

#### Descriptive Naming
- **Expected**: kebab-case after prefix
- **Example**: `assessment-code-quality.md`
- **Fixed**: Underscores converted to hyphens

### Directory Organization

Files are organized by type:
- `implementation_plans/` - Implementation plans
- `assessments/` - Assessments and evaluations
- `design_documents/` - Design documents
- `research/` - Research findings
- `templates/` - Templates and examples
- `bug_reports/` - Bug reports
- `completed_plans/` - Completed plans and summaries

## Advanced Usage

### Individual Script Usage

#### Fix Naming Conventions Only
```bash
python AgentQMS/toolkit/maintenance/fix_naming_conventions.py \
    --directory docs/artifacts \
    --limit 5 \
    --dry-run
```

#### Reorganize Files Only
```bash
python AgentQMS/toolkit/maintenance/reorganize_files.py \
    --directory docs/artifacts \
    --limit 5 \
    --dry-run
```

### Validation Only
```bash
python AgentQMS/toolkit/maintenance/reorganize_files.py \
    --directory docs/artifacts \
    --validate-only
```

## Troubleshooting

### Issue: Changes not applied
**Solution**: Remove `--dry-run` flag

### Issue: Too many files processed
**Solution**: Add `--max-files N` to limit scope

### Issue: File already exists error
**Check**: Validation detected conflict
**Action**: Review proposed changes, resolve manually

### Issue: Confidence too low warning
**Meaning**: Content analysis uncertain about file type
**Action**: File skipped for safety, review manually

## Best Practices

1. **Always start with dry-run**
   ```bash
   ./automated_compliance_fix.sh --max-files 3 --dry-run
   ```

2. **Use small limits initially**
   - Start with 3-5 files
   - Review changes
   - Increase gradually

3. **Review validation output**
   - Check for conflicts
   - Verify proposed changes make sense
   - Look for warnings

4. **Keep backups**
   - Don't delete backup directories immediately
   - Review changes in git
   - Commit incrementally

5. **Use git for safety**
   ```bash
   git status  # Check before
   git diff    # Review changes
   git reset --hard  # Rollback if needed
   ```

## Confidence Scores

### Frontmatter Type (0.95)
- File has valid frontmatter with `type` field
- **High confidence** - uses explicit metadata

### Content Analysis (0.50-0.85)
- Based on keyword pattern matching
- Multiple patterns increase confidence
- **Minimum 0.85** required for moves

### Prefix Match (0.90)
- File has valid type prefix
- **High confidence** - explicit naming

## Error Messages

### `⚠️ Skipping: confidence too low (0.75 < 0.85)`
**Meaning**: Content analysis uncertain  
**Action**: File skipped, no changes made  
**Fix**: Add correct prefix manually or improve frontmatter

### `❌ Target file already exists`
**Meaning**: Destination file collision  
**Action**: Operation blocked  
**Fix**: Resolve conflict manually

### `✋ Reached file limit (N)`
**Meaning**: Processed maximum files  
**Action**: Stopped processing  
**Fix**: Increase limit or run again

### `ℹ️ Skipping registry/index file`
**Meaning**: Protected system file  
**Action**: File excluded  
**Fix**: None needed (working as designed)

## Testing

Run the test suite to verify fixes:

```bash
./AgentQMS/toolkit/test_compliance_fixes.sh
```

Expected output: `🎉 All tests passed!`

## Support

For issues or questions:
1. Check bug report: `docs/artifacts/bug_reports/2025-11-28_2153_BUG_001_automated-compliance-fix-bugs.md`
2. Review implementation plan: `docs/artifacts/implementation_plans/2025-11-28_2207_implementation_plan_fix-automated-compliance-bugs.md`
3. Check commit history for recent changes

## Version History

### 2025-11-28 (Current)
- ✅ Added processing limits (`--max-files`)
- ✅ Added dry-run mode
- ✅ Fixed prefix mappings
- ✅ Removed duplicate file bug
- ✅ Added registry file protection
- ✅ Added pre-execution validation
- ✅ Improved confidence thresholds

### Previous
- Basic naming convention fixes
- Simple directory reorganization
- No safety features
