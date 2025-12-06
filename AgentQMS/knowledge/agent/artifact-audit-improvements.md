---
title: "Artifact Audit Tool - Safety Features & Smart Date Inference"
date: "2025-12-06 20:45 (KST)"
type: "guide"
category: "reference"
status: "active"
version: "1.0"
tags: ["artifact_audit", "safety", "date_inference", "directory_exclusion"]
---

# Artifact Audit Tool - Safety Features & Smart Date Inference

## Overview

The improved artifact audit tool prevents the compliance disaster from PR #6 by adding:
- **Smart date inference** (git history → filesystem → present date)
- **Directory exclusion** (archive/, deprecated/, vlm_reports/ excluded by default)
- **Pre-flight safety checks** (preview, confirmation, automatic backup)
- **VLM report standardization** (BUG prefix → vlm_report_ type)

This prevents accidental corruption of archived/legacy artifacts and ensures accurate date preservation.

## Quick Start

### Safe Audit (Preview First)
```bash
cd AgentQMS/interface

# Preview batch 1 changes (dry-run, safe)
make audit-fix-batch BATCH=1

# Apply batch 1 fixes (with confirmation + auto-backup)
make audit-fix-batch-apply BATCH=1

# Preview all artifacts (excludes archive/deprecated)
make audit-fix-all --dry-run

# Fix all artifacts (with safety checks)
make audit-fix-all
```

### VLM Report Migration
```bash
# Preview migration
make audit-vlm-migrate

# Apply migration
make audit-vlm-migrate-apply
```

### Reporting
```bash
# Report violations without fixing
make audit-report

# Report including excluded directories
make audit-report-all
```

## Safety Features

### 1. Directory Exclusion (Default)

**Protected directories** (not modified by default):
- `archive/` - Legacy archived artifacts
- `deprecated/` - Deprecated artifacts
- `vlm_reports/` - Non-standard location (custom validation)

**To include excluded directories:**
```bash
python artifact_audit.py --all --include-excluded
make audit-fix-all-force  # Requires explicit YES confirmation
```

**Configuration** (`.agentqms/settings.yaml`):
```yaml
validation:
  excluded_directories:
    - archive
    - deprecated
```

### 2. Smart Date Inference

**No longer defaults to present date.** Instead, uses priority:

1. **Git creation date** (initial commit adding file)
   - Most reliable for artifacts with git history
   - Uses `git log --diff-filter=A` to find first commit

2. **Git last modified date** (most recent commit)
   - Falls back if creation date unavailable
   - Uses `git log -1` to get latest change

3. **Filesystem modification time** (`stat().st_mtime`)
   - Fallback if git history unavailable
   - Accurate for files not tracked in git

4. **Present date** (last resort only)
   - Used only if all other methods fail
   - Ensures every artifact gets a valid date

**Example:**
```python
# Artifact created in commit from 2025-11-15
# Even if you run audit today (2025-12-06),
# date will be "2025-11-15 HH:MM (KST)"
# NOT "2025-12-06 HH:MM (KST)"
```

### 3. Pre-flight Safety Checks

**Before any modifications, the tool:**
1. Previews files grouped by directory
2. Shows total count and breakdown
3. Prompts for confirmation: `Continue? [y/N]`
4. Creates automatic git stash backup: `pre-audit-backup-YYYYMMDD_HHMMSS`

**Example output:**
```
📋 Preview of files to be processed:
   Total files: 79

   assessments/ (20 files)
      - 2025-11-11_2343_assessment-ai-documentation-and-scripts-cleanup.md
      - 2025-11-12_1200_assessment-data-contract-assessment.md
      ... and 18 more

   bug_reports/ (7 files)
      - 2025-11-28_0000_BUG_001_dominant-edge-extension-failure.md
      ... and 6 more

⚠️  This will modify the above files!
   Continue? [y/N]:
```

**To skip confirmation** (use with caution):
```bash
python artifact_audit.py --all --no-confirm
```

**To skip automatic backup** (dangerous):
```bash
python artifact_audit.py --all --no-stash
```

## VLM Report Standardization

### Problem
VLM reports used incorrect `BUG` prefix instead of dedicated type.

**Before:**
```
docs/artifacts/vlm_reports/2025-12-03_BUG-001_inference-overlay-misalignment_69.md
type: bug_report
category: troubleshooting
```

### Solution
New `vlm_report` artifact type with proper naming.

**After:**
```
docs/artifacts/vlm_reports/2025-12-03_1200_vlm_report_inference-overlay-misalignment_69.md
type: vlm_report
category: evaluation
```

### Migration
```bash
# Preview changes
make audit-vlm-migrate

# Apply migration
make audit-vlm-migrate-apply
```

**What it does:**
- Renames files: `BUG-XXX` → `vlm_report_` pattern
- Updates frontmatter type: `bug_report` → `vlm_report`
- Updates category: `troubleshooting` → `evaluation`
- Adds default timestamp: `1200` (noon)

## Configuration

### Exclude Custom Directories

Edit `.agentqms/settings.yaml`:
```yaml
validation:
  excluded_directories:
    - archive
    - deprecated
    - my_custom_dir
```

### Default Settings
```yaml
validation:
  excluded_directories:
    - archive
    - deprecated
  # vlm_reports is always excluded (non-standard location)
```

## Command Reference

### Direct Python (from project root)
```bash
# Preview batch 1
python AgentQMS/agent_tools/audit/artifact_audit.py --batch 1 --dry-run

# Fix batch 1 with confirmation
python AgentQMS/agent_tools/audit/artifact_audit.py --batch 1

# Fix specific files
python AgentQMS/agent_tools/audit/artifact_audit.py --files path/to/file.md

# Fix all (excluding archive/deprecated)
python AgentQMS/agent_tools/audit/artifact_audit.py --all

# Fix all (including excluded dirs)
python AgentQMS/agent_tools/audit/artifact_audit.py --all --include-excluded

# Report violations only
python AGentQMS/agent_tools/audit/artifact_audit.py --all --report

# Dangerous: no confirm, no backup
python artifact_audit.py --all --no-confirm --no-stash
```

### Via Makefile (from AgentQMS/interface)
```bash
make audit-fix-batch BATCH=1           # Preview
make audit-fix-batch-apply BATCH=1     # Apply
make audit-fix-all                     # Fix all (safe)
make audit-fix-all-force               # Fix all (requires YES)
make audit-report                      # Report only
make audit-report-all                  # Report + excluded
make audit-vlm-migrate                 # Preview migration
make audit-vlm-migrate-apply           # Apply migration
```

## Troubleshooting

### No files found
**Problem**: `❌ No files to process after applying filters`

**Solution**: Check current directory
```bash
# Must run from project root
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
python AGentQMS/agent_tools/audit/artifact_audit.py --all

# Or use Makefile from AgentQMS/interface
cd AgentQMS/interface
make audit-report
```

### Files excluded unexpectedly
**Problem**: Expected file not found in preview

**Check if it's in an excluded directory:**
```bash
# Preview with exclusions shown
make audit-report

# Preview including excluded
make audit-report-all
```

### Date inference not working
**Problem**: Artifact still has old/wrong date

**Check git history:**
```bash
git log --follow --oneline -- docs/artifacts/your-file.md
```

**If no git history**, use filesystem date:
```bash
stat docs/artifacts/your-file.md | grep Modify
```

**Last resort**, manually fix frontmatter:
```yaml
date: "2025-12-06 20:45 (KST)"
```

### Stash backup failed
**Problem**: Warning about git stash backup

**Options:**
1. Ensure you're in a git repository: `git status`
2. Have unstaged changes that can be stashed
3. Use `--no-stash` to skip (risky)

**To recover from backup stash:**
```bash
git stash list                    # Find your stash
git stash pop stash@{0}           # Restore latest
```

## How It Prevents PR #6 Disaster

| Problem | Before | After |
|---------|--------|-------|
| **Date overwriting** | Defaulted to present date | Smart git/filesystem inference |
| **Archive corruption** | No exclusions | Excluded by default, override available |
| **Silent errors** | No preview/confirmation | Preview + confirmation + backup |
| **VLM naming** | Used BUG prefix | Proper vlm_report_ type + migration |
| **No safety net** | Direct modifications | Auto git stash backup before changes |

## Examples

### Fix Assessment Artifacts Only
```bash
python AGentQMS/agent_tools/audit/artifact_audit.py \
  --files docs/artifacts/assessments/*.md \
  --dry-run
```

### Dry-run All Before Production Fix
```bash
# Step 1: Preview
make audit-fix-all --dry-run

# Step 2: Review output carefully

# Step 3: Apply
make audit-fix-all
```

### Batch Operations with Explicit Control
```bash
# Preview batch 1
make audit-fix-batch BATCH=1

# Confirm the changes look good, then apply
make audit-fix-batch-apply BATCH=1

# Move to batch 2
make audit-fix-batch BATCH=2
make audit-fix-batch-apply BATCH=2
```

### Include Only Specific Excluded Directory
```bash
# Fix archive only (dangerous!)
python AGentQMS/agent_tools/audit/artifact_audit.py \
  --files docs/artifacts/archive/*.md \
  --include-excluded \
  --dry-run
```

## See Also

- `AgentQMS/knowledge/agent/system.md` - Core agent rules
- `AgentQMS/knowledge/agent/tool_catalog.md` - Tool discovery
- `AgentQMS/agent_tools/audit/artifact_audit.py` - Implementation
- `.agentqms/settings.yaml` - Configuration
