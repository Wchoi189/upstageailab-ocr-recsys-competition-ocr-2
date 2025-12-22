# Artifact Archive System

## Overview

The Artifact Archive System allows you to automatically move completed artifacts marked with `status: "archived"` from `docs/artifacts/` to `archive/artifacts/` while preserving the directory structure.

## How It Works

1. **Mark artifacts as archived** by setting `status: "archived"` in frontmatter
2. **Preview what will be archived**: `make archive-artifacts-dry`
3. **Move files to archive**: `make archive-artifacts`

## Usage

### Step 1: Mark Artifacts for Archival

Edit the frontmatter of any artifact you want to archive:

```yaml
---
status: "archived"  # Change from "completed" or "active"
---
```

### Step 2: Preview

```bash
cd AgentQMS/interface
make archive-artifacts-dry
```

This shows which files will be moved (dry-run mode).

### Step 3: Apply

```bash
cd AgentQMS/interface
make archive-artifacts
```

Or run the script directly:

```bash
uv run python AgentQMS/agent_tools/archive/archive_artifacts.py
```

## Directory Structure

Files are moved while preserving structure:

```
docs/artifacts/implementation_plans/2024-12-04_plan.md
  → archive/artifacts/implementation_plans/2024-12-04_plan.md

docs/artifacts/assessments/2025-01-15_assessment.md
  → archive/artifacts/assessments/2025-01-15_assessment.md
```

## Benefits

✅ **Automatic**: No manual file moving
✅ **Safe**: Preview mode before applying changes
✅ **Preserves structure**: Easy to find archived files
✅ **AgentQMS compliant**: Archive directory excluded from validation
✅ **Reversible**: Easy to restore files if needed

## Validation

The `archive/` directory is automatically excluded from AgentQMS validation, so archived files won't clutter validation reports.

After archiving, run validation to confirm:

```bash
make validate
```

## Restoring Archived Files

To restore an archived file:

1. Move it back from `archive/artifacts/` to `docs/artifacts/`
2. Update its status from `"archived"` to `"active"` or `"completed"`

```bash
# Example
mv archive/artifacts/implementation_plans/2024-12-04_plan.md \
   docs/artifacts/implementation_plans/

# Then edit frontmatter:
status: "active"  # or "completed"
```

## Script Location

`AgentQMS/agent_tools/archive/archive_artifacts.py`

## Command Reference

```bash
# Preview (dry-run)
make archive-artifacts-dry

# Apply changes
make archive-artifacts

# Direct script usage with options
uv run python AgentQMS/agent_tools/archive/archive_artifacts.py --help
```

## Options

- `--dry-run`: Preview without moving files
- `--all`: Process all archived files (default behavior)

## Example Workflow

```bash
# 1. Mark implementation plan as archived
vim docs/artifacts/implementation_plans/2025-12-14_plan.md
# Change: status: "completed" → status: "archived"

# 2. Preview what will be archived
cd AgentQMS/interface && make archive-artifacts-dry

# Output:
# 📋 Found 1 artifact(s) marked for archival:
#    • implementation_plans/2025-12-14_plan.md

# 3. Apply archival
make archive-artifacts

# 4. Verify
ls archive/artifacts/implementation_plans/
# 2025-12-14_plan.md ✅

# 5. Validate (archived files won't appear)
make validate
# Total files: 29 (3 fewer than before)
```

## Status Lifecycle

```
draft → active → completed → archived
         ↓                      ↓
      deferred            (moved to archive/)
```

- **draft**: Being written
- **active**: Currently working on
- **completed**: Finished, still in active directory
- **archived**: Completed and moved to archive
- **deferred**: Postponed (stays in place)
