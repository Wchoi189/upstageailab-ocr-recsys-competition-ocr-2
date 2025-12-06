# Phase 4 Quick Reference: Artifact Versioning & Lifecycle

## View Artifact Status

```bash
# Full dashboard (default)
cd AgentQMS/interface
make artifacts-status

# Compact table view
make artifacts-status-compact

# Aging information only
make artifacts-status-aging

# JSON output for scripting
make artifacts-status-json

# Show artifacts older than 6 months (180 days)
make artifacts-status-threshold DAYS=180

# Show artifacts older than 3 months (90 days)
make artifacts-status-threshold DAYS=90
```

## Artifact Frontmatter Template

All artifacts should include versioning metadata:

```yaml
---
type: design  # or assessment, implementation_plan, etc.
title: "My Artifact Title"
date: "2025-01-20 10:00 (KST)"
category: reference  # or: development, architecture, evaluation, etc.
status: active  # or: draft, superseded, archived
version: "1.0"
lifecycle_state: active  # or: draft, superseded, archived
tags:
  - design
  - v1
author: ai-agent
branch: main
---
```

## Versioning Rules

### Semantic Version Format: MAJOR.MINOR
- **Example**: 1.0, 1.1, 2.0, 2.3
- **Bump MAJOR**: Breaking changes, complete restructuring
- **Bump MINOR**: Enhancements, clarifications, bug fixes

### Artifact Lifecycle States
```
draft → active → superseded → archived
```

| State | Meaning | Use Case |
|-------|---------|----------|
| draft | Under development | New artifacts not yet approved |
| active | Current, in use | Approved and actively referenced |
| superseded | Replaced by newer version | Marked obsolete but kept for reference |
| archived | No longer needed | Historical archive, removed from circulation |

### Artifact Aging Categories
```
0-89 days:     ✅ OK (green)
90-179 days:   ⚠️  Warning (yellow) - Schedule review
180-364 days:  🚨 Stale (red) - Urgent review needed
365+ days:     📦 Archive (purple) - Archive or replace
```

## Python API Usage

### Check Artifact Age

```python
from pathlib import Path
from AgentQMS.agent_tools.utilities.versioning import ArtifactAgeDetector

detector = ArtifactAgeDetector()
age_days = detector.get_artifact_age(Path("docs/artifacts/design/my-design.md"))
category = detector.get_age_category(age_days)

print(f"Age: {age_days} days, Category: {category}")
```

### Extract/Update Version

```python
from pathlib import Path
from AgentQMS.agent_tools.utilities.versioning import VersionManager, SemanticVersion

mgr = VersionManager()

# Extract current version
version = mgr.extract_version_from_frontmatter(Path("docs/artifacts/design/my-design.md"))
if version:
    print(f"Current version: {version.major}.{version.minor}")

# Update to new version
new_version = SemanticVersion(1, 1)
mgr.update_version_in_frontmatter(Path("docs/artifacts/design/my-design.md"), new_version)
```

### Check Lifecycle State

```python
from pathlib import Path
from AgentQMS.agent_tools.utilities.versioning import ArtifactLifecycle

lifecycle = ArtifactLifecycle()
current_state = lifecycle.get_current_state(Path("docs/artifacts/design/my-design.md"))
print(f"Current state: {current_state}")

# Check if transition is allowed
if lifecycle.can_transition(Path("..."), "draft", "active"):
    lifecycle.transition(Path("..."), "active")
```

## Files Location

- **Versioning Module**: `AgentQMS/agent_tools/utilities/versioning.py`
- **Status Dashboard**: `AgentQMS/agent_tools/utilities/artifacts_status.py`
- **Design Document**: `docs/artifacts/design_documents/2025-01-20_1000_design-artifact-versioning-lifecycle.md`
- **Completion Assessment**: `docs/artifacts/assessments/2025-01-20_1100_assessment-phase4-completion.md`
- **Implementation Plan**: `docs/artifacts/implementation_plans/2025-12-06_1200_implementation_plan_agentqms-metadata-branch-versioning.md`

## Common Tasks

### Create New Artifact with Versioning

```bash
# Create artifact
make create-design NAME=my-design TITLE="My Design"

# Edit the created file to add lifecycle_state: draft
# Keep version: "1.0" as starting point
```

### Update Artifact After Review

```bash
# Change lifecycle_state: draft → active
# Optionally bump version if significant changes

# Validate
make validate
```

### Mark Artifact as Superseded

```yaml
# In the artifact frontmatter:
lifecycle_state: superseded
superseded_by: "2025-01-20_1200_design-new-feature.md"
deprecation_date: "2025-06-20"
archived_date: "2025-09-20"  # When to expect removal
```

### Archive Old Artifact

```yaml
lifecycle_state: archived
archived_date: "2025-01-20"
```

## Dashboard Example Output

```
📊 ARTIFACT STATUS DASHBOARD
============================

📈 SUMMARY:
   Total Artifacts:  125
   ✅ Healthy:       112 (90%)
   ⚠️  Warning:       8 (6%)
   🚨 Stale:         3 (2%)
   📦 Archive:       2 (2%)

🔄 LIFECYCLE STATES:
   DRAFT: 5
   ACTIVE: 110
   SUPERSEDED: 8
   ARCHIVED: 2

📅 AGE DISTRIBUTION:
   0-30 days: 45
   31-90 days: 67
   91-180 days: 8
   181-365 days: 3
   365+ days: 2

⚠️  ITEMS NEEDING ATTENTION:
ARTIFACT                    AGE    STATE       VERSION
my-old-design.md           185d   active      1.2
legacy-feature.md          352d   active      1.0
deprecated-api.md          401d   superseded  1.0
```

## Troubleshooting

### Dashboard shows "error" status
- Artifact may be missing version or lifecycle_state field
- Check artifact frontmatter has required fields
- Run `make validate` to see specific errors

### Path not found error
- Script auto-detects artifact location
- Can explicitly specify: `python artifacts_status.py --root /path/to/artifacts`
- Works from any working directory

### JSON export failing
- Ensure artifact files are readable
- Check YAML frontmatter is valid
- Some fields may be missing (will show as null in JSON)

## Next Steps

Phase 4 is complete. Next phase (Phase 5) will:
- Add version field to existing 113 artifacts
- Add lifecycle_state field to existing artifacts
- Standardize timestamps

All tools created in Phase 4 will be used to monitor artifact health during migration.
