---
ads_version: "1.0"
title: "Artifact Versioning Lifecycle"
date: "2025-12-06 15:48 (KST)"
type: "design"
category: "architecture"
status: "active"
version: "1.0"
tags: ['design', 'architecture', 'documentation']
---



# Artifact Versioning & Lifecycle Management

**Status**: Implemented
**Phase**: Phase 4 of AgentQMS Metadata Enhancement
**Created**: 2025-01-20
**Last Updated**: 2025-01-20

## Overview

Phase 4 introduces comprehensive artifact versioning and lifecycle management to the AgentQMS framework. This enables tracking artifact evolution, detecting stale content, and managing artifact states through their complete lifecycle.

## Components

### 1. Semantic Versioning (versioning.py)

The `SemanticVersion` class implements MAJOR.MINOR versioning for artifacts:

```python
class SemanticVersion:
    major: int
    minor: int

    def bump_major() -> SemanticVersion:
        """Increment major version, reset minor to 0."""

    def bump_minor() -> SemanticVersion:
        """Increment minor version."""
```

**Version Format**: `MAJOR.MINOR` (e.g., `1.0`, `2.3`)

**Increment Rules**:
- **Bump Major**: Breaking changes, complete restructuring, significant content overhaul
- **Bump Minor**: Enhancements, clarifications, bug fixes, non-breaking updates

**Usage in Frontmatter**:
```yaml
version: "1.0"
last_updated: "2025-01-20T10:30:00Z"
```

### 2. Artifact Lifecycle States

The `ArtifactLifecycle` class manages artifact states through a complete lifecycle:

```
draft → active → superseded → archived
```

**State Transitions**:

| From      | To          | Trigger                                  |
|-----------|-------------|------------------------------------------|
| draft     | active      | Review complete, ready for use           |
| active    | superseded  | Newer version published, marked obsolete |
| active    | archived    | No longer needed, removed from active    |
| superseded| archived    | Cleanup after reasonable transition      |
| draft     | archived    | Rejected or cancelled                    |

**State Semantics**:

- **draft**: Content under development, not yet approved for use
- **active**: Current, approved content in regular use
- **superseded**: Replaced by newer version, kept for reference only
- **archived**: Removed from active circulation, historical reference

**Frontmatter Fields**:
```yaml
lifecycle_state: "active"
superseded_by: "2025-01-20_design_newfeature.md"  # if superseded
deprecation_date: "2025-06-20"  # when to expect removal
archived_date: "2025-09-20"     # when archived
```

### 3. Artifact Aging Detection

The `ArtifactAgeDetector` class identifies artifacts needing review based on age:

**Age Categories**:

| Category | Days Old | Status           | Action                           |
|----------|----------|------------------|----------------------------------|
| OK       | 0-89     | ✅ Current       | No action needed                 |
| Warning  | 90-179   | ⚠️ Aging         | Schedule review soon             |
| Stale    | 180-364  | 🚨 Stale         | Urgent review required           |
| Archive  | 365+     | 📦 Archive Ready | Archive or replace immediately   |

**Implementation**:
```python
def get_age_category(age_days: int | None) -> str:
    """Determine age category based on days old."""
    if age_days is None:
        return "unknown"
    if age_days < 90:
        return "ok"
    if age_days < 180:
        return "warning"
    if age_days < 365:
        return "stale"
    return "archive"
```

### 4. Version Management

The `VersionManager` class handles version extraction and updates from artifact YAML frontmatter:

```python
def extract_version_from_frontmatter(path: Path) -> SemanticVersion | None:
    """Extract version from artifact frontmatter."""

def update_version_in_frontmatter(path: Path, version: SemanticVersion) -> bool:
    """Update version in artifact frontmatter."""
```

**Supported Formats**:
```yaml
# Format 1: String version
version: "1.2"

# Format 2: Structured version (alternative)
version:
  major: 1
  minor: 2
```

### 5. Artifact Status Dashboard

The `artifacts_status.py` utility provides comprehensive visibility into artifact health:

**Available Views**:

```bash
# Default: dashboard with summary and alerts
make artifacts-status

# Compact table view
make artifacts-status-compact

# Aging information only
make artifacts-status-aging

# JSON output for scripting
make artifacts-status-json

# Show artifacts older than threshold
make artifacts-status-threshold DAYS=90
```

**Dashboard Components**:

1. **Summary Statistics**
   - Total artifacts
   - Health percentages by category
   - Error count

2. **Lifecycle Distribution**
   - Count of artifacts in each state

3. **Age Distribution**
   - Breakdown by age ranges (0-30d, 31-90d, etc.)

4. **Attention Items**
   - Artifacts needing review or action

## Integration with AgentQMS

### Frontmatter Template

All artifacts should include versioning metadata:

```yaml
---
artifact_type: design
title: "My Design Document"
version: "1.0"
last_updated: "2025-01-20T10:30:00Z"
lifecycle_state: "active"
branch: "main"
tags:
  - design
  - v1
---
```

### Validation Rules

**Strict Mode (default)**: Artifacts without version/lifecycle fields fail validation
**Lenient Mode**: Missing fields trigger warnings but don't fail validation

```bash
# Strict validation (default)
make validate

# Lenient validation (backward compatible)
validate_artifacts.py --lenient-plugins
```

### Artifact Workflow

1. **Create** artifact in draft state
2. **Develop** content with version 0.1, 0.2, etc.
3. **Review** and transition to active, version 1.0
4. **Maintain** with minor bumps (1.1, 1.2, etc.)
5. **Supersede** when replaced, bump major version in successor
6. **Archive** when no longer needed

## Usage Examples

### Creating a Versioned Artifact

```bash
# Create new artifact
make create-design NAME=feature-api TITLE="API Design"

# Edit frontmatter to add versioning
version: "1.0"
lifecycle_state: "draft"  # Start in draft
```

### Checking Artifact Health

```bash
# Full dashboard
make artifacts-status-dashboard

# Check artifacts older than 6 months
make artifacts-status-threshold DAYS=180

# Get JSON for external processing
make artifacts-status-json > artifact_health.json
```

### Updating Artifact Version

After making significant changes:

```python
from pathlib import Path
from AgentQMS.agent_tools.utilities.versioning import VersionManager, SemanticVersion

mgr = VersionManager()
new_version = SemanticVersion(1, 1)  # From 1.0 to 1.1
mgr.update_version_in_frontmatter(Path("docs/artifacts/my-design.md"), new_version)
```

### Transitioning Artifact State

When artifact is ready for use:

```python
from pathlib import Path
from AgentQMS.agent_tools.utilities.versioning import ArtifactLifecycle

lifecycle = ArtifactLifecycle()
artifact_path = Path("docs/artifacts/my-design.md")

# Transition: draft → active
if lifecycle.can_transition(artifact_path, "draft", "active"):
    lifecycle.transition(artifact_path, "active")
```

## Validation & Compliance

### Schema Validation

All versioning fields are validated against schemas:

```yaml
# AgentQMS/conventions/schemas/artifacts/versioning.yaml
version:
  type: string
  pattern: "^\\d+\\.\\d+$"
  description: "Semantic version in MAJOR.MINOR format"

lifecycle_state:
  type: string
  enum: ["draft", "active", "superseded", "archived"]

last_updated:
  type: string
  format: "date-time"  # ISO 8601
```

### Compliance Checks

Run validation with versioning checks:

```bash
# Full validation including versioning
make validate

# Check only versioning compliance
make validate-naming  # includes version format check
```

## Makefile Targets

### Artifact Status Monitoring

| Target | Purpose |
|--------|---------|
| `artifacts-status` | Show default dashboard view |
| `artifacts-status-dashboard` | Full dashboard with summary and details |
| `artifacts-status-compact` | Compact table view |
| `artifacts-status-aging` | Age information only |
| `artifacts-status-json` | JSON output for scripting |
| `artifacts-status-threshold DAYS=N` | Show artifacts older than N days |

### Utility Classes

**Module**: `AgentQMS/agent_tools/utilities/versioning.py`

**Classes**:
- `SemanticVersion`: Version representation and bumping
- `ArtifactLifecycle`: State machine and transitions
- `ArtifactAgeDetector`: Age calculation and categorization
- `VersionManager`: Frontmatter extraction/updates
- `VersionValidator`: Version format validation

## Quality Assurance

### Testing

All versioning utilities include:
- Type hints with modern syntax (X | None, tuple[...])
- Comprehensive docstrings
- Error handling for missing/malformed data
- Support for multiple YAML frontmatter formats

### Future Enhancements

**Phase 5**: Batch migration tools for updating artifact versions
**Phase 6**: Automated archival workflows
**Phase 7**: Version history tracking and rollback

## See Also

- [Toolkit Deprecation Roadmap](./toolkit-deprecation-roadmap.md)
- [Migration Guide](../../../.copilot/context/migration-guide.md)
- [AgentQMS Schema Reference](../../conventions/schemas/README.md)
- [Artifact Workflow Documentation](../../knowledge/agent/README.md)
