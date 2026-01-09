---
title: "Creating Artifact Type Plugins"
version: "1.0"
last_updated: "2026-01-10"
status: active
category: developer-guide
tags: [plugins, artifacts, development, AgentQMS]
---

# Creating Custom Artifact Type Plugins

This guide explains how to create custom artifact types for AgentQMS using the plugin system. After completing Phase 4 of the artifact consolidation roadmap, **all artifact types are defined via plugins** - there are no hardcoded templates.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Plugin Structure](#plugin-structure)
3. [Required Fields](#required-fields)
4. [Optional Features](#optional-features)
5. [Validation Rules](#validation-rules)
6. [Template System](#template-system)
7. [Frontmatter Configuration](#frontmatter-configuration)
8. [Testing Your Plugin](#testing-your-plugin)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Quick Start

### 1. Create Plugin File
Create a YAML file in `AgentQMS/.agentqms/plugins/artifact_types/`:

```yaml
# my_artifact.yaml
name: my_artifact
version: "1.0"
description: "Brief description of this artifact type"
scope: project

metadata:
  filename_pattern: "{date}_{name}.md"
  directory: my_artifacts/
  frontmatter:
    ads_version: "1.0"
    type: my_artifact
    category: development
    status: draft
    version: "1.0"
    tags: [custom, my_artifact]

validation:
  required_fields:
    - title
    - date
    - type
    - status
  required_sections:
    - "## Purpose"
    - "## Content"

template: |
  # {title}
  
  ## Purpose
  Describe the purpose of this artifact.
  
  ## Content
  Main content goes here.
```

### 2. Validate Plugin
The plugin is automatically validated on load. Check for errors:

```python
from AgentQMS.tools.core.plugins import get_plugin_registry

registry = get_plugin_registry()
if registry.has_errors():
    for error in registry.validation_errors:
        print(f"{error.plugin_path}: {error.error_message}")
```

### 3. Use Your Plugin
Your artifact type will automatically appear in the MCP schema:

```python
import asyncio
from AgentQMS.mcp_server import list_tools

async def check_available():
    tools = await list_tools()
    create_artifact = next(t for t in tools if t.name == "create_artifact")
    enum = create_artifact.inputSchema["properties"]["artifact_type"]["enum"]
    print(f"Available types: {enum}")

asyncio.run(check_available())
```

## Plugin Structure

### File Naming Convention
- **Location:** `AgentQMS/.agentqms/plugins/artifact_types/`
- **Filename:** `{artifact_type}.yaml` (e.g., `bug_report.yaml`, `design_document.yaml`)
- **Format:** YAML (must parse successfully)

### JSON Schema
All plugins must conform to:
```
AgentQMS/standards/schemas/plugin_artifact_type.json
```

### Validation Rules
Centralized validation rules are in:
```
.agentqms/schemas/artifact_type_validation.yaml
```

Key validation checks:
- **Canonical Types:** Only names in canonical_types list are allowed
- **Prohibited Types:** Names in prohibited_types are rejected with suggestions
- **Metadata Requirements:** ads_version, type, category, status required
- **Naming Convention:** Follow `YYYY-MM-DD_HHMM_{type}_{description}.md`

## Required Fields

Every plugin MUST have these top-level fields:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | string | Unique identifier (lowercase, underscores) | `change_request` |
| `version` | string | Semantic version | `"1.0"`, `"2.1.0"` |
| `metadata` | object | File generation and frontmatter config | See below |
| `template` | string | Markdown template with placeholders | See [Template System](#template-system) |

### Metadata Object Structure
```yaml
metadata:
  filename_pattern: "{date}_CR_{name}.md"     # REQUIRED
  directory: "change_requests/"               # REQUIRED
  frontmatter:                                # REQUIRED
    ads_version: "1.0"                        # REQUIRED (ADS compliance)
    type: change_request                      # REQUIRED (matches plugin name)
    category: planning                        # REQUIRED
    status: draft                             # REQUIRED
    version: "1.0"                            # REQUIRED
    tags: [change, planning]                  # Recommended
```

**Critical:** The `frontmatter.ads_version` field is required for ADS v1.0 compliance. Missing this field will cause validation failure.

## Optional Fields

### Description
```yaml
description: |
  Multi-line description of this artifact type.
  Explain when and how it should be used.
```

### Extends (Inheritance)
```yaml
extends: base_document
# Inherits metadata, validation, and template from base_document plugin
```

### Scope (Namespace)
```yaml
scope: project          # Default
# scope: builtin        # Framework-provided types
# scope: org_acme       # Organization-specific types
```

### Template Variables
```yaml
template_variables:
  author: "[Author Name]"
  priority: "Medium"
  deadline: "[YYYY-MM-DD]"
```

## Validation Rules

### Required Fields Validation
```yaml
validation:
  required_fields:
    - title
    - date
    - type
    - category
    - status
    - severity      # Custom field for bug reports
```

### Required Sections
```yaml
validation:
  required_sections:
    - "## Summary"
    - "## Background"
    - "## Proposal"
    - "## Impact Assessment"
```

### Filename Prefix
```yaml
validation:
  filename_prefix: "CR_"     # Enforces filenames start with CR_
```

### Allowed Statuses
```yaml
validation:
  allowed_statuses:
    - draft
    - review
    - approved
    - rejected
    - implemented
    - archived
```

### Custom Validators (Advanced)
```yaml
validation:
  custom_validators:
    - name: "Risk Assessment Validator"
      module: "my_org.validators.risk"
      function: "validate_risk_levels"
      config:
        max_high_risks: 3
        require_mitigation: true
```

## Template System

### Basic Template
Templates use `{placeholder}` syntax for variable substitution:

```yaml
template: |
  # {title}
  
  **Date:** {date}
  **Author:** {author}
  
  ## Overview
  {overview}
```

### Available Built-in Variables
- `{title}` - Artifact title
- `{date}` - Creation date (YYYY-MM-DD)
- `{time}` - Creation time (HHMM)
- `{name}` - Kebab-case name
- `{type}` - Artifact type

### Custom Variables
Define custom variables with defaults:

```yaml
template_variables:
  project_phase: "Phase 1"
  stakeholders: "[List stakeholders]"
  budget: "TBD"
  
template: |
  # Project Planning - {title}
  
  **Phase:** {project_phase}
  **Stakeholders:** {stakeholders}
  **Budget:** {budget}
```

### Advanced Template Features

#### Conditional Sections
```yaml
template: |
  # {title}
  
  ## Summary
  {summary}
  
  {{if priority == "high"}}
  ## Urgency Notice
  ⚠️ This is a high-priority item requiring immediate attention.
  {{endif}}
```

#### Loops (Phase 2 Feature)
```yaml
template: |
  ## Team Members
  {{for member in team_members}}
  - **{member.name}** ({member.role})
  {{endfor}}
```

## Frontmatter Configuration

### Minimal Frontmatter
```yaml
metadata:
  frontmatter:
    ads_version: "1.0"      # REQUIRED
    type: my_artifact       # REQUIRED
    category: development   # REQUIRED
    status: draft           # REQUIRED
    version: "1.0"          # REQUIRED
```

### Rich Frontmatter Example
```yaml
metadata:
  frontmatter:
    ads_version: "1.0"
    type: experiment_report
    category: research
    status: active
    version: "1.0"
    
    # Custom fields
    experiment_id: "EXP-{id}"
    hypothesis: ""
    methodology: ""
    start_date: "{date}"
    end_date: ""
    researcher: "{author}"
    funding_source: "Internal"
    
    # Classification
    tags: [experiment, research, analysis]
    priority: "medium"
    visibility: "internal"
    
    # Tracking
    related_artifacts: []
    dependencies: []
```

## Testing Your Plugin

### 1. Validation Test
```python
from AgentQMS.tools.core.plugins import get_plugin_registry

registry = get_plugin_registry(force=True)  # Reload from disk

# Check if your plugin loaded
artifact_types = registry.get_artifact_types()
if "my_artifact" in artifact_types:
    print("✅ Plugin loaded successfully")
else:
    print("❌ Plugin failed to load")
    
# Check for validation errors
if registry.has_errors():
    for error in registry.validation_errors:
        if "my_artifact" in error.plugin_path:
            print(f"Error: {error.error_message}")
```

### 2. Template Test
```python
from AgentQMS.tools.core.artifact_templates import ArtifactTemplates

templates = ArtifactTemplates()
available = templates.get_available_templates()

if "my_artifact" in available:
    print("✅ Template available")
    
    # Test template rendering
    metadata = templates._get_available_artifact_types()
    my_template = metadata["my_artifact"]["template"]
    print(my_template.get("content_template", "")[:200])
```

### 3. MCP Schema Test
```python
import asyncio
from AgentQMS.mcp_server import list_tools

async def test_mcp_schema():
    tools = await list_tools()
    create_artifact = next(t for t in tools if t.name == "create_artifact")
    enum = create_artifact.inputSchema["properties"]["artifact_type"]["enum"]
    
    if "my_artifact" in enum:
        print("✅ Artifact type appears in MCP schema")
    else:
        print("❌ Artifact type NOT in MCP schema")
        print(f"Available: {enum}")

asyncio.run(test_mcp_schema())
```

### 4. End-to-End Creation Test
```bash
# Using MCP server
cd AgentQMS/bin
make create-plan NAME=test-artifact TITLE="Test My Plugin"

# Check if artifact was created in correct directory
ls -la docs/artifacts/my_artifacts/
```

## Troubleshooting

### Common Validation Errors

#### Error: "Unknown artifact type 'my_artifact'"
**Cause:** Plugin name not in canonical_types list.

**Fix:** Check `.agentqms/schemas/artifact_type_validation.yaml`:
```yaml
canonical_types:
  # Add your type here if it should be canonical
  - my_artifact
```

Or use an existing canonical name if this is a variant.

#### Error: "Missing required frontmatter field: ads_version"
**Cause:** Frontmatter missing ADS v1.0 compliance field.

**Fix:** Add to metadata.frontmatter:
```yaml
metadata:
  frontmatter:
    ads_version: "1.0"  # Add this
    type: my_artifact
    # ... rest of frontmatter
```

#### Error: "Prohibited artifact type 'template'"
**Cause:** Using a prohibited type name.

**Fix:** Rename your plugin:
```yaml
# Prohibited types and their replacements:
# - template → use a specific type (e.g., "document_template")
# - design → use "design_document"
# - research → use "assessment"
```

#### Error: "Plugin file does not parse as valid YAML"
**Cause:** YAML syntax error.

**Fix:** Validate YAML syntax:
```bash
python -c "import yaml; yaml.safe_load(open('my_artifact.yaml'))"
```

Common issues:
- Missing quotes around strings with special characters
- Incorrect indentation (use 2 spaces, not tabs)
- Unmatched brackets or quotes

#### Error: "Missing required field: metadata.directory"
**Cause:** Incomplete metadata section.

**Fix:** Ensure metadata has all required fields:
```yaml
metadata:
  filename_pattern: "{date}_{name}.md"    # REQUIRED
  directory: "my_artifacts/"              # REQUIRED (was missing)
  frontmatter:                            # REQUIRED
    ads_version: "1.0"
    type: my_artifact
```

### Plugin Not Appearing in Enum

If your plugin passes validation but doesn't appear in the MCP schema enum:

1. **Check Plugin Registry:**
   ```python
   from AgentQMS.tools.core.plugins import get_plugin_registry
   registry = get_plugin_registry(force=True)
   types = registry.get_artifact_types()
   print("Loaded types:", list(types.keys()))
   ```

2. **Check Template Availability:**
   ```python
   from AgentQMS.tools.core.artifact_templates import ArtifactTemplates
   templates = ArtifactTemplates()
   available = templates.get_available_templates()
   print("Available templates:", available)
   ```

3. **Force Schema Reload:**
   ```python
   # MCP server caches schema; restart to pick up changes
   # Or call list_tools() which regenerates dynamically
   ```

### Debugging Tips

#### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from AgentQMS.tools.core.plugins import get_plugin_registry
registry = get_plugin_registry(force=True)
```

#### Check Plugin Discovery Paths
```python
from AgentQMS.tools.core.plugins import get_plugin_loader

loader = get_plugin_loader()
paths = loader.get_discovery_paths()
print("Plugin discovery paths:", paths)
```

#### Validate Against JSON Schema
```bash
# Install jsonschema CLI tool
pip install check-jsonschema

# Validate your plugin
check-jsonschema \
  --schemafile AgentQMS/standards/schemas/plugin_artifact_type.json \
  AgentQMS/.agentqms/plugins/artifact_types/my_artifact.yaml
```

## Examples

### Example 1: Simple Document Type
```yaml
name: meeting_notes
version: "1.0"
description: "Quick meeting notes with attendees and action items"
scope: project

metadata:
  filename_pattern: "{date}_meeting_{name}.md"
  directory: "meetings/"
  frontmatter:
    ads_version: "1.0"
    type: meeting_notes
    category: documentation
    status: active
    version: "1.0"
    tags: [meeting, notes]

validation:
  required_fields:
    - title
    - date
    - attendees
  required_sections:
    - "## Attendees"
    - "## Discussion"
    - "## Action Items"

template: |
  # Meeting Notes: {title}
  
  **Date:** {date}
  
  ## Attendees
  - {attendees}
  
  ## Discussion
  ### Topic 1
  - Key points discussed
  
  ## Action Items
  - [ ] Action 1 (Owner: Name, Due: Date)

template_variables:
  attendees: "[List attendees]"
```

### Example 2: Technical Specification
```yaml
name: tech_spec
version: "1.0"
description: "Technical specification with architecture diagrams and API contracts"
scope: project

metadata:
  filename_pattern: "{date}_spec_{name}.md"
  directory: "specifications/"
  frontmatter:
    ads_version: "1.0"
    type: tech_spec
    category: design
    status: draft
    version: "1.0"
    complexity: "medium"
    review_status: "pending"
    tags: [specification, technical, design]

validation:
  required_fields:
    - title
    - date
    - type
    - status
    - complexity
    - version
  required_sections:
    - "## Overview"
    - "## Architecture"
    - "## API Specification"
    - "## Data Models"
    - "## Security Considerations"
  allowed_statuses:
    - draft
    - in_review
    - approved
    - deprecated

template: |
  # Technical Specification: {title}
  
  **Version:** {version}
  **Complexity:** {complexity}
  **Status:** {status}
  
  ## Overview
  ### Purpose
  {purpose}
  
  ### Scope
  {scope}
  
  ## Architecture
  ### System Components
  {components}
  
  ### Diagram
  ```mermaid
  graph TD
      A[Component A] --> B[Component B]
      B --> C[Component C]
  ```
  
  ## API Specification
  ### Endpoints
  | Method | Path | Description |
  |--------|------|-------------|
  | GET | /api/resource | Get resource |
  
  ## Data Models
  ### Model: {model_name}
  ```json
  {
    "field1": "type",
    "field2": "type"
  }
  ```
  
  ## Security Considerations
  - **Authentication:** {auth_method}
  - **Authorization:** {authz_model}
  - **Data Protection:** {data_protection}
  
  ## Implementation Notes
  {implementation_notes}

template_variables:
  purpose: "[Describe the purpose]"
  scope: "[Define the scope]"
  components: "[List system components]"
  model_name: "ResourceModel"
  auth_method: "OAuth 2.0"
  authz_model: "RBAC"
  data_protection: "TLS 1.3 + AES-256"
  implementation_notes: "[Add implementation guidance]"
```

### Example 3: Bug Report (Complete)
See `AgentQMS/.agentqms/plugins/artifact_types/bug_report.yaml` for a production example with:
- Rich frontmatter (severity, priority, assignee)
- Comprehensive validation rules
- Structured template with error logging
- Template variables for all fields

## Best Practices

### 1. Naming Conventions
- Use `snake_case` for plugin names
- Keep names descriptive but concise (2-3 words max)
- Avoid generic names like "document" or "file"
- Check `.agentqms/schemas/artifact_type_validation.yaml` for prohibited names

### 2. Frontmatter Design
- Always include `ads_version`, `type`, `category`, `status`, `version`
- Add domain-specific fields (e.g., `severity` for bugs, `priority` for tasks)
- Use arrays for multi-value fields (`tags`, `related_artifacts`)
- Provide sensible defaults in frontmatter section

### 3. Template Quality
- Start with a clear title using `{title}` variable
- Organize with markdown headers (##, ###)
- Include placeholder text with `[descriptive instructions]`
- Add checklist items for action-oriented artifacts
- Use code blocks for technical content
- Consider mermaid diagrams for architecture/flow

### 4. Validation Strategy
- Define `required_sections` for structure enforcement
- Use `required_fields` for critical frontmatter
- Set `allowed_statuses` for workflow enforcement
- Add `filename_prefix` for easy filtering (e.g., "BUG_", "RFC_")

### 5. Documentation
- Write clear `description` field explaining when to use this type
- Add comments in YAML for maintainer guidance
- Include example values in `template_variables`
- Document any custom validators

### 6. Testing
- Test plugin validation before committing
- Verify template renders correctly with sample data
- Check that artifact appears in MCP schema enum
- Create at least one artifact using the plugin
- Review generated artifact for completeness

## Integration with AgentQMS

### Automatic Features
Your plugin automatically gets:
- **MCP Integration:** Appears in create_artifact tool enum
- **Validation:** Enforced at artifact creation time
- **Directory Management:** Subdirectories created automatically
- **Filename Generation:** Pattern-based naming with date/time
- **Frontmatter Injection:** Auto-populated from plugin definition
- **Template Rendering:** Variables substituted at creation time

### Workflow Integration
Plugins integrate with AgentQMS workflows:
1. **Discovery:** Loaded from `.agentqms/plugins/artifact_types/`
2. **Validation:** Checked against JSON schema and validation rules
3. **Registration:** Added to PluginRegistry if valid
4. **Exposure:** MCP schema dynamically updated
5. **Creation:** Used by create_artifact tool
6. **Indexing:** Artifacts indexed in directory INDEX.md
7. **Tracking:** Status changes tracked in project compass

## Migration from Hardcoded Templates

If you previously had hardcoded templates in `artifact_templates.py`:

1. **Extract Template Data:**
   ```python
   # Old hardcoded format
   "my_type": {
       "description": "...",
       "template": {...}
   }
   ```

2. **Convert to Plugin:**
   - Create `my_type.yaml`
   - Map `description` → `description`
   - Map `template.directory` → `metadata.directory`
   - Map `template.frontmatter` → `metadata.frontmatter`
   - Map `template.content_template` → `template`
   - Add validation rules if any

3. **Test Equivalence:**
   ```python
   # Check that plugin produces same output as hardcoded
   from tests.test_plugin_vs_hardcoded_equivalence import test_equivalence
   test_equivalence("my_type")
   ```

4. **Deprecate Hardcoded:**
   - Remove from `artifact_templates.py`
   - Archive in `AgentQMS/tools/archive/`

See `docs/artifacts/implementation_plans/2026-01-10_0417_implementation_plan_phase4-hardcoded-removal-migration.md` for full migration guide.

## Further Reading

- **JSON Schema:** `AgentQMS/standards/schemas/plugin_artifact_type.json`
- **Validation Rules:** `.agentqms/schemas/artifact_type_validation.yaml`
- **Plugin Loader:** `AgentQMS/tools/core/plugins/loader.py`
- **Plugin Validator:** `AgentQMS/tools/core/plugins/validation.py`
- **Template System:** `AgentQMS/tools/core/artifact_templates.py`
- **MCP Server:** `AgentQMS/mcp_server.py` (dynamic schema)

## Support

### Getting Help
- **GitHub Issues:** Report bugs or request features
- **Documentation:** Check `AgentQMS/docs/guides/`
- **Examples:** Browse `.agentqms/plugins/artifact_types/` for working plugins
- **Test Suite:** Run `pytest AgentQMS/tests/test_artifact_type_validation.py`

### Contributing
- Submit plugin examples via PR
- Improve documentation
- Report validation issues
- Suggest new features

---

**Version:** 1.0  
**Last Updated:** 2026-01-10  
**Phase:** 6 (Developer Documentation - Session 8)  
**Related:** 
- Phase 3: Validation Schema & Naming Conflicts
- Phase 4: Hardcoded Template Removal
- Phase 5: Dynamic MCP Schema
