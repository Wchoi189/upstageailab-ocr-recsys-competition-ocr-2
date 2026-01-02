# AgentQMS MCP Server

## Overview

The AgentQMS MCP server exposes artifact workflow tools and standards as discoverable resources and executable tools, dramatically improving productivity by eliminating the need to remember CLI paths and syntax.

## Purpose

- **Discoverability**: Make hidden CLI tools easily discoverable
- **Standardization**: Enforce artifact standards automatically
- **Reduced Verbosity**: Schema-focused, low memory footprint
- **Automation**: Auto-validation, auto-indexing, auto-tracking

## Available Resources

Resources provide read-only access to standards and configuration. Use `read_resource` to access.

### 1. `agentqms://standards/index`
**Standards Index - Complete 4-Tier Hierarchy**

The master index showing all standards organized by tier:
- **Tier 1 (SST)**: System-Spanning Truths - architecture, naming, file placement
- **Tier 2 (Framework)**: Project frameworks - contracts, tools, coding standards
- **Tier 3 (Agents)**: Agent-specific configs - Claude, Gemini, Cursor, Copilot
- **Tier 4 (Workflows)**: Specific workflows - compliance, gap analysis

**Usage**:
```python
content = read_resource(
    ServerName="agentqms",
    Uri="agentqms://standards/index"
)
```

### 2. `agentqms://standards/artifact_types`
**Artifact Types Taxonomy**

Defines the complete artifact type system:
- Allowed types: assessment, audit, bug_report, design_document, implementation_plan, completed_plan, vlm_report
- Locations for each type
- Naming conventions
- Prohibited types and their replacements

**Usage**:
```python
content = read_resource(
    ServerName="agentqms",
    Uri="agentqms://standards/artifact_types"
)
```

### 3. `agentqms://standards/workflows`
**Workflow Requirements**

Workflow validation protocols and requirements for artifact creation, validation, and lifecycle management.

### 4. `agentqms://templates/list`
**Template Catalog (Dynamic)**

Dynamically generated list of available artifact templates. Returns JSON.

**Example Response**:
```json
{
  "templates": [
    "assessment",
    "audit",
    "bug_report",
    "design_document",
    "implementation_plan",
    "completed_plan",
    "vlm_report"
  ]
}
```

### 5. `agentqms://config/settings`
**QMS Settings**

Framework configuration including:
- Validation rules (strict mode, naming, frontmatter, structure)
- Path mappings
- Tool mappings
- Automation settings

---

## Available Tools

Tools perform operations with side effects. Use `call_tool` to execute.

### 1. `create_artifact`
**Create New Artifact with Auto-Validation**

Creates a new artifact following all project standards, with automatic validation, indexing, and tracking.

**Parameters**:
- `artifact_type` (required): One of the allowed types
- `name` (required): Kebab-case identifier (e.g., "api-refactor")
- `title` (required): Human-readable title
- `description` (optional): Detailed description
- `tags` (optional): Comma-separated tags

**Example**:
```python
result = call_tool(
    ServerName="agentqms",
    ToolName="create_artifact",
    Arguments={
        "artifact_type": "implementation_plan",
        "name": "kie-optimization",
        "title": "KIE Pipeline Optimization",
        "description": "Optimize KIE training pipeline for performance",
        "tags": "optimization,kie,performance"
    }
)
```

**Returns**:
```json
{
  "success": true,
  "file_path": "docs/artifacts/implementation_plans/2026-01-02_1905_implementation_plan_kie-optimization.md",
  "message": "Created implementation_plan: ..."
}
```

**What Happens Automatically**:
1. ✅ Creates artifact from template
2. ✅ Validates naming and structure
3. ✅ Registers in tracking database
4. ✅ Updates artifact indexes
5. ✅ Suggests next steps

### 2. `validate_artifact`
**Validate Against Standards**

Validates artifact(s) against naming conventions and structure standards.

**Parameters**:
- `file_path` (optional): Path to specific artifact
- `validate_all` (optional): Set to `true` to validate all artifacts

**Example - Single File**:
```python
result = call_tool(
    ServerName="agentqms",
    ToolName="validate_artifact",
    Arguments={
        "file_path": "docs/artifacts/implementation_plans/2026-01-02_1905_implementation_plan_kie-optimization.md"
    }
)
```

**Example - All Files**:
```python
result = call_tool(
    ServerName="agentqms",
    ToolName="validate_artifact",
    Arguments={
        "validate_all": true
    }
)
```

### 3. `list_artifact_templates`
**List Available Templates**

Returns all available artifact templates with details.

**Example**:
```python
result = call_tool(
    ServerName="agentqms",
    ToolName="list_artifact_templates",
    Arguments={}
)
```

**Returns**:
```json
{
  "templates": [
    "assessment",
    "audit",
    "bug_report",
    "design_document",
    "implementation_plan",
    "completed_plan",
    "vlm_report"
  ]
}
```

### 4. `check_compliance`
**Check Overall Compliance**

Generates a compliance report showing validation status of all artifacts.

**Example**:
```python
result = call_tool(
    ServerName="agentqms",
    ToolName="check_compliance",
    Arguments={}
)
```

**Returns**:
```json
{
  "total_files": 45,
  "valid_files": 43,
  "invalid_files": 2,
  "compliance_rate": 95.6,
  "violations": [...]
}
```

---

## Typical Workflows

### Creating an Implementation Plan

```python
# 1. List available templates (optional)
templates = call_tool(
    ServerName="agentqms",
    ToolName="list_artifact_templates",
    Arguments={}
)

# 2. Create the artifact
result = call_tool(
    ServerName="agentqms",
    ToolName="create_artifact",
    Arguments={
        "artifact_type": "implementation_plan",
        "name": "feature-name",
        "title": "Feature Implementation",
        "tags": "feature,enhancement"
    }
)

# 3. Artifact is auto-validated, indexed, and tracked
# File path returned in result["file_path"]
```

### Checking Artifact Compliance

```python
# Check overall compliance
report = call_tool(
    ServerName="agentqms",
    ToolName="check_compliance",
    Arguments={}
)

# If violations found, validate specific files
if report["invalid_files"] > 0:
    for violation in report["violations"]:
        result = call_tool(
            ServerName="agentqms",
            ToolName="validate_artifact",
            Arguments={"file_path": violation["file"]}
        )
```

---

## Technical Details

### Server Implementation
- **Location**: `AgentQMS/mcp_server.py`
- **Dependencies**: `mcp`, `AgentQMS.tools.core.artifact_workflow`
- **Auto-Discovery**: Finds project root by locating `AgentQMS/` directory

### URI Scheme
All resources use the `agentqms://` scheme:
- `agentqms://standards/*` - Standards and rules
- `agentqms://templates/*` - Template information
- `agentqms://config/*` - Configuration

### Error Handling
- Unknown resources return clear error messages
- Missing files are reported with specific paths
- Tool errors return structured JSON with error details

### Integration with Existing Tools
The MCP server wraps the existing `ArtifactWorkflow` class from `AgentQMS/tools/core/artifact_workflow.py`, preserving all existing functionality while making it discoverable.

---

## Benefits

✅ **No Path Memorization**: Tools discoverable via MCP
✅ **Standards Enforced**: Automatic validation ensures compliance
✅ **Reduced Friction**: Create artifacts with single tool call
✅ **Auto-Indexing**: Indexes updated automatically
✅ **Auto-Tracking**: Artifacts registered in tracking DB
✅ **Clear Errors**: Structured error responses
✅ **Low Memory**: Schema-focused, minimal footprint

---

## Troubleshooting

### Tool Not Found
Ensure the AI assistant has been restarted to reload MCP configuration.

### Import Errors
The server requires `AgentQMS` to be in the Python path. Verify it runs from project root:
```bash
uv run --directory /workspaces/upstageailab-ocr-recsys-competition-ocr-2 python AgentQMS/mcp_server.py
```

### Validation Failures
Check the specific error messages returned. Common issues:
- Incorrect artifact type
- Non-kebab-case name
- Missing required frontmatter fields
