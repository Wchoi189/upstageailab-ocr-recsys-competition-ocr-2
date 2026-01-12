---
title: "Plugin Inspection MCP Resource Design"
doc_id: "design-plugin-artifact-types-mcp-resource"
version: "1.0"
status: "draft"
type: "design_document"
category: "architecture"
ads_version: "1.0"
date: "2026-01-10 10:00 (KST)"
tags: [mcp, plugin, discovery, design]
---

# Plugin Inspection MCP Resource Design

**Resource URI**: `agentqms://plugins/artifact_types`

**Purpose**: Expose plugin artifact type metadata and validation rules through MCP, enabling dynamic discovery and inspection of available artifact types from AI agents and MCP clients.

## Executive Summary

Currently, artifact types are hardcoded in the MCP tool schema, making the system:
- **Brittle**: Schema must be manually updated when plugins are added
- **Opaque**: MCP clients cannot discover available types at runtime
- **Inflexible**: Users cannot add custom artifact types without code changes

This resource design provides:
- **Dynamic Discovery**: All available artifact types (hardcoded + plugins + standards)
- **Metadata Exposure**: Source, validation rules, template information
- **Runtime Inspection**: AI agents can query available types without schema hardcoding
- **Extensibility**: Users can add custom types via plugin system

## Resource Specification

### URI
```
agentqms://plugins/artifact_types
```

### MIME Type
```
application/json
```

### HTTP Equivalent
```
GET /plugins/artifact_types
```

## Response Schema

### Response Root Object

```json
{
  "artifact_types": [
    { "artifact_type_details" }
  ],
  "summary": {
    "total": 11,
    "sources": {
      "hardcoded": 8,
      "plugin": 3
    },
    "validation_enabled": true,
    "last_updated": "2026-01-10T10:30:00Z"
  },
  "metadata": {
    "version": "1.0",
    "plugin_discovery_enabled": true,
    "conflict_detection": true
  }
}
```

### Artifact Type Detail Object

Each artifact type in `artifact_types` array:

```json
{
  "name": "audit",
  "source": "plugin",
  "description": "Framework audits, compliance checks, and quality evaluations",
  "category": "compliance",
  "version": "1.0",

  "metadata": {
    "filename_pattern": "{date}_audit-{name}.md",
    "directory": "audits/",
    "template_variables": {
      "date": "2025-11-29",
      "category": "compliance",
      "status": "draft"
    }
  },

  "validation": {
    "required_fields": ["title", "date", "type", "status", "category"],
    "required_sections": ["## Executive Summary", "## Findings"],
    "allowed_statuses": ["draft", "active", "completed"],
    "filename_prefix": "audit-"
  },

  "frontmatter": {
    "type": "audit",
    "category": "compliance",
    "status": "draft",
    "version": "1.0",
    "tags": ["audit"]
  },

  "template_preview": {
    "first_300_chars": "# Audit: {title}\n\n**Audit Date**: {date}\n...",
    "line_count": 120,
    "sections": ["# Audit", "## Executive Summary", "## Findings"]
  },

  "plugin_info": {
    "plugin_name": "audit",
    "plugin_path": ".agentqms/plugins/artifact_types/audit.yaml",
    "plugin_scope": "project"
  },

  "conflicts": {
    "exists_in_multiple_sources": false,
    "conflict_sources": []
  }
}
```

### Field Definitions

| Field                                  | Type    | Description                                                                   |
| -------------------------------------- | ------- | ----------------------------------------------------------------------------- |
| `name`                                 | string  | Artifact type identifier (lowercase, kebab-case)                              |
| `source`                               | enum    | Source: "hardcoded", "plugin", "standards", or "hardcoded (plugin available)" |
| `description`                          | string  | Human-readable description of artifact type                                   |
| `category`                             | string  | Category for organization (e.g., "compliance", "development")                 |
| `version`                              | string  | Schema version of this artifact type (e.g., "1.0")                            |
| `metadata.filename_pattern`            | string  | Pattern for generated filename                                                |
| `metadata.directory`                   | string  | Artifact storage directory                                                    |
| `metadata.template_variables`          | object  | Default template variables for this type                                      |
| `validation.required_fields`           | array   | Required frontmatter fields                                                   |
| `validation.required_sections`         | array   | Required markdown sections                                                    |
| `validation.allowed_statuses`          | array   | Allowed status values                                                         |
| `validation.filename_prefix`           | string  | Expected filename prefix                                                      |
| `frontmatter.*`                        | object  | Default frontmatter structure                                                 |
| `template_preview.first_300_chars`     | string  | First 300 chars of template (truncated)                                       |
| `template_preview.line_count`          | number  | Total lines in template                                                       |
| `template_preview.sections`            | array   | Top-level markdown sections                                                   |
| `plugin_info.plugin_name`              | string  | Name of plugin providing this type                                            |
| `plugin_info.plugin_path`              | string  | Path to plugin YAML file                                                      |
| `plugin_info.plugin_scope`             | string  | Plugin scope ("project", "framework")                                         |
| `conflicts.exists_in_multiple_sources` | boolean | True if defined in multiple sources                                           |
| `conflicts.conflict_sources`           | array   | List of sources defining this type                                            |

## Usage Examples

### Example 1: Discover All Available Types

```bash
curl http://localhost:8000/plugins/artifact_types
```

**Use case**: AI agent listing all available artifact types to display to user

**Response**: Full array of all 11+ artifact types with complete metadata

### Example 2: Discover Custom/Plugin Types Only

```bash
curl http://localhost:8000/plugins/artifact_types | \
  jq '.artifact_types[] | select(.source == "plugin")'
```

**Use case**: Find types contributed by plugins, not built-in

**Response**: 3 plugin types (audit, change_request, ocr_experiment_report)

### Example 3: Find Type by Name

```bash
curl http://localhost:8000/plugins/artifact_types | \
  jq '.artifact_types[] | select(.name == "audit")'
```

**Use case**: Get detailed metadata for specific artifact type

**Response**: Single artifact type with all fields

### Example 4: Check for Conflicts

```bash
curl http://localhost:8000/plugins/artifact_types | \
  jq '.artifact_types[] | select(.conflicts.exists_in_multiple_sources == true)'
```

**Use case**: Identify naming conflicts between systems

**Response**: Conflicting artifact types (if any exist)

### Example 5: Find Types by Category

```bash
curl http://localhost:8000/plugins/artifact_types | \
  jq '.artifact_types[] | select(.category == "compliance")'
```

**Use case**: Filter artifact types by functional category

**Response**: Types in compliance category (e.g., "audit")

## Implementation Plan

### Phase 1: Core Discovery (Session 1 - COMPLETED)

**Status**: ✅ Done

Deliverables:
- ✅ `_get_available_artifact_types()` method implemented
- ✅ Metadata collection from all sources
- ✅ Conflict detection logic
- ✅ MCP helper function `_get_available_artifact_types()`

### Phase 2: MCP Resource Definition (Session 2 - TODO)

**Estimated**: 1-2 hours

Tasks:
1. Add `agentqms://plugins/artifact_types` to RESOURCES list in mcp_server.py
2. Implement `_get_plugin_artifact_types()` handler function
3. Format response according to this schema
4. Add resource to `list_resources()` response

Code location:
- File: `AgentQMS/mcp_server.py`
- Lines: ~100-150 (resources definition area)

### Phase 3: Resource Handler (Session 2 - TODO)

**Estimated**: 1-2 hours

Add async handler:
```python
@app.read_resource()
async def read_resource(uri: str) -> list[ReadResourceContents]:
    # ... existing handlers ...

    if uri == "agentqms://plugins/artifact_types":
        content = await _get_plugin_artifact_types()
        return [ReadResourceContents(content=content, mime_type="application/json")]
```

### Phase 4: Testing (Session 2 - TODO)

**Estimated**: 1-2 hours

Tests to add:
- Resource discovery (list_resources includes new URI)
- Resource reading (read_resource returns valid JSON)
- Response schema validation
- All artifact types included
- Metadata completeness
- Conflict detection accuracy

Test file:
- File: `tests/test_mcp_plugin_resource.py`

### Phase 5: Client Integration (Session 3 - TODO)

**Estimated**: 1 hour

Update MCP clients to use dynamic discovery:
- Remove hardcoded enum from mcp tool schema
- Generate enum dynamically from resource
- Add type checking validation

## API Evolution

### Current State (Baseline)

```json
{
  "artifact_types": ["assessment", "audit", "bug_report", ...]
}
```

- Simple array of strings
- No metadata
- No source information
- No validation rules exposed

### Phase 1 Result (In Progress)

```json
{
  "templates": [...],
  "summary": {
    "total": 11,
    "hardcoded": 8,
    "plugin": 3,
    "with_conflicts": 0
  }
}
```

- Enhanced template list with metadata
- Summary statistics
- Source tracking

### Phase 2 Result (Proposed)

```json
{
  "artifact_types": [
    {
      "name": "audit",
      "source": "plugin",
      "validation": {...},
      "metadata": {...}
    }
  ],
  "summary": {...}
}
```

- Complete artifact type catalog
- Validation rules exposed
- Template information included
- Full plugin metadata

## Backward Compatibility

### Migration Path

1. **Stage 1** (Current): Keep hardcoded enum, add new resource
   - Old clients continue working
   - New resource available for discovery
   - Zero breaking changes

2. **Stage 2** (Phase 5): Make enum dynamic
   - Schema still valid
   - Values now from resource
   - Old clients still work if they hardcoded values

3. **Stage 3** (Future): Deprecate hardcoded enum
   - Recommend using resource
   - Keep enum for 1-2 versions
   - Clear deprecation warning

### Compatibility Guarantees

- ✅ Existing artifact creation workflows unaffected
- ✅ MCP clients using hardcoded enums continue working
- ✅ New resource available immediately (optional)
- ✅ No breaking changes in Phase 1-2

## Performance Considerations

### Response Size

Current estimate:
- Per artifact type: 800 bytes (name + validation + metadata)
- Total response: ~11 types × 800 bytes = ~9 KB
- Acceptable for network transmission

Optimization:
- Implement response caching (1-5 minute TTL)
- Add `Cache-Control: max-age=300` header
- Invalidate on plugin changes

### Discovery Latency

- Plugin loading: ~50-100ms (first call, then cached)
- Template gathering: ~10-20ms
- JSON serialization: ~5-10ms
- Total: ~100-150ms (acceptable)

### Caching Strategy

```python
class PluginArtifactTypesCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.cache = None
        self.cache_time = None

    def get(self):
        # Return cached if fresh
        # Otherwise rebuild and cache
```

## Security Considerations

- ✅ Read-only resource (no mutation)
- ✅ No credentials exposed
- ✅ No sensitive paths in response
- ✅ Plugin paths sanitized for display
- ✅ Validation rules exposed (safe to show)

## Integration Points

### With MCP Tool `create_artifact`

Currently:
```json
"artifact_type": {
  "type": "string",
  "enum": ["assessment", "audit", ...] // Hardcoded
}
```

After Phase 5:
```python
# Enum generated from resource at startup
available_types = await get_plugin_artifact_types_list()
schema["properties"]["artifact_type"]["enum"] = available_types
```

### With MCP Resource `agentqms://templates/list`

Relationship:
- `agentqms://templates/list`: Flat list of template names
- `agentqms://plugins/artifact_types`: Complete artifact type catalog with validation

Both accessible; templates/list is lightweight, plugins/artifact_types is comprehensive

### With Artifact Workflow

The `ArtifactWorkflow` class will use dynamic discovery:
```python
workflow = ArtifactWorkflow()
types = workflow.templates._get_available_artifact_types()

# Validation uses types from discovery
```

## Success Criteria

✅ Phase 1 (COMPLETE):
- ✅ `_get_available_artifact_types()` implemented
- ✅ Dynamic discovery functional
- ✅ 19 tests passing
- ✅ Source metadata tracked

📋 Phase 2 (TO DO):
- ⏳ MCP resource defined and registered
- ⏳ Resource handler implemented
- ⏳ Response schema valid
- ⏳ All 11+ types included
- ⏳ Tests for resource

🎯 Phase 3 (TO DO):
- ⏳ Schema enum fully dynamic
- ⏳ Client integration tested
- ⏳ Backward compatibility verified

## Known Limitations

1. **Plugin Hot-Reload**: Currently requires application restart
   - Future improvement: Watch plugin directory for changes

2. **Distributed Systems**: Cache invalidation not yet implemented
   - Future: Add cache invalidation webhook for multi-instance deployments

3. **Standards Integration**: Standards YAML not yet integrated
   - Future: Include standards metadata in response

## References

- [AgentQMS MCP Server](../../AgentQMS/mcp_server.py)
- [Artifact Templates](../../AgentQMS/tools/core/artifact_templates.py)
- [Plugin System](../../AgentQMS/tools/core/plugins/__init__.py)
- [Session 1 Roadmap](../roadmap/00_agentqms_artifact_consolidation.yaml)

## Document History

| Version | Date       | Changes                 |
| ------- | ---------- | ----------------------- |
| 1.0     | 2026-01-10 | Initial design document |

---

**Document Status**: Ready for implementation review
**Phase**: Design complete, ready for Session 2 development
**Author**: AI Agent
**Last Updated**: 2026-01-10
