# Artifact Type Naming Conflicts Resolution

**Date:** 2026-01-10
**Phase:** Phase 3 - Session 5
**Status:** Resolved

## Summary

This document explains the resolution of naming conflicts between hardcoded templates, plugins, and standards YAML definitions for artifact types.

## Conflicts Identified

### 1. **design** vs **design_document**

**Issue:** Both `design.yaml` and `design_document.yaml` plugins existed with identical content.

**Resolution:**
- **Canonical Name:** `design_document`
- **Frontmatter Type:** `design` (for backward compatibility)
- **Plugin:** `design_document.yaml` (active)
- **Deprecated:** `design.yaml` → `design.yaml.deprecated`

**Rationale:** `design_document` is more descriptive and consistent with other artifact type names (e.g., `bug_report`, `vlm_report`, `implementation_plan`).

### 2. **research** → **assessment**

**Issue:** `research.yaml` plugin existed, but standards YAML prohibited `research` type and recommended using `assessment`.

**Resolution:**
- **Canonical Name:** `assessment`
- **Plugin:** `assessment.yaml` (active)
- **Deprecated:** `research.yaml` → `research.yaml.deprecated`

**Rationale:** Research is fundamentally a form of assessment/evaluation. Using a single type reduces confusion and aligns with project standards.

### 3. **template** → docs/_templates/

**Issue:** `template.yaml` artifact type plugin existed, but templates should not be artifacts.

**Resolution:**
- **Location:** Templates belong in `docs/_templates/` directory
- **Artifact Type:** Not applicable (templates are not artifacts)
- **Deprecated:** `template.yaml` → `template.yaml.deprecated`

**Rationale:** Templates are reusable content structures, not project artifacts. They should not appear in artifact tracking systems.

## Implementation Changes

### File Changes

```bash
# Deprecated plugins (kept for reference, not loaded)
AgentQMS/.agentqms/plugins/artifact_types/design.yaml.deprecated
AgentQMS/.agentqms/plugins/artifact_types/research.yaml.deprecated
AgentQMS/.agentqms/plugins/artifact_types/template.yaml.deprecated

# Active canonical plugins
AgentQMS/.agentqms/plugins/artifact_types/design_document.yaml  ✅
AgentQMS/.agentqms/plugins/artifact_types/assessment.yaml       ✅
```

### Validation Rules

Centralized validation rules in `.agentqms/schemas/artifact_type_validation.yaml`:

```yaml
canonical_types:
  design_document:
    frontmatter_type: "design"  # For backward compatibility
    aliases: ["design"]  # Legacy name recognized but warned

  assessment:
    frontmatter_type: "assessment"
    # No aliases (research deprecated)

prohibited_types:
  - name: "research"
    use_instead: "assessment"
    reason: "Research is a form of assessment; use assessment type"

  - name: "design"
    use_instead: "design_document"
    reason: "Naming conflict resolved: use design_document as canonical name"

  - name: "template"
    use_instead: "docs/_templates/"
    reason: "Templates belong in docs/_templates/ not artifacts/"
```

### Code Changes

**PluginValidator Enhanced:**
- Now loads `.agentqms/schemas/artifact_type_validation.yaml`
- Validates plugin names against canonical types
- Rejects prohibited type names with clear error messages
- Suggests correct alternative when validation fails

**Example Validation:**
```python
# Before (allowed both design and design_document)
validator.validate(plugin_data, "artifact_type")  # No errors

# After (enforces canonical names)
plugin_data = {"name": "design", ...}
errors = validator.validate(plugin_data, "artifact_type")
# errors = ["Plugin name 'design' is an alias. Use canonical name 'design_document' instead."]

plugin_data = {"name": "research", ...}
errors = validator.validate(plugin_data, "artifact_type")
# errors = ["Prohibited artifact type 'research'. Use 'assessment' instead. Reason: ..."]
```

## Migration Guide for Users

### For Existing Artifacts

**No action required** - Existing artifacts remain valid:
- Artifacts with `type: design` frontmatter continue to work
- Artifacts with `type: research` frontmatter continue to work
- File naming conventions unchanged

### For New Artifacts

**Use canonical names:**

```bash
# ❌ Old (deprecated)
cd AgentQMS/bin && make create-design NAME=my-feature
cd AgentQMS/bin && make create-research NAME=investigation

# ✅ New (canonical)
cd AgentQMS/bin && make create-design NAME=my-feature  # Still works, creates design_document
cd AgentQMS/bin && make create-assessment NAME=investigation
```

### For MCP Integration

**MCP Schema Updates:**
- `artifact_type` enum now includes canonical names only
- `design_document`, `assessment` available in dropdown
- `design`, `research`, `template` removed from options

### For Plugin Developers

**Creating Custom Artifact Types:**
1. Choose a canonical name (not in prohibited list)
2. Follow naming convention: `lowercase_with_underscores`
3. Use `.agentqms/schemas/artifact_type_validation.yaml` as reference
4. Validation runs automatically on plugin load

## Testing

### Validation Tests

```python
# Test prohibited types are rejected
def test_prohibited_type_rejected():
    validator = PluginValidator(validation_rules_path=rules_path)
    plugin = {"name": "research", "metadata": {...}}
    errors = validator.validate(plugin, "artifact_type")
    assert "Prohibited artifact type 'research'" in errors[0]
    assert "Use 'assessment' instead" in errors[0]

# Test aliases trigger warnings
def test_alias_triggers_warning():
    validator = PluginValidator(validation_rules_path=rules_path)
    plugin = {"name": "design", "metadata": {...}}
    errors = validator.validate(plugin, "artifact_type")
    assert "Plugin name 'design' is an alias" in errors[0]
    assert "Use canonical name 'design_document'" in errors[0]

# Test canonical types accepted
def test_canonical_type_accepted():
    validator = PluginValidator(validation_rules_path=rules_path)
    plugin = {"name": "design_document", "metadata": {...}, "template": "..."}
    errors = validator.validate(plugin, "artifact_type")
    assert errors == []
```

## Benefits

1. **Single Source of Truth:** One canonical name per artifact type
2. **Clear Validation:** Immediate feedback on invalid type names
3. **Reduced Confusion:** No duplicate/conflicting type definitions
4. **Better Documentation:** Clear mapping of types to purposes
5. **Extensibility:** New types validated against centralized rules

## Rollout Plan

1. ✅ **Phase 3 Session 5:** Resolve conflicts, deprecate plugins
2. **Phase 4 Session 6:** Remove hardcoded templates from artifact_templates.py
3. **Phase 5 Session 7:** Make MCP schema fully dynamic (no hardcoded enums)
4. **Phase 6 Session 8:** Update developer documentation

## References

- **Validation Schema:** `.agentqms/schemas/artifact_type_validation.yaml`
- **Standards Index:** `AgentQMS/standards/tier1-sst/artifact-types.yaml`
- **Plugin Validator:** `AgentQMS/tools/core/plugins/validation.py`
- **Active Plugins:** `AgentQMS/.agentqms/plugins/artifact_types/`
- **Roadmap:** `project_compass/roadmap/00_agentqms_artifact_consolidation.yaml`
