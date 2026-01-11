# Template Defaults Externalization

**Date**: 2026-01-11
**Component**: AgentQMS artifact templates system
**Change Type**: Refactoring - configuration externalization

## Summary

Moved hardcoded template default values from `artifact_templates.py` to external YAML configuration file to improve maintainability and reduce script complexity.

## Changes

### Files Modified
- **AgentQMS/tools/core/artifact_templates.py**
  - Removed ~40 lines of hardcoded default dictionaries
  - Added `_load_template_defaults()` method to load from external config
  - Added caching to avoid repeated file reads
  - Updated `create_content()` and `create_frontmatter()` to use config

### Files Created
- **AgentQMS/config/template_defaults.yaml**
  - General defaults for all artifact types
  - Bug report specific defaults
  - Frontmatter denylist configuration
  - Versioned and documented with comments

## Benefits

1. **Readability**: Reduced artifact_templates.py from ~614 to ~570 lines
2. **Maintainability**: Template defaults can be updated without touching Python code
3. **Extensibility**: Easy to add new artifact-type-specific defaults
4. **Transparency**: Configuration values are now clearly visible in dedicated file
5. **AI-friendly**: AI agents can understand and modify config without parsing complex Python

## Configuration Structure

```yaml
version: "1.0"
defaults:           # General defaults for all types
  title: "..."
  description: "..."

bug_report:         # Type-specific defaults
  bug_id: "001"
  summary: "..."

frontmatter_denylist:  # Fields excluded from frontmatter
  - expected_behavior
  - actual_behavior
```

## Backward Compatibility

- ✅ Fully backward compatible
- ✅ Falls back to minimal defaults if config file missing
- ✅ Falls back if PyYAML not available
- ✅ All existing artifact creation workflows continue working

## Testing

- ✅ Template initialization
- ✅ Config loading and caching
- ✅ Content creation with defaults
- ✅ Frontmatter denylist enforcement
- ✅ End-to-end artifact creation

## Future Enhancements

Consider externalizing additional configuration:
- Artifact type validation rules (currently in plugins)
- Directory naming conventions
- Filename pattern templates
- Error message templates
