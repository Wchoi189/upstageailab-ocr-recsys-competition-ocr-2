---
type: "design_document"
category: "development"
status: "active"
version: "1.0"
ads_version: "1.0"
date: "2026-01-12 12:10 (KST)"
title: "Configuration Externalization Pattern: tools/core/ Standardization"
tags: ["design-document", "refactoring", "configuration", "pattern", "architecture"]
---

# Configuration Externalization Pattern: tools/core/ Standardization

## Executive Summary

This document defines a standard pattern for externalizing embedded configurations from all Python modules in `AgentQMS/tools/core/`. The pattern ensures consistency, maintainability, and configurability across the entire core tooling layer.

**Scope**: `AgentQMS/tools/core/*.py` (currently ~10 modules)
**Pattern**: Load-from-YAML with caching and fallback defaults
**Goal**: Zero embedded configuration lists/dicts in Python code

---

## Pattern: Configuration Externalization

### 1. Identification Phase

**What gets externalized**:
- ✅ Hardcoded lists (e.g., `["a", "b", "c"]`)
- ✅ Hardcoded dictionaries with static values
- ✅ Magic numbers and strings (e.g., `300`, `"main"`, `"---"`)
- ✅ Regex patterns (if more than one instance)
- ✅ Default values and fallbacks
- ✅ Mapping/translation tables

**What does NOT get externalized**:
- ❌ Loop variables and temporary data structures
- ❌ Dynamic data computed at runtime
- ❌ Singleton instances
- ❌ Function/method definitions

**Example - artifact_templates.py**:
```python
# ✅ EXTERNALIZE: Hardcoded default frontmatter
{
    "type": name,
    "category": "development",  # <- Hardcoded
    "status": "active",         # <- Hardcoded
    "version": "1.0",           # <- Hardcoded
}

# ✅ EXTERNALIZE: Magic number
if time_diff < 300:  # <- Hardcoded window

# ✅ EXTERNALIZE: Hardcoded denylist
system_args = {"output_dir", "interactive"}  # <- Hardcoded

# ❌ DON'T EXTERNALIZE: Computed at runtime
now = datetime.now()

# ❌ DON'T EXTERNALIZE: Loop variable
for key, value in kwargs.items():
```

---

### 2. Configuration File Structure

**Location**: `AgentQMS/config/{module_name}_config.yaml`

**Naming Convention**:
- `AgentQMS/tools/core/artifact_templates.py` → `AGentQMS/config/artifact_template_config.yaml`
- `AgentQMS/tools/core/plugins.py` → `AgentQMS/config/plugins_config.yaml`
- `AgentQMS/tools/core/tool_registry.py` → `AgentQMS/config/tool_registry_config.yaml`

**Structure**:
```yaml
# Use section headers to organize logically
# Section 1: Related configs
setting_1: value
setting_2: value

# Section 2: Different context
other_setting_1: value

# Subsections for complex structures
parent_setting:
  child_setting_1: value
  child_setting_2: value
```

**Comments**:
- Add header comments explaining what each section does
- Add inline comments for non-obvious values
- Include examples where helpful
- Document placeholders (e.g., `{artifact_type}`)

---

### 3. Python Module Changes

**Step 1: Update imports (add ConfigLoader if not present)**
```python
from AgentQMS.tools.utils.config_loader import ConfigLoader

# In __init__:
self._config_loader = ConfigLoader(cache_size=5)
self._config_cache: dict[str, Any] | None = None
```

**Step 2: Create config accessor method**
```python
def _get_config(self) -> dict[str, Any]:
    """Get cached configuration, loading if necessary.

    Configuration is loaded once and cached for the lifetime of the object.
    Falls back to safe defaults if config file not found.
    """
    if self._config_cache is None:
        self._config_cache = self._load_config()
    return self._config_cache

def _load_config(self) -> dict[str, Any]:
    """Load configuration from artifact_template_config.yaml.

    Uses ConfigLoader for consistent configuration management.
    Falls back to comprehensive defaults if config file not found.
    """
    config_path = Path(__file__).resolve().parent.parent / "config" / "{module_name}_config.yaml"

    # Define safe defaults matching the expected config structure
    defaults = {
        # ... all expected keys with fallback values
    }

    return self._config_loader.get_config(config_path, defaults=defaults)
```

**Step 3: Replace hardcoded values**

Replace all instances of hardcoded values with config lookups:

**Before**:
```python
denylist = {"output_dir", "interactive", "steps_to_reproduce"}
```

**After**:
```python
config = self._get_config()
denylist_items = config.get("denylist_keys", [])
denylist = set(denylist_items)
```

**Step 4: Document configuration keys**

Add docstring to class explaining config keys:

```python
class MyClass:
    """Description of class.

    Configuration Keys (from {module_name}_config.yaml):
        - setting_1 (str): Description
        - setting_2 (list[str]): Description
        - parent_setting (dict):
            - child_1 (str): Description
    """
```

---

### 4. Configuration File Template

```yaml
# {Module Name} Configuration
# Controls behavior of {module_name}.py without code changes
#
# Edit this file to modify default behavior. Changes are loaded automatically.

# ============================================================================
# SECTION 1: Brief description
# ============================================================================

# Individual setting with explanation
setting_key: value

# Grouped settings
parent_key:
  child_1: value
  child_2: value
  child_3:
    - item1
    - item2

# ============================================================================
# NOTES FOR CONFIGURATION MAINTAINERS
# ============================================================================
#
# When modifying this file:
# 1. Maintain YAML syntax (proper indentation, quote strings with special chars)
# 2. All changes take effect on next module operation
# 3. Invalid YAML will trigger a warning; defaults will be used
# 4. Never delete required top-level keys
#
```

---

## Module Inventory: tools/core/

### Modules Requiring Externalization

| Module | Status | Embedded Configs | Priority |
|--------|--------|------------------|----------|
| `artifact_templates.py` | **DONE** | 8 major configs | ✅ Complete |
| `tool_registry.py` | TODO | TBD | P1 |
| `plugins.py` | TODO | TBD | P2 |
| `validators.py` | TODO | TBD | P2 |
| `artifact_creator.py` | TODO | TBD | P3 |
| `import_manager.py` | TODO | TBD | P3 |
| Others (if present) | TODO | TBD | P4 |

### Analysis Template for Each Module

When analyzing each module for externalization:

```markdown
## {Module Name} - Configuration Analysis

### Embedded Configurations Identified

1. **Name**: {config_name}
   - Location: Line N in {method_name}()
   - Type: {list|dict|string}
   - Current value: {example}
   - Frequency: {how many times is it hardcoded?}
   - Severity: {HIGH|MEDIUM|LOW}
   - Config key: {proposed_config_key}

### Configuration File Structure

```yaml
{section_1}:
  {key}: {example_value}
```

### Python Changes Required

- Method 1: Change in {method_name}() (X lines)
- Method 2: Change in {method_name}() (X lines)

### Benefits

- {benefit_1}
- {benefit_2}
```

---

## Best Practices

### ✅ DO

1. **Provide comprehensive defaults**
   ```python
   defaults = {
       "required_key": "safe_fallback_value",
       "optional_key": None,
   }
   ```

2. **Cache configuration**
   ```python
   def _get_config(self) -> dict:
       if self._config_cache is None:
           self._config_cache = self._load_config()
       return self._config_cache
   ```

3. **Document configuration keys**
   - What each setting controls
   - Valid values
   - Impact of changes
   - Examples

4. **Use logical section headers**
   ```yaml
   # ============================================================================
   # SECTION NAME
   # ============================================================================
   ```

5. **Provide fallback defaults**
   - Comprehensive enough that config file is optional
   - All paths (with and without config) should work

### ❌ DON'T

1. **Don't externalize computed values**
   ```python
   # ❌ WRONG: This is computed, not configured
   target_date: "{today_plus_7d}"

   # ✅ RIGHT: This is configured
   target_date_offset_days: 7  # Then compute: now + timedelta(days=7)
   ```

2. **Don't repeat defaults in docstrings and code**
   ```python
   # ❌ WRONG: Duplication
   # "Default is 'main'" <- documented here
   config.get("default_branch", "main")  # ... and here

   # ✅ RIGHT: Single source of truth
   defaults = {"default_branch": "main"}
   config.get("default_branch", defaults["default_branch"])
   ```

3. **Don't hardcode file paths in config loading**
   ```python
   # ❌ WRONG: Hardcoded paths make testing hard
   config_path = "/workspace/AgentQMS/config/artifact_template_config.yaml"

   # ✅ RIGHT: Relative to module location
   config_path = Path(__file__).resolve().parent.parent / "config" / "artifact_template_config.yaml"
   ```

4. **Don't make loading a blocking operation**
   ```python
   # ❌ WRONG: Loads on every method call
   def some_method(self):
       config = self._load_config()  # Reloads YAML file every time!

   # ✅ RIGHT: Cache with lazy loading
   def some_method(self):
       config = self._get_config()  # Returns cached copy
   ```

---

## Standardized Configuration Loading Pattern

### Minimum Implementation

```python
from pathlib import Path
from typing import Any
from AgentQMS.tools.utils.config_loader import ConfigLoader

class MyModule:
    """Description."""

    def __init__(self):
        self._config_loader = ConfigLoader(cache_size=5)
        self._config_cache: dict[str, Any] | None = None

    def _get_config(self) -> dict[str, Any]:
        """Get cached configuration."""
        if self._config_cache is None:
            self._config_cache = self._load_config()
        return self._config_cache

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML."""
        config_path = Path(__file__).resolve().parent.parent / "config" / "mymodule_config.yaml"
        defaults = {
            "key1": "value1",
            "key2": ["item1", "item2"],
        }
        return self._config_loader.get_config(config_path, defaults=defaults)

    def some_method(self):
        config = self._get_config()
        value = config.get("key1")
        # ...
```

---

## Workflow for New Modules

### Phase: Externalization

1. **Analyze** the module for embedded configurations
   - Run grep_search for common patterns: `[|{|"|set(`
   - Identify all hardcoded values
   - Group by logical section

2. **Create** configuration file
   - Follow naming convention: `{module_name}_config.yaml`
   - Organize by section
   - Include comprehensive comments
   - Test YAML syntax is valid

3. **Implement** configuration loading
   - Add `_get_config()` and `_load_config()` methods
   - Create defaults dict matching config structure
   - Add caching to avoid repeated YAML loads

4. **Replace** all hardcoded values
   - For each configuration item:
     - Remove hardcoded value
     - Add config lookup: `config.get("key", default)`
     - Test functionality is unchanged

5. **Test** thoroughly
   - Without config file (fallback defaults work)
   - With config file (values load correctly)
   - Config changes take effect immediately
   - All module functions work as before

6. **Document** changes
   - Update class docstring
   - List all configuration keys
   - Provide examples

---

## Checklist for Complete Externalization

For each module, verify:

- ✅ All hardcoded lists in config file
- ✅ All hardcoded dicts in config file
- ✅ All magic numbers/strings in config file
- ✅ All repeated patterns in config file
- ✅ Configuration file has section headers
- ✅ Configuration file has inline comments
- ✅ Python module has `_get_config()` method
- ✅ Python module has `_load_config()` method
- ✅ Config is cached (lazy loading)
- ✅ Comprehensive defaults provided
- ✅ Module docstring documents config keys
- ✅ All tests pass
- ✅ Functionality unchanged (backward compatible)

---

## Benefits of Standardization

1. **Consistency**
   - All `tools/core/` modules follow same pattern
   - New developers know where to find configurations
   - Easy to move between modules

2. **Maintainability**
   - Non-developers can modify behavior without code knowledge
   - No need to recompile or restart (mostly)
   - Changes tracked in version control

3. **Testability**
   - Configurations can be mocked in unit tests
   - Test fixtures can override defaults
   - Easier to test different scenarios

4. **Scalability**
   - Easy to add new configuration options
   - Easy to support environment-specific configs
   - Pattern extends to config merging (env overrides)

5. **Operational Excellence**
   - DevOps can configure behavior without developer involvement
   - Fast iteration on settings without code changes
   - Audit trail of config changes in git

---

## Example: Progression from Hardcoded to Externalized

### Before (artifact_templates.py - original)

```python
class ArtifactTemplates:
    def _convert_plugin_to_template(self, name, plugin_def):
        template = {
            "frontmatter": metadata.get(
                "frontmatter",
                {
                    "type": name,
                    "category": "development",      # ← Hardcoded
                    "status": "active",              # ← Hardcoded
                    "version": "1.0",                # ← Hardcoded
                    "tags": [name],
                },
            ),
        }
```

### After (artifact_templates.py - externalized)

```python
class ArtifactTemplates:
    def _convert_plugin_to_template(self, name, plugin_def):
        config = self._get_config()
        default_fm = config["frontmatter_defaults"].copy()

        # Replace placeholders
        default_fm = {
            k: v.replace("{artifact_type}", name) if isinstance(v, str) else v
            for k, v in default_fm.items()
        }

        template = {
            "frontmatter": metadata.get("frontmatter", default_fm),
        }
```

### Configuration file (artifact_template_config.yaml)

```yaml
frontmatter_defaults:
  type: "{artifact_type}"
  category: "development"
  status: "active"
  version: "1.0"
  tags:
    - "{artifact_type}"
```

**Result**: Code cleaner, configuration separated, easily modifiable without code changes.

---

## Conclusion

Configuration externalization following this pattern:

1. ✅ Removes embedded configs from code
2. ✅ Creates single source of truth (config files)
3. ✅ Enables non-developers to modify behavior
4. ✅ Improves testability and maintainability
5. ✅ Scales across all `tools/core/` modules
6. ✅ Follows existing patterns in the codebase

**Next Steps**: Apply this pattern to remaining modules in priority order.