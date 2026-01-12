---
type: "walkthrough"
category: "development"
status: "active"
version: "1.0"
date: "2026-01-12 12:15 (KST)"
title: "How to Analyze and Externalize Configurations: Step-by-Step Guide"
tags: ["walkthrough", "refactoring", "configuration", "tutorial", "how-to"]
---

# How to Analyze and Externalize Configurations: Step-by-Step Guide

## Objective

Learn how to identify embedded configurations in Python modules and externalize them following the standardized pattern established for `artifact_templates.py`.

**Time to Complete**: 30-45 minutes per module
**Difficulty**: Intermediate

---

## Step 1: Analyze the Target Module

### 1a. Search for Common Configuration Patterns

Open the target file and search for these patterns:

**Pattern 1: Hardcoded Lists**
```python
# Search for: \[.*\]
my_list = ["value1", "value2", "value3"]
denylist = {"item1", "item2"}
tags = [some_var]  # Even with variables!
```

**Pattern 2: Hardcoded Dictionaries**
```python
# Search for: {.*:.*}
defaults = {
    "key": "value",
    "key2": "value2",
}
```

**Pattern 3: Magic Numbers/Strings**
```python
# Search for: hardcoded numbers/quoted strings
timeout = 300
default_branch = "main"
pattern = "YYYY-MM-DD_HHMM"
```

**Pattern 4: Status/Constant Enums**
```python
status = "active"
category = "development"
version = "1.0"
```

### 1b. Create Configuration Inventory

For each configuration found, record:

```
Configuration: {Name}
├─ Location: {File.py:line_number} in {method_name}()
├─ Type: {list|dict|string|number}
├─ Current Value: {example}
├─ Frequency: {Appears once|Repeated N times}
├─ Severity: {HIGH|MEDIUM|LOW}
├─ Reason for Externalization: {Benefit}
└─ Proposed Config Key: {section.subsection.key}
```

### Example from artifact_templates.py

```
Configuration: Frontmatter Default Values
├─ Location: artifact_templates.py:162 in _convert_plugin_to_template()
├─ Type: dict
├─ Current Value: {"type": name, "category": "development", "status": "active", "version": "1.0", "tags": [name]}
├─ Frequency: Repeated once (but duplicated in defaults logic)
├─ Severity: HIGH (static values that users might want to change)
├─ Reason for Externalization: Allow ops team to change defaults without code modification
└─ Proposed Config Key: frontmatter_defaults.{type|category|status|version|tags}
```

---

## Step 2: Search Using Grep

Use regex search to find all instances systematically:

### Search 1: Hardcoded Lists
```bash
grep -n "\[.*\]" AgentQMS/tools/core/your_module.py | grep -v "^#"
```

### Search 2: Magic Numbers (3+ digit numbers)
```bash
grep -nE '\b[0-9]{3,}\b' AgentQMS/tools/core/your_module.py
```

### Search 3: Quoted Constant Strings
```bash
grep -nE '"[a-z_]+":|"[a-z_]+"' AgentQMS/tools/core/your_module.py | head -20
```

### Search 4: Configuration-like Patterns
```bash
grep -nE 'defaults|settings|config|const|CONST' AgentQMS/tools/core/your_module.py
```

---

## Step 3: Categorize Configurations

Group configurations by logical section in a table:

| Category | Configuration | Type | Lines | Priority |
|----------|---|---|---|---|
| **Authentication** | username_pattern | string | 145 | P1 |
| | password_min_length | number | 146 | P1 |
| **Validation** | required_fields | list | 89 | P2 |
| | max_name_length | number | 92 | P2 |
| **Formatting** | date_format | string | 234 | P3 |
| | delimiter | string | 250 | P3 |

---

## Step 4: Determine What to Externalize

**Externalize** (HIGH priority):
- ✅ Values that change between environments
- ✅ Values that users might want to customize
- ✅ Values repeated in multiple places
- ✅ Static lookup tables
- ✅ Threshold values
- ✅ Format strings

**Don't Externalize** (Skip these):
- ❌ Computed values
- ❌ Loop variables
- ❌ Temporary data structures
- ❌ Object instances
- ❌ Function/method calls
- ❌ Single-use magic constants

### Example Decision Matrix

| Item | Externalize? | Reason |
|------|---|---|
| `denylist = ["a", "b", "c"]` | ✅ YES | Repeated in config, users customize |
| `timeout = 300` | ✅ YES | Environment-specific, might change |
| `branch = "main"` | ✅ YES | Might differ per org/team |
| `now = datetime.now()` | ❌ NO | Computed at runtime, not config |
| `for key in kwargs:` | ❌ NO | Loop variable, temporary |
| `x = y + 7` | ❌ NO | Computation, not configuration |
| `date_fmt = "%Y-%m-%d"` | ✅ YES | Used multiple times, format string |

---

## Step 5: Design Configuration File Structure

### 5a. Choose File Location

**Standard location**:
```
AgentQMS/config/{module_name}_config.yaml
```

**Examples**:
- `AgentQMS/tools/core/plugins.py` → `AGentQMS/config/plugins_config.yaml`
- `AgentQMS/tools/core/validators.py` → `AgentQMS/config/validators_config.yaml`
- `AgentQMS/tools/core/tool_registry.py` → `AgentQMS/config/tool_registry_config.yaml`

### 5b. Design Section Hierarchy

Group configurations logically:

```yaml
# ============================================================================
# SECTION 1: Logically Related Group
# ============================================================================

key1: value1
key2: value2

nested_group:
  key3: value3
  key4: value4

# ============================================================================
# SECTION 2: Different Context
# ============================================================================

key5: value5
```

### 5c. Write Configuration File

**Template to follow**:

```yaml
# {Module Name} Configuration
# Controls behavior of {module_name}.py without code changes

# ============================================================================
# SECTION 1: {Brief Description}
# ============================================================================

# Setting explanation (1-2 lines)
setting_key: value

# Setting with options
status_choices:
  - "active"
  - "inactive"
  - "pending"

# Nested settings
parent_key:
  child1: value1
  child2: value2

# ============================================================================
# NOTES FOR CONFIGURATION MAINTAINERS
# ============================================================================
#
# When modifying this file:
# 1. Maintain valid YAML syntax
# 2. Don't delete required keys
# 3. Changes take effect on next operation
#
```

---

## Step 6: Implement in Python Module

### 6a. Add Configuration Methods

Add to your class `__init__`:

```python
from AgentQMS.tools.utils.config_loader import ConfigLoader
from pathlib import Path
from typing import Any

class YourModule:
    def __init__(self):
        # ... existing init code ...
        self._config_loader = ConfigLoader(cache_size=5)
        self._config_cache: dict[str, Any] | None = None
```

Add new methods to class:

```python
def _get_config(self) -> dict[str, Any]:
    """Get cached configuration, loading if necessary."""
    if self._config_cache is None:
        self._config_cache = self._load_config()
    return self._config_cache

def _load_config(self) -> dict[str, Any]:
    """Load configuration from YAML file.

    Uses ConfigLoader for caching. Falls back to safe defaults
    if config file not found.
    """
    config_path = (
        Path(__file__).resolve().parent.parent
        / "config" / "your_module_config.yaml"
    )

    # Define all expected keys with safe defaults
    defaults = {
        "setting1": "default_value",
        "setting2": ["default", "values"],
        "nested": {
            "key": "value",
        },
    }

    return self._config_loader.get_config(config_path, defaults=defaults)
```

### 6b. Replace Hardcoded Values

**Pattern 1: Single value replacement**

**Before**:
```python
def some_method(self):
    denylist = ["a", "b", "c"]
    # ... use denylist ...
```

**After**:
```python
def some_method(self):
    config = self._get_config()
    denylist = config.get("denylist", [])
    # ... use denylist ...
```

**Pattern 2: Multiple replacements in same method**

**Before**:
```python
def create_item(self):
    status = "active"
    version = "1.0"
    category = "development"
    # ... use these ...
```

**After**:
```python
def create_item(self):
    config = self._get_config()
    status = config.get("default_status", "active")
    version = config.get("default_version", "1.0")
    category = config.get("default_category", "development")
    # ... use these ...
```

**Pattern 3: Nested configuration access**

**Before**:
```python
if type_name == "bug_report":
    default_id = "001"
```

**After**:
```python
config = self._get_config()
bug_report_config = config.get("artifact_types", {}).get("bug_report", {})
default_id = bug_report_config.get("default_id", "001")
```

### 6c. Cache Configuration Wisely

**For frequently called methods**:
```python
def frequently_called(self):
    # Cache the config lookup to avoid dictionary traversal
    config = self._get_config()
    important_setting = config.get("important_key")
    # ... rest of method
```

**For infrequently called methods**:
```python
def rarely_called(self):
    # Direct access is fine
    config = self._get_config()
    value = config.get("rarely_used_key")
    # ... rest of method
```

---

## Step 7: Testing

### 7a. Test Without Config File

**Verify fallback defaults work**:
```bash
# Move config file temporarily
mv AGentQMS/config/your_module_config.yaml \
   AGentQMS/config/your_module_config.yaml.bak

# Run tests
pytest tests/test_your_module.py -v

# Restore config
mv AGentQMS/config/your_module_config.yaml.bak \
   AGentQMS/config/your_module_config.yaml
```

**Expected result**: All tests pass using fallback defaults.

### 7b. Test With Modified Config

**Verify config values are loaded**:
```yaml
# Edit AGentQMS/config/your_module_config.yaml
# Change one value to something obviously different

test_value: "TEST_VALUE_12345"
```

**Test that the value is loaded**:
```python
def test_config_loading():
    module = YourModule()
    config = module._get_config()
    assert config["test_value"] == "TEST_VALUE_12345"
```

**Restore original config**.

### 7c. Test Functionality Unchanged

```bash
# Run full test suite
pytest tests/test_your_module.py -v

# All existing tests should pass without modification
# No functional changes, only refactoring
```

---

## Step 8: Document Changes

### 8a. Update Class Docstring

```python
class YourModule:
    """Description of module.

    Configuration Keys (from your_module_config.yaml):
        - setting1 (str): Description of what this controls
        - setting2 (list[str]): Description of what this controls
        - nested_config (dict):
            - nested_key1 (str): Description
            - nested_key2 (int): Description

    Examples:
        >>> module = YourModule()
        >>> module.some_method()  # Uses config values
    """
```

### 8b. Update README

If there's a configuration guide, add:

```markdown
## Configuration

Edit `AGentQMS/config/your_module_config.yaml` to customize behavior without code changes.

### Available Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `setting1` | str | "default" | What this controls |
| `setting2` | list | ["a", "b"] | What this controls |
```

---

## Complete Example: Converting a Module

### Original Module (tool_registry.py excerpt)

```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, tool):
        # Hardcoded restrictions
        if len(name) > 50:
            raise ValueError("Name too long")

        required_attrs = ["name", "execute", "description"]
        for attr in required_attrs:
            if not hasattr(tool, attr):
                raise ValueError(f"Missing {attr}")

        # Hardcoded defaults
        metadata = {
            "status": "active",
            "version": "1.0",
            "category": "general",
        }

        self.tools[name] = {**tool.__dict__, **metadata}
```

### Refactored Module

```python
from AgentQMS.tools.utils.config_loader import ConfigLoader
from pathlib import Path
from typing import Any

class ToolRegistry:
    """Registry for tools.

    Configuration Keys (from tool_registry_config.yaml):
        - max_tool_name_length (int): Maximum tool name length
        - required_tool_attributes (list[str]): Required tool attributes
        - tool_metadata_defaults (dict): Default metadata for tools
    """

    def __init__(self):
        self.tools = {}
        self._config_loader = ConfigLoader(cache_size=5)
        self._config_cache: dict[str, Any] | None = None

    def _get_config(self) -> dict[str, Any]:
        """Get cached configuration."""
        if self._config_cache is None:
            self._config_cache = self._load_config()
        return self._config_cache

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from tool_registry_config.yaml."""
        config_path = (
            Path(__file__).resolve().parent.parent
            / "config" / "tool_registry_config.yaml"
        )

        defaults = {
            "max_tool_name_length": 50,
            "required_tool_attributes": ["name", "execute", "description"],
            "tool_metadata_defaults": {
                "status": "active",
                "version": "1.0",
                "category": "general",
            },
        }

        return self._config_loader.get_config(config_path, defaults=defaults)

    def register_tool(self, name, tool):
        config = self._get_config()

        max_len = config.get("max_tool_name_length", 50)
        if len(name) > max_len:
            raise ValueError(f"Name too long (max {max_len})")

        required_attrs = config.get("required_tool_attributes", [])
        for attr in required_attrs:
            if not hasattr(tool, attr):
                raise ValueError(f"Missing {attr}")

        metadata = config.get("tool_metadata_defaults", {}).copy()
        self.tools[name] = {**tool.__dict__, **metadata}
```

### Configuration File (tool_registry_config.yaml)

```yaml
# Tool Registry Configuration

# ============================================================================
# TOOL VALIDATION
# ============================================================================

# Maximum length for tool names (prevents database issues)
max_tool_name_length: 50

# Required attributes every tool must have
required_tool_attributes:
  - name
  - execute
  - description

# ============================================================================
# TOOL METADATA DEFAULTS
# ============================================================================

# Default metadata applied to all registered tools
tool_metadata_defaults:
  status: "active"
  version: "1.0"
  category: "general"
```

---

## Common Pitfalls to Avoid

### ❌ Pitfall 1: Externalizing Runtime Computed Values

```python
# ❌ WRONG: This is computed, not configured
config = {
    "today": datetime.now().strftime("%Y-%m-%d"),  # Runtime value!
}

# ✅ RIGHT: Store format, compute at runtime
config = {
    "date_format": "%Y-%m-%d",
}
# Then in code:
today = datetime.now().strftime(config["date_format"])
```

### ❌ Pitfall 2: Hardcoding Defaults in Multiple Places

```python
# ❌ WRONG: Default duplicated
defaults = {
    "max_length": 100,
}
config.get("max_length", 100)  # Hardcoded again!

# ✅ RIGHT: Single source of truth
defaults = {"max_length": 100}
config.get("max_length", defaults["max_length"])
```

### ❌ Pitfall 3: Not Caching Configuration

```python
# ❌ SLOW: Loads YAML file on every call
def some_method(self):
    config = self._load_config()  # Reloads file!
    # ...

# ✅ FAST: Caches after first load
def some_method(self):
    config = self._get_config()  # Returns cached copy
    # ...
```

### ❌ Pitfall 4: Complex Placeholder Logic

```python
# ❌ COMPLEX: String templating in Python
frontmatter = {
    "type": config["type"].replace("{artifact_type}", name)
}

# ✅ SIMPLE: Let config have the value
# Config: frontmatter_defaults: {type: "{artifact_type}"}
# Code:
frontmatter = config["frontmatter_defaults"].copy()
for k, v in frontmatter.items():
    if isinstance(v, str):
        frontmatter[k] = v.replace("{artifact_type}", name)
```

---

## Verification Checklist

Before committing your changes:

- ✅ Configuration file created in correct location
- ✅ Configuration file has proper YAML syntax
- ✅ All hardcoded values identified and moved to config
- ✅ `_get_config()` method implemented
- ✅ `_load_config()` method implemented with defaults
- ✅ Configuration is cached (lazy loading)
- ✅ All hardcoded value replacements working
- ✅ Class docstring updated with config keys
- ✅ All existing tests pass
- ✅ Module functions identically to before refactoring
- ✅ Configuration file documented with comments
- ✅ Fallback defaults handle missing config gracefully

---

## Next Modules to Refactor

Once complete with one module, proceed in this order:

1. **artifact_templates.py** - ✅ DONE
2. **tool_registry.py** - Similar patterns, good candidate
3. **plugins.py** - Plugin loading, configuration driven
4. **validators.py** - Validation rules and patterns
5. **artifact_creator.py** - Creation settings and templates
6. **Others in tools/core/** - As time permits

Each module follows the same pattern, so second and subsequent modules will be faster.

---

## Quick Reference

### File Locations

```
Source module: AGentQMS/tools/core/{module_name}.py
Config file:   AGentQMS/config/{module_name}_config.yaml
Tests:         AGentQMS/tests/test_{module_name}.py
```

### Code Templates

**Loading config**:
```python
config = self._get_config()
value = config.get("key", "fallback")
```

**Accessing nested config**:
```python
config = self._get_config()
nested = config.get("section", {}).get("subsection", {})
value = nested.get("key", "fallback")
```

**In defaults dict**:
```python
defaults = {
    "section": {
        "key": "value",
    },
}
```

**In YAML**:
```yaml
section:
  key: value
```

---

## Conclusion

By following this walkthrough, you can systematically externalize configurations from any Python module while:

- Maintaining backward compatibility
- Improving code clarity
- Enabling non-developers to customize behavior
- Following established patterns for consistency

**Estimated time per module**: 30-45 minutes (after first module, subsequent modules faster)
