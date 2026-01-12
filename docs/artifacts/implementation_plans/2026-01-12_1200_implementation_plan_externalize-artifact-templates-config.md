---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
date: "2026-01-12 12:00 (KST)"
title: "Configuration Externalization: artifact_templates.py"
tags: ["implementation-plan", "refactoring", "configuration", "maintenance"]
---

# Configuration Externalization: artifact_templates.py

## Overview

This plan externalizes all embedded lists, dictionaries, and configuration values from `artifact_templates.py` to make them easily configurable and maintainable without code changes.

**Target File**: `AgentQMS/tools/core/artifact_templates.py`
**New Config File**: `AgentQMS/config/artifact_template_config.yaml`
**Effort**: 2-3 hours
**Risk Level**: Low (all changes are internal, no API changes)

---

## Identified Embedded Configurations

### 1. Default Frontmatter Template (Lines 159-169)

**Location**: `_convert_plugin_to_template()` method

**Current**:
```python
"frontmatter": metadata.get(
    "frontmatter",
    {
        "type": name,
        "category": "development",
        "status": "active",
        "version": "1.0",
        "tags": [name],
    },
),
```

**Issues**:
- Hardcoded defaults if plugin doesn't define them
- Can't change defaults without code modification
- Category, status, version are duplicated elsewhere

**Solution**: Move to config file under `frontmatter_defaults`

---

### 2. Frontmatter Denylist (Line 394-395)

**Location**: `create_frontmatter()` method

**Current**:
```python
system_args = {"output_dir", "interactive", "steps_to_reproduce"}
denylist = denylist_from_config | system_args
```

**Issues**:
- Hardcoded system arguments list
- Hard to extend without code change
- Duplicated from config (also loads from YAML)

**Solution**: Move to config file under `frontmatter_denylist`

---

### 3. Default Branch Name (Lines 382-386)

**Location**: `create_frontmatter()` method

**Current**:
```python
if "branch" not in kwargs:
    if UTILITIES_AVAILABLE and _get_current_branch is not None:
        try:
            frontmatter["branch"] = _get_current_branch()
        except Exception:
            frontmatter["branch"] = "main"  # Hardcoded!
    else:
        frontmatter["branch"] = "main"  # Hardcoded!
```

**Issues**:
- "main" is hardcoded in two places
- Can't easily change default branch

**Solution**: Move to config as `default_branch`

---

### 4. YAML Frontmatter Delimiter (Line 402, 406)

**Location**: `create_frontmatter()` method

**Current**:
```python
lines = ["---"]
# ... frontmatter lines ...
lines.append("---")
```

**Issues**:
- Hardcoded YAML delimiter
- Can't easily switch to different format (TOML, JSON)
- Used in multiple places

**Solution**: Move to config as `frontmatter_delimiter`

---

### 5. Date Format Strings (Multiple locations)

**Location**: `create_filename()`, `create_frontmatter()`, `create_content()`

**Current**:
```python
# Line 282: timestamp filename format
timestamp = now.strftime("%Y-%m-%d_%H%M")

# Line 376: KST timestamp format
frontmatter["date"] = datetime.now(kst).strftime("%Y-%m-%d %H:%M (KST)")

# Line 429: content template defaults
"start_date": now.strftime("%Y-%m-%d"),
"target_date": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
"assessment_date": now.strftime("%Y-%m-%d"),
```

**Issues**:
- Multiple date formats scattered throughout
- Hard to maintain consistency
- Can't change format globally

**Solution**: Move to config under `date_formats`

---

### 6. Content Template Default Values (Lines 429-437)

**Location**: `create_content()` method

**Current**:
```python
defaults = {
    "title": title,
    "start_date": now.strftime("%Y-%m-%d"),
    "target_date": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
    "assessment_date": now.strftime("%Y-%m-%d"),
}
```

**Issues**:
- Default values hardcoded
- Durations (7 days for target_date) not configurable
- Can't adjust without code change

**Solution**: Move to config under `content_defaults`

---

### 7. Duplicate Detection Configuration (Lines 488, 494)

**Location**: `create_artifact()` method

**Current**:
```python
if template_type == "bug_report":
    pattern_base = f"BUG_{current_date}_*_*_{descriptive_name}.md"
else:
    pattern_base = f"{current_date}_*_{template_type}_{normalized_name}.md"

# ... later ...
if time_diff < 300:  # 5 minutes = 300 seconds
```

**Issues**:
- Glob patterns hardcoded for each artifact type
- Time window (300 seconds) hardcoded
- Can't customize per-type patterns

**Solution**: Move to config under `duplicate_detection`

---

### 8. Character and Name Normalization Rules (Multiple locations)

**Location**: `create_filename()` method

**Current**:
```python
normalized_name = name.lower().replace(" ", "-").replace("_", "-").replace("--", "-").strip("-")

# Bug report separators
for sep in ("-", "_"):
    dup = f"{hint}{sep}"
    if normalized_name.startswith(dup):
        normalized_name = normalized_name[len(dup):].lstrip("-_")
```

**Issues**:
- Normalization rules hardcoded
- Separator handling hardcoded
- Can't change naming conventions without code change

**Solution**: Move to config under `naming_conventions`

---

## Proposed Configuration File Structure

Create `AgentQMS/config/artifact_template_config.yaml`:

```yaml
# Artifact Template Configuration
# Externalized settings for artifact generation and formatting

# ============================================================================
# FRONTMATTER CONFIGURATION
# ============================================================================

# Default frontmatter fields when plugin doesn't define them
frontmatter_defaults:
  type: "{artifact_type}"  # Use {artifact_type} placeholder for the type
  category: "development"
  status: "active"
  version: "1.0"
  tags: ["{artifact_type}"]  # Use {artifact_type} placeholder

# Fields to exclude from frontmatter (merged with config denylists)
# These represent internal/system arguments that shouldn't be in frontmatter
frontmatter_denylist:
  - "output_dir"
  - "interactive"
  - "steps_to_reproduce"

# Default branch name when git branch detection fails
default_branch: "main"

# Frontmatter format delimiter
frontmatter_delimiter: "---"

# ============================================================================
# DATE/TIME CONFIGURATION
# ============================================================================

date_formats:
  # Format for timestamps in artifact filenames (YYYY-MM-DD_HHMM)
  filename_timestamp: "%Y-%m-%d_%H%M"

  # Format for date-only fields (YYYY-MM-DD)
  date_only: "%Y-%m-%d"

  # Format for full timestamp with timezone (YYYY-MM-DD HH:MM (KST))
  timestamp_with_tz: "%Y-%m-%d %H:%M (KST)"

# ============================================================================
# CONTENT TEMPLATE DEFAULTS
# ============================================================================

# Default values for content template rendering
content_defaults:
  # Duration (in days) for target date relative to start date
  target_date_offset_days: 7

  # Template placeholder defaults
  default_values:
    start_date: "{today}"  # Placeholder: will be replaced with actual date
    target_date: "{today_plus_7d}"  # Placeholder: 7 days from now
    assessment_date: "{today}"  # Placeholder: today's date

# ============================================================================
# NAMING AND NORMALIZATION RULES
# ============================================================================

naming_conventions:
  # Character replacements for name normalization
  replacements:
    - [" ", "-"]      # Replace spaces with hyphens
    - ["_", "-"]      # Replace underscores with hyphens
    - ["--", "-"]     # Collapse double hyphens to single

  # Characters to strip from start/end
  strip_chars: "-_"

  # Convert to lowercase
  lowercase: true

  # Separators for type deduplication (remove type prefix from name)
  type_prefix_separators: ["-", "_"]

# ============================================================================
# DUPLICATE DETECTION CONFIGURATION
# ============================================================================

duplicate_detection:
  # Time window (in seconds) to consider files as duplicates
  recent_file_window_seconds: 300
  recent_file_window_description: "5 minutes"

  # Glob patterns for finding recent duplicates by artifact type
  # {date} = current date in YYYY-MM-DD format
  # {name} = artifact name/slug
  # {id} = numeric ID (bug reports only)
  patterns:
    bug_report: "bug_{date}_*_*_{name}.md"
    default: "{date}_*_{type}_{name}.md"

# ============================================================================
# SPECIAL ARTIFACT TYPE CONFIGURATIONS
# ============================================================================

# Bug report specific settings
artifact_types:
  bug_report:
    # Default bug ID when none can be extracted from name
    default_id: "001"

    # Expected format for bug IDs in names
    # Example: "123_description" or "BUG_123_description"
    id_extraction_patterns:
      - "^(\\d+)_"        # Leading digits: "123_description"
      - "BUG_(\\d+)_"     # With BUG prefix: "BUG_123_description"

    # Separators for extracting bug ID and description
    separators: ["_"]
```

---

## Implementation Steps

### Step 1: Create Configuration File

Create `AgentQMS/config/artifact_template_config.yaml` with the structure above.

**Effort**: 30 minutes

---

### Step 2: Update `_load_template_defaults()` Method

Extend to load the new configuration file:

**Current** (lines 114-128):
```python
def _load_template_defaults(self) -> dict[str, Any]:
    """Load template defaults from external YAML configuration.

    Uses ConfigLoader for consistent configuration management.
    Falls back to minimal defaults if config file not found.
    """
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "template_defaults.yaml"

    defaults = {
        "defaults": {},
        "bug_report": {},
        "frontmatter_denylist": []
    }

    return self._config_loader.get_config(config_path, defaults=defaults)
```

**Change To**:
```python
def _load_template_defaults(self) -> dict[str, Any]:
    """Load template defaults from external YAML configuration.

    Loads artifact_template_config.yaml for all artifact generation settings.
    Uses ConfigLoader for consistent configuration management.
    Falls back to safe defaults if config file not found.
    """
    config_path = Path(__file__).resolve().parent.parent / "config" / "artifact_template_config.yaml"

    # Safe defaults that match the config file structure
    defaults = {
        "frontmatter_defaults": {
            "type": "{artifact_type}",
            "category": "development",
            "status": "active",
            "version": "1.0",
            "tags": ["{artifact_type}"],
        },
        "frontmatter_denylist": [
            "output_dir",
            "interactive",
            "steps_to_reproduce",
        ],
        "default_branch": "main",
        "frontmatter_delimiter": "---",
        "date_formats": {
            "filename_timestamp": "%Y-%m-%d_%H%M",
            "date_only": "%Y-%m-%d",
            "timestamp_with_tz": "%Y-%m-%d %H:%M (KST)",
        },
        "content_defaults": {
            "target_date_offset_days": 7,
            "default_values": {
                "start_date": "{today}",
                "target_date": "{today_plus_7d}",
                "assessment_date": "{today}",
            },
        },
        "naming_conventions": {
            "replacements": [
                [" ", "-"],
                ["_", "-"],
                ["--", "-"],
            ],
            "strip_chars": "-_",
            "lowercase": True,
            "type_prefix_separators": ["-", "_"],
        },
        "duplicate_detection": {
            "recent_file_window_seconds": 300,
            "recent_file_window_description": "5 minutes",
            "patterns": {
                "bug_report": "bug_{date}_*_*_{name}.md",
                "default": "{date}_*_{type}_{name}.md",
            },
        },
        "artifact_types": {
            "bug_report": {
                "default_id": "001",
                "id_extraction_patterns": [
                    "^(\\d+)_",
                    "BUG_(\\d+)_",
                ],
                "separators": ["_"],
            },
        },
    }

    return self._config_loader.get_config(config_path, defaults=defaults)
```

**Effort**: 30 minutes

---

### Step 3: Update `_convert_plugin_to_template()` Method

Use frontmatter defaults from config:

**Current** (lines 159-169):
```python
template: dict[str, Any] = {
    "filename_pattern": filename_pattern,
    "directory": directory,
    "frontmatter": metadata.get(
        "frontmatter",
        {
            "type": name,
            "category": "development",
            "status": "active",
            "version": "1.0",
            "tags": [name],
        },
    ),
    "content_template": template_content,
}
```

**Change To**:
```python
# Load default frontmatter from config
config = self._load_template_defaults()
default_frontmatter = config["frontmatter_defaults"].copy()

# Replace placeholders with actual artifact type name
default_frontmatter = {
    k: v.replace("{artifact_type}", name) if isinstance(v, str) else
       [vi.replace("{artifact_type}", name) if isinstance(vi, str) else vi for vi in v] if isinstance(v, list) else v
    for k, v in default_frontmatter.items()
}

template: dict[str, Any] = {
    "filename_pattern": filename_pattern,
    "directory": directory,
    "frontmatter": metadata.get("frontmatter", default_frontmatter),
    "content_template": template_content,
}
```

**Effort**: 45 minutes

---

### Step 4: Update `create_filename()` Method

Use naming conventions and bug report config:

**Replace hardcoded separators** (line ~335):

**Before**:
```python
for sep in ("-", "_"):
```

**After**:
```python
config = self._load_template_defaults()
separators = config.get("naming_conventions", {}).get("type_prefix_separators", ["-", "_"])
for sep in separators:
```

**Effort**: 30 minutes

---

### Step 5: Update `create_frontmatter()` Method

Use denylist and branch config from external config:

**Replace hardcoded denylist** (lines 394-395):

**Before**:
```python
config = self._load_template_defaults()
denylist_from_config = set(config.get("frontmatter_denylist", []))

# Always exclude system args
system_args = {"output_dir", "interactive", "steps_to_reproduce"}
denylist = denylist_from_config | system_args
```

**After**:
```python
config = self._load_template_defaults()
frontmatter_denylist = config.get("frontmatter_denylist", [])
denylist = set(frontmatter_denylist)
```

**Replace hardcoded branch and delimiters** (lines 382-406):

**Before**:
```python
if "branch" not in kwargs:
    if UTILITIES_AVAILABLE and _get_current_branch is not None:
        try:
            frontmatter["branch"] = _get_current_branch()
        except Exception:
            frontmatter["branch"] = "main"
    else:
        frontmatter["branch"] = "main"

# ... later ...
lines = ["---"]
# ... loop ...
lines.append("---")
```

**After**:
```python
if "branch" not in kwargs:
    if UTILITIES_AVAILABLE and _get_current_branch is not None:
        try:
            frontmatter["branch"] = _get_current_branch()
        except Exception:
            frontmatter["branch"] = config.get("default_branch", "main")
    else:
        frontmatter["branch"] = config.get("default_branch", "main")

# ... later ...
delimiter = config.get("frontmatter_delimiter", "---")
lines = [delimiter]
# ... loop ...
lines.append(delimiter)
```

**Effort**: 45 minutes

---

### Step 6: Update `create_content()` Method

Use date formats and content defaults from config:

**Replace hardcoded date formats** (lines 429-431):

**Before**:
```python
defaults = {
    "title": title,
    "start_date": now.strftime("%Y-%m-%d"),
    "target_date": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
    "assessment_date": now.strftime("%Y-%m-%d"),
}
```

**After**:
```python
config = self._load_template_defaults()
date_formats = config.get("date_formats", {})
content_defaults_config = config.get("content_defaults", {})

date_only_fmt = date_formats.get("date_only", "%Y-%m-%d")
offset_days = content_defaults_config.get("target_date_offset_days", 7)

defaults = {
    "title": title,
    "start_date": now.strftime(date_only_fmt),
    "target_date": (now + timedelta(days=offset_days)).strftime(date_only_fmt),
    "assessment_date": now.strftime(date_only_fmt),
}
```

**Effort**: 30 minutes

---

### Step 7: Update `create_artifact()` Method

Use duplicate detection config:

**Replace hardcoded patterns and window** (lines 481-495):

**Before**:
```python
if template_type == "bug_report":
    pattern_base = f"BUG_{current_date}_*_*_{descriptive_name}.md"
else:
    pattern_base = f"{current_date}_*_{template_type}_{normalized_name}.md"

# ... later ...
if time_diff < 300:  # 5 minutes = 300 seconds
```

**After**:
```python
config = self._load_template_defaults()
dup_config = config.get("duplicate_detection", {})
patterns = dup_config.get("patterns", {})

# Get pattern for this artifact type, fallback to default
pattern_template = patterns.get(
    template_type,
    patterns.get("default", "{date}_*_{type}_{name}.md")
)

# Substitute placeholders
pattern_base = pattern_template.format(
    date=current_date,
    name=normalized_name if template_type != "bug_report" else descriptive_name,
    type=template_type,
    id="*"
)

# ... later ...
window_seconds = dup_config.get("recent_file_window_seconds", 300)
if time_diff < window_seconds:
```

**Effort**: 45 minutes

---

## Configuration Caching Strategy

To avoid loading YAML on every artifact operation, cache the config:

**Add to `__init__` method**:
```python
self._config_cache: dict[str, Any] | None = None

def _get_config(self) -> dict[str, Any]:
    """Get cached configuration, loading if necessary."""
    if self._config_cache is None:
        self._config_cache = self._load_template_defaults()
    return self._config_cache
```

Then replace all `config = self._load_template_defaults()` with `config = self._get_config()`.

**Effort**: 15 minutes

---

## Summary of Changes

| Component | Lines Affected | External? | Config Key |
|-----------|----------------|-----------|-----------|
| Frontmatter defaults | 160-169 | YES | `frontmatter_defaults` |
| Frontmatter denylist | 394-395 | YES | `frontmatter_denylist` |
| Default branch | 382-386 | YES | `default_branch` |
| YAML delimiter | 402, 406 | YES | `frontmatter_delimiter` |
| Date formats | 282, 376, 429-431 | YES | `date_formats` |
| Content defaults | 429-437 | YES | `content_defaults` |
| Duplicate detection | 481-495, 488 | YES | `duplicate_detection` |
| Naming rules | ~282-298 | YES | `naming_conventions` |

**Total Lines Changed**: ~80-100
**Total Lines Added to Config**: ~100
**Net Result**: More maintainable, cleaner code

---

## Benefits

✅ **Easier Maintenance**: Change behavior without touching Python code
✅ **Better Configurability**: DevOps/operators can adjust without developers
✅ **Cleaner Code**: Removes ~80 lines of embedded configs from class
✅ **Reusability**: Other tools can import and use same config
✅ **Scalability**: Patterns for other tools in `tools/core/`
✅ **Testability**: Config can be mocked in unit tests

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|-----------|
| Config file not found | Low | Comprehensive fallback defaults provided |
| Config schema change | Low | Validate config structure on load |
| Performance (extra YAML loads) | Low | Implement config caching |
| Backward compatibility | None | All changes are internal, no API changes |

**Overall Risk**: **VERY LOW**

---

## Next Steps

1. Create `AgentQMS/config/artifact_template_config.yaml`
2. Execute steps 2-7 in order
3. Add config caching
4. Test with: `pytest tests/test_artifact_templates.py -v`
5. Verify all 7 artifact types still load correctly
6. Apply same pattern to other `tools/core/` files:
   - `tool_registry.py`
   - `plugins.py`
   - Any other files with embedded configs

