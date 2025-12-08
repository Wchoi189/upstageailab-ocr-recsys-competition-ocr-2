---
title: "Additional Performance Optimization Suggestions"
date: "2025-12-06 18:09 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---







# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Additional Performance Optimization Suggestions**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Additional Performance Optimization Suggestions

## Overview

Beyond the three-phase implementation plan, here are additional optimization opportunities for the Streamlit Command Builder app.

## 1. Widget Key Optimization

### Problem
Streamlit widgets with complex keys can cause unnecessary re-renders. Long keys with dynamic values can trigger cache invalidation.

### Solution
Use stable, simple widget keys:

```python
# BEFORE:
widget_key = f"{SCHEMA_PREFIX}__{key}__{some_dynamic_value}"

# AFTER:
widget_key = f"{SCHEMA_PREFIX}__{key}"  # Keep keys simple and stable
```

**Impact:** Reduces unnecessary widget re-creation and cache invalidation.

## 2. Session State Cleanup

### Problem
Session state can accumulate unused keys over time, increasing memory usage and slowing down state lookups.

### Solution
Periodically clean up unused session state keys:

```python
def cleanup_session_state():
    """Remove unused session state keys."""
    keys_to_keep = {
        "command_builder_state",
        "command_builder_state_hash",
        # ... other essential keys
    }

    keys_to_remove = [
        key for key in st.session_state.keys()
        if key.startswith("command_builder_") and key not in keys_to_keep
    ]

    for key in keys_to_remove:
        if key not in st.session_state:
            continue
        # Only remove if not recently used
        st.session_state.pop(key, None)
```

**Impact:** Reduces memory usage and improves state lookup performance.

## 3. YAML Schema Validation Caching

### Problem
Schema validation happens on every render, even if schema hasn't changed.

### Solution
Cache validation results based on schema file modification time:

```python
import os
from pathlib import Path

@st.cache_data(ttl=3600)
def _get_schema_mtime(schema_path: str) -> float:
    """Get schema file modification time for cache invalidation."""
    return os.path.getmtime(schema_path)

def validate_schema(schema_path: str, values: dict) -> list[str]:
    """Validate with cache based on file mtime."""
    mtime = _get_schema_mtime(schema_path)
    cache_key = f"validation_{schema_path}_{mtime}_{hash(str(values))}"

    # Use cached validation if available
    # ... validation logic ...
```

**Impact:** Reduces redundant validation work.

## 4. Checkpoint Scanning Optimization

### Problem
Full directory scans for checkpoints are expensive, especially with many experiments.

### Solution
Use incremental scanning with file watching (if available) or shorter cache TTL:

```python
# Option 1: Incremental scanning
def get_available_checkpoints_incremental(self) -> list[str]:
    """Only scan new directories since last scan."""
    # Track last scan time
    # Only scan directories modified since last scan
    # Merge with cached results
    pass

# Option 2: Background scanning
import threading

def scan_checkpoints_background():
    """Scan checkpoints in background thread."""
    # Update cache asynchronously
    pass
```

**Impact:** Reduces checkpoint scanning time from 50-100ms to < 10ms.

## 5. Form State Optimization

### Problem
Form state is stored in session state with complex nested structures, causing serialization overhead.

### Solution
Use flatter state structure and only store changed values:

```python
# BEFORE:
st.session_state[f"{prefix}__{key}"] = value  # Many keys

# AFTER:
# Store only changed values in a single dict
if "form_changes" not in st.session_state:
    st.session_state["form_changes"] = {}
st.session_state["form_changes"][f"{prefix}__{key}"] = value
```

**Impact:** Reduces session state serialization time.

## 6. Import Optimization

### Problem
Heavy imports at module level slow down app startup.

### Solution
Use lazy imports for heavy dependencies:

```python
# BEFORE (at top of file):
from ocr.models.core import registry  # Heavy import

# AFTER (inside function):
def get_registry():
    from ocr.models.core import registry  # Lazy import
    return registry
```

**Impact:** Faster app startup, especially on first load.

## 7. Component Rendering Optimization

### Problem
All form fields render even if not visible, causing unnecessary work.

### Solution
Use Streamlit's conditional rendering more aggressively:

```python
# BEFORE:
for element in elements:
    render_element(element)  # Renders all

# AFTER:
for element in elements:
    if _is_visible(element, values):  # Check visibility first
        render_element(element)
    else:
        values[element.key] = None  # Set to None without rendering
```

**Impact:** Reduces rendering time for hidden fields.

## 8. Memory-Mapped File Access

### Problem
Repeated file reads for configs and schemas cause I/O overhead.

### Solution
Use memory-mapped files for frequently accessed configs (advanced):

```python
import mmap

def load_config_mmap(config_path: str) -> dict:
    """Load config using memory-mapped file."""
    with open(config_path, 'r') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            return yaml.safe_load(mm)
```

**Impact:** Faster config loading for large files.

## 9. Debouncing User Input

### Problem
Rapid user input (typing, slider changes) triggers multiple reruns.

### Solution
Debounce user input to reduce reruns:

```python
# Use Streamlit's built-in debouncing where possible
# For custom inputs, implement debouncing:
import time

def debounced_input(key: str, delay: float = 0.5):
    """Debounce input to reduce reruns."""
    if key not in st.session_state:
        st.session_state[key] = {"value": "", "last_update": 0}

    current_time = time.time()
    if current_time - st.session_state[key]["last_update"] > delay:
        # Process input
        pass
```

**Impact:** Reduces unnecessary reruns during rapid input.

## 10. Progressive Image Loading (if applicable)

### Problem
If the app loads images, they can block rendering.

### Solution
Load images progressively with placeholders:

```python
# Show placeholder first
placeholder = st.empty()
placeholder.image("placeholder.png")

# Load actual image in background
actual_image = load_image_async()
placeholder.image(actual_image)
```

**Impact:** Improves perceived performance for image-heavy pages.

## 11. Database/File System Indexing

### Problem
Repeated file system scans for configs and checkpoints.

### Solution
Maintain an index file that's updated when configs change:

```python
# Create index on first run or when configs change
def build_config_index():
    """Build index of all config files."""
    index = {
        "models": list_models(),
        "architectures": list_architectures(),
        # ... etc
    }
    # Save to JSON file
    # Use index instead of scanning
    pass
```

**Impact:** Eliminates file system scans entirely.

## 12. WebSocket/Server-Sent Events (Advanced)

### Problem
Full page reruns on every interaction.

### Solution
Use Streamlit's experimental features or custom components for real-time updates without full reruns (advanced, may not be applicable).

## Implementation Priority

### High Priority (Quick Wins)
1. Widget Key Optimization
2. Session State Cleanup
3. Import Optimization

### Medium Priority (Moderate Impact)
4. Form State Optimization
5. Component Rendering Optimization
6. Debouncing User Input

### Low Priority (Advanced/Complex)
7. YAML Schema Validation Caching
8. Checkpoint Scanning Optimization
9. Memory-Mapped File Access
10. Database/File System Indexing

## Testing Considerations

For each optimization:
1. Measure performance impact
2. Verify no regressions
3. Test edge cases
4. Monitor memory usage
5. Document any trade-offs

## When to Apply

- **After Phase 1:** Apply high-priority optimizations
- **After Phase 2:** Apply medium-priority optimizations
- **After Phase 3:** Apply low-priority optimizations if needed

## Notes

- Not all optimizations may be applicable to this specific app
- Some optimizations may have trade-offs (complexity vs. performance)
- Measure before and after to verify impact
- Don't over-optimize prematurely
