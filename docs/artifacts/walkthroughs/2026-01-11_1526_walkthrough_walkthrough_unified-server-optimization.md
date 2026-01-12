---
ads_version: "1.0"
type: "walkthrough"
category: "documentation"
status: "active"
version: "1.0"
tags: "None"
title: "Unified Server Optimization & Refactoring"
date: "2026-01-11 15:26 (KST)"
branch: "main"
description: "Walkthrough of the unified server optimization."
---

# Walkthrough: Unified Server Optimization & Refactoring

## 1. Goal
Optimize the `unified_server.py` to reduce file system I/O overhead during resource lookups and improve code maintainability by externalizing configuration.

## 2. Changes Implemented

### 2.1 Configuration Externalization
Moved inline definitions to YAML files (Config-as-Code).

- **[NEW] `scripts/mcp/config/resources.yaml`**: Contains `RESOURCES_CONFIG` entries.
- **[NEW] `scripts/mcp/config/tools.yaml`**: Contains `Tool` definitions.

### 2.2 Logic Optimization in `unified_server.py`
- **Pre-Computed Lookup Maps**: initialized `URI_MAP` and `PATH_MAP` at startup.
    - `URI_MAP`: O(1) lookup for exact URIs.
    - `PATH_MAP`: O(1) lookup for resolved paths (handle aliases).
- **Graceful Error Handling**: Implemented a fix to check if a resource's `path` is `None` (for virtual resources) before attempting to `.resolve()` it, preventing startup crashes.
- **Dynamic Loaders**: Added `load_resources_config` and `load_tools_definitions` to load YAMLs at runtime.

### 2.3 Code Consolidation
- Extracted dynamic handlers (`agentqms://`, `experiments://`) into helper functions:
    - `_handle_experiments_list()`
    - `_handle_plugin_artifacts()`
    - `_handle_templates_list()`

## 3. Verification Results

### Automatic Verification
Ran `scripts/mcp/verify_server.py`:

```
--- Verifying Resources ---
Total Resources: 18
URI_MAP Size: 18
PATH_MAP Size: 14  <-- Correct (4 virtual resources excluded)
✅ Found compass://compass.json
✅ Found virtual resource agentqms://templates/list with path=None

--- Verifying Tools ---
Total Tools Loaded: 27
✅ Found validate_artifact tool

--- Verifying Read Resource (Logic) ---
✅ Successfully read virtual resource: 1 chunks
```

### Key Findings
- **Startup Time**: Negligible impact.
- **Cognitive Load**: `unified_server.py` reduced by approx. 400 lines (moved to config).
- **Correctness**: Virtual resources are correctly identified and do not crash the path resolver.

## 4. Next Steps
- Monitor server logs for any runtime edge cases.
- Consider adding schema validation for the YAML files in the future.
