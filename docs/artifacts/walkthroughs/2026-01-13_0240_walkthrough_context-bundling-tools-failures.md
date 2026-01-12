# Assessment: Context Bundling & MCP Tooling Pain Points

## Executive Summary
The current Context Bundling and MCP Tooling system suffers from fragmentation, manual overhead, and "silent failures" where available resources (like `path_utils`) are not effectively surfaced to the agent. This assessment identifies key pain points and proposes a unified architecture to resolve them.

## Investigation: The Missing "Path Utils"
**User Report:** `path utils` or `utility catalog` should have been suggested but wasn't.
**Finding:** Confirmed Missing.
- **Manifest:** [AgentQMS/standards/tier2-framework/tool-catalog.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/standards/tier2-framework/tool-catalog.yaml) lists "utilities" but omits `path_utils`.
- **Bundles:** [AgentQMS/.agentqms/plugins/context_bundles/pipeline-development.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/.agentqms/plugins/context_bundles/pipeline-development.yaml) does not include `path_utils` or [paths.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/paths.py).
- **Logic:** [AgentQMS/tools/core/context_bundle.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/core/context_bundle.py) has hardcoded suggestion lists that also omit it.

## Identified Pain Points

### 1. Fragmentation of Tool Definitions
- **Issue:** Tool definitions are duplicated or split between [unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py) (for aggregation) and individual [mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/mcp_server.py) files (for implementation).
- **Impact:** Refactoring one requires updating the other. My recent refactor fixed [mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/mcp_server.py), but [unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py) still carries its own copy of tool metadata.
- **Evidence:** [unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py) defines `TOOLS_CONFIG` separately from the source-of-truth in `agent-debug-toolkit`.

### 2. Manual Path Management ("Import Hell")
- **Issue:** Scripts rely on fragile `sys.path.insert` hacks to find their dependencies.
- **Impact:** Moving files breaks utilities. Tests require intricate setup to run.
- **Evidence:** [mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/mcp_server.py) duplicates [ConfigLoader](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/config_loader.py#49-223) import logic with a fallback.

### 3. "Silent" Context Bundling
- **Issue:** The system expects the agent to "know" what's available or relies on opaque triggers.
- **Impact:** Useful utilities (like `path_utils`) sit unused because the agent isn't notified of their existence during relevant tasks.
- **Goal:** Context should be *proactive* and *explicit*.

### 4. Fragile URI/Alias Resolution
- **Issue:** Previous debugging showed `file://` vs `bundle://` mismatches.
- **Impact:** Tools fail to read resources even when they technically exist.

## Proposed Permanent Resolution: "Unified Registry"
1.  **Single Source of Truth**: All tools and bundles define their metadata in one place (e.g., `manifest.yaml` inside their package). Both [mcp_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/agent-debug-toolkit/src/agent_debug_toolkit/mcp_server.py) and [unified_server.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/scripts/mcp/unified_server.py) read from this single manifest.
2.  **Auto-Discovery**: The unified server scans for these manifests instead of having hardcoded lists.
3.  **Active Context Injection**: When a user enters a "coding" mode or opens specific file types (e.g., [.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/verify_refactor.py)), the system should proactively dump a "Utility Index" into the context, listing available helpers like `path_utils`.

## Next Steps
1.  Locate the missing `utility_catalog`.
2.  Prototype the "Unified Registry" pattern for one tool group.
3.  Implement the "Active Context Injection" for utilities.
