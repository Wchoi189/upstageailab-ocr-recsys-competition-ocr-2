# Discovery Specification

**Tier**: 2 (Framework)
**Scope**: Tool Discovery, Context Bundling, and Script Manifests.

## 1. Discovery Rules

**Mechanism**: `list_tools.py`
*   **Source**: Scans `AgentQMS/tools/` for `_tool.py` or defined manifest.
*   **Manifest**: `utility-scripts-manifest.yaml` (The "Yellow Components").
    *   Scripts outside the core tool definition.

### Detection Logic
1.  **Workflows**: Detected via `task_boundary` calls + context.
2.  **Tools**: Auto-discovered via `__all__` in `__init__.py`.

## 2. Context Bundling (Smart Loading)

**System**: `context_bundle.py`
*   **Logic**: Uses `keywords` in specs + Task Description.
*   **Limit**: < 32k tokens (soft limit), chunks priority.
*   **Bundle Strategy**:
    *   **Direct Mention**: If task mentions "Hydra", load `configuration.spec.md`.
    *   **Implicit**: If task is "Fix Bug", load `debugging-sessions.yaml` logic (now `misc.spec.md`).

## 3. Script Manifest
Key Utils located in `scripts/`:
*   `analyze_imports.py` (Audit)
*   `generate_mechanized_graph.py` (Docs)
*   `benchmark.py` (Performance)
