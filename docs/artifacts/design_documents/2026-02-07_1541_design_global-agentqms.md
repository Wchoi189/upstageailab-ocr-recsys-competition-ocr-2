# Design: Global Stateless AgentQMS

## 1. Problem Statement
Currently, `AgentQMS` (and its `unified_mcp` server) is tightly coupled to the project it resides in. The `project_root` is determined by locating the `AgentQMS` source code directory and looking at its parent. This means:
1.  You cannot run `AgentQMS` from a global installation to manage an arbitrary project.
2.  You cannot easily use one project's `AgentQMS` instance to manipulate another project's artifacts without complex path hacks.
3.  The server fails to start if it cannot find the `AgentQMS` code within the expected directory structure.

## 2. Proposed Architecture: Global & Stateless

To make `AgentQMS` a globally usable, stateless tool (similar to `git`, `npm`, or `uv`), we need to decouple the **Framework Scope** (where the code lives) from the **Project Scope** (where the work happens).

### 2.1. Dynamic Project Resolution
Modify `AgentQMS/tools/utils/config/config.py` to determine `project_root` dynamically:

1.  **Explict Override**: Check for `AGENTQMS_PROJECT_ROOT` environment variable.
2.  **Current Working Directory**: Default to `os.getcwd()` or traverse up from CWD to find a marker (e.g., `.agentqms/`, `.git`, `pyproject.toml`).
3.  **Fallback**: If no marker is found, assume CWD is the project root (allow running in empty dirs for `init`).

**Current Logic (Hardcoded):**
```python
def _detect_project_root(self, framework_root: Path) -> Path:
    # Looks relative to framework_root (the code location)
    if (framework_root.parent / "AGENTS.yaml").exists()...
```

**New Logic (Dynamic):**
```python
def _detect_project_root(self, framework_root: Path) -> Path:
    # 1. Env Var
    if env_root := os.getenv("AGENTQMS_PROJECT_ROOT"):
        return Path(env_root).resolve()

    # 2. Search upwards from CWD
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if (parent / ".agentqms").exists() or (parent / "AGENTS.yaml").exists():
            return parent

    # 3. Fallback to CWD (for init)
    return cwd
```

### 2.2. CLI Entry Point
Create a dedicated CLI entry point that acts as the global interface.

```bash
# Install globally
pip install agentqms-core # Comment: the cli should be "aqms" so that this will align with existing aqms cli

# Run anywhere
agentqms status
agentqms init
agentqms server start
```

### 2.3. The `init` Command
Since a scaffold might not exist, we need an initialization command.

**Command:** `agentqms init`
**Actions:**
1.  Create `.agentqms/` directory in CWD.
2.  Create default `settings.yaml`, `registry.yaml`, and `AGENTS.yaml`.
3.  (Optional) Scaffold standard directories (`docs/`, `tests/`) as defined in a "starter" template.

## 3. Implementation Plan

### Phase 1: Core Decoupling
- [ ] Refactor `ConfigLoader._detect_project_root` to use CWD traversal.
- [ ] Update `paths.py` to ensure it doesn't assume relative paths to `__file__`.
- [ ] Verify `unified_server.py` respects the dynamically resolved root.

### Phase 2: CLI Wrapper
- [ ] Create `AgentQMS/cli.py` (or update existing) to handle global commands.
- [ ] Register `main` entry point in `pyproject.toml`.

### Phase 3: Scaffolding (`init`)
- [ ] Implement `init` command in CLI.
- [ ] Create strict templates for the minimal `.agentqms` structure.

## 4. Addressing User Questions

### Interoperability
With this change, `unified_mcp` becomes a true "server". You can run:
```bash
# In Project B terminal
export AGENTQMS_PROJECT_ROOT="/path/to/project-b"
agentqms server start
```
This instance will manage Project B, regardless of where the `agentqms` code is installed.

### Corruption Risks
*   **Registry/Index**: The "registry" (data) is stored in `.agentqms/` outside the repo source code or `.mcp-telemetry.jsonl`. Using a distinct Project Root prevents cross-project contamination.
*   **Templates**: Templates are loaded from the Framework (builtin) and the Project (custom). Global usage will still load usage-specific templates correctly.
