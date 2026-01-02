# Import Handling Protocol

**Document ID**: `PROTO-DEV-023`
**Status**: ACTIVE
**Type**: Development Protocol

**For AI Agents**: Concise protocol - framework-agnostic import patterns.

---

## TL;DR

- Use `AgentQMS.agent_tools.utils.paths` for path resolution.
- Use absolute imports for framework (`from AgentQMS...`).
- Never manually manipulate `sys.path`.
- Never hardcode paths.

---

## Framework Path Utilities

```python
from AgentQMS.agent_tools.utils.paths import (
    get_framework_root,
    get_project_root,
    get_container_path,
)
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

# Ensure project root is on sys.path
ensure_project_root_on_sys_path()
```

---

## Import Patterns

### Framework Imports

```python
# ✅ CORRECT
from AgentQMS.agent_tools.core.artifact_workflow import ArtifactWorkflow
from AgentQMS.agent_tools.compliance.validate_artifacts import ArtifactValidator
from AgentQMS.agent_tools.utils.config import load_config
```

### Project-Specific Imports

```python
# ✅ CORRECT
from project_module import ProjectClass
from .local_module import LocalClass
```

### Rules

- ✅ If an import works, don't change it
- ✅ Don't convert absolute to relative imports unnecessarily
- ✅ Test imports in the actual runtime environment

---

## Common Pitfalls

### ❌ Don't Manually Manipulate sys.path

```python
# ❌ WRONG
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ✅ CORRECT
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path
ensure_project_root_on_sys_path()
```

### ❌ Don't Use Relative Imports for Framework

```python
# ❌ WRONG
from ...agent_tools.core import artifact_workflow

# ✅ CORRECT
from AgentQMS.agent_tools.core import artifact_workflow
```

### ❌ Don't Hardcode Paths

```python
# ❌ WRONG
framework_path = "/workspaces/project/AgentQMS"

# ✅ CORRECT
from AgentQMS.agent_tools.utils.paths import get_framework_root
framework_path = get_framework_root()
```

---

## Debugging Import Issues

1. **Check framework installation** - Is `AgentQMS/` in the project?
2. **Use framework path utilities** - Don't manipulate sys.path manually
3. **Verify project structure** - Framework expects containerized layout
4. **Check configuration** - Ensure framework root detection works

---

## Related

- **Path Utilities**: `AgentQMS/agent_tools/utils/paths.py`
- **Runtime Utilities**: `AgentQMS/agent_tools/utils/runtime.py`
- **Configuration**: `AgentQMS/agent_tools/utils/config.py`

