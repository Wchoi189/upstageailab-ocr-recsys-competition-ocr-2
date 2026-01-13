---
title: "Reusable Utilities — AI Integration Guide"
format: "markdown"
audience: "AI agents / LLMs"
integration_point: ".github/copilot-instructions.md"
last_updated: "2026-01-11"
---

# Reusable Utilities — AI Agent Integration Guide

**For copilot-instructions.md insertion**

---

## Section: Utility Scripts Discovery

Before writing custom code for configuration, paths, timestamps, or git operations, check if a reusable utility exists.

### Quick Reference Table

| Use Case | Utility | Import | Key Benefit |
|----------|---------|--------|------------|
| Load YAML config files | ConfigLoader | `from AgentQMS.tools.utils.config_loader import ConfigLoader` | ~2000x faster via caching |
| Find standard directories | paths | `from AgentQMS.tools.utils.paths import get_project_root, get_data_dir` | No hardcoded paths |
| Create/format timestamps | timestamps | `from AGentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst` | KST timezone handling |
| Get git branch/commit | git | `from AGentQMS.tools.utils.git import get_current_branch, get_commit_hash` | Graceful fallbacks |

### Copy-Paste Code Examples

**Load YAML configuration:**
```python
from AgentQMS.tools.utils.config_loader import ConfigLoader
loader = ConfigLoader()
config = loader.load('configs/train.yaml')
```

**Get project directory:**
```python
from AgentQMS.tools.utils.paths import get_data_dir
data_dir = get_data_dir()
```

**Create timestamp:**
```python
from AGentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst
timestamp = format_kst(get_kst_timestamp(), "%Y-%m-%d %H:%M:%S")
```

**Get git info:**
```python
from AGentQMS.tools.utils.git import get_current_branch, get_commit_hash
branch = get_current_branch()
commit = get_commit_hash()
```

### Discovery Resources

- **Quick Reference**: `context/utility-scripts/quick-reference.md`
- **Full Index**: `context/utility-scripts/utility-scripts-index.yaml`
- **Detailed Docs**: `context/utility-scripts/by-category/*/`

### Common Patterns

**Pattern 1: Load config and use paths**
```python
from AGentQMS.tools.utils.config_loader import ConfigLoader
from AGentQMS.tools.utils.paths import get_configs_dir
import os

loader = ConfigLoader()
config_path = os.path.join(get_configs_dir(), 'train.yaml')
config = loader.load(config_path)
```

**Pattern 2: Artifact with timestamp**
```python
from AGentQMS.tools.utils.paths import get_artifacts_dir
from AGentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst
import os

timestamp = get_kst_timestamp()
filename = f"report_{format_kst(timestamp, '%Y%m%d_%H%M%S')}.md"
filepath = os.path.join(get_artifacts_dir(), filename)
```

**Pattern 3: Artifact with metadata**
```python
from AGentQMS.tools.utils.git import get_current_branch, get_commit_hash
from AGentQMS.tools.utils.timestamps import format_kst, get_kst_timestamp

metadata = {
    'branch': get_current_branch(),
    'commit': get_commit_hash(),
    'timestamp': format_kst(get_kst_timestamp(), "%Y-%m-%d %H:%M:%S"),
}
```

### Key Performance Notes

- **ConfigLoader**: ~2000x speedup for repeated loads (LRU caching)
- **paths**: Avoids refactoring when directory structure changes
- **timestamps**: Consistent KST handling (important for international team)
- **git**: No subprocess overhead (fast, reliable)

### What NOT to Do

❌ Do NOT: `yaml.safe_load(open('config.yaml'))`
✅ DO: Use `ConfigLoader().load('config.yaml')`

❌ Do NOT: Hardcode paths like `/home/user/project/data/...`
✅ DO: Use `get_data_dir()`, `get_configs_dir()`, etc.

❌ Do NOT: Use `datetime.datetime.now()`
✅ DO: Use `get_kst_timestamp()`

❌ Do NOT: Call subprocess for git (`subprocess.run(['git', ...])`)
✅ DO: Use `get_current_branch()`, `get_commit_hash()`

### Help & Examples

- For detailed docs: See `context/utility-scripts/by-category/` directories
- For quick answers: Check `quick-reference.md`
- For machine-parseable index: See `utility-scripts-index.yaml`

### Automatic Discovery

This utility bundle is automatically suggested by the context system when you mention relevant keywords:
- "load yaml", "load config"
- "project root", "find directory"
- "timestamp", "current time"
- "branch", "commit hash"

---

## Integration Points

These utilities are:
- ✅ Production-ready
- ✅ Tested and documented
- ✅ Integrated with context bundling
- ✅ Available in all agent sessions
- ✅ Recommended for all projects

**Source Location**: `AgentQMS/tools/utils/`
**Tests**: `tests/utils/`
**Status**: Active & maintained
