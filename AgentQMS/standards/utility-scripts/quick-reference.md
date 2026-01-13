---
title: "Utility Scripts Quick Reference"
format: "markdown with frontmatter"
optimized_for: "AI agent consumption"
last_updated: "2026-01-11"
---

# Utility Scripts Quick Reference

**Machine-parseable format for AI agents. Use this for quick lookups.**

## Lookup Table (AI-Optimized)

| Utility | Module | Import | Key Benefit | Use When |
|---------|--------|--------|-------------|----------|
| **ConfigLoader** | `config_loader` | `from AgentQMS.tools.utils.config_loader import ConfigLoader` | ~2000x faster (caching) | Loading YAML config files |
| **paths** | `paths` | `from AgentQMS.tools.utils.paths import get_project_root` | No hardcoded paths | Finding standard directories |
| **timestamps** | `timestamps` | `from AgentQMS.tools.utils.timestamps import get_kst_timestamp` | KST timezone handling | Artifact metadata timestamps |
| **git** | `git` | `from AgentQMS.tools.utils.git import get_current_branch` | Graceful fallbacks | Detecting branch/commit info |

## Code Snippets (Copy-Paste Ready)

### Load YAML Config

```python
from AgentQMS.tools.utils.config_loader import ConfigLoader

loader = ConfigLoader()
config = loader.load('configs/train.yaml')
```

**Benefits**:
- Automatic LRU caching (100-item, 1h TTL)
- ~2000x speedup on repeated loads
- Graceful fallback to empty dict if missing
- Thread-safe

**Note**: First call ~5ms, cached calls ~0.002ms

---

### Get Standard Path

```python
from AgentQMS.tools.utils.paths import (
    get_project_root,
    get_data_dir,
    get_configs_dir,
    get_outputs_dir,
    get_models_dir,
    get_artifacts_dir,
)

root = get_project_root()
data = get_data_dir()
```

**Available paths**:
- `get_project_root()` → Repository root
- `get_data_dir()` → `data/`
- `get_configs_dir()` → `configs/`
- `get_outputs_dir()` → `outputs/`
- `get_models_dir()` → `data/checkpoints/`
- `get_artifacts_dir()` → `docs/artifacts/`

**Why use**: Avoid hardcoded paths, easier refactoring

---

### Create KST Timestamp

```python
from AgentQMS.tools.utils.timestamps import (
    get_kst_timestamp,
    format_kst,
    get_timestamp_age,
)

# Get current KST timestamp
now = get_kst_timestamp()

# Format it
formatted = format_kst(now, "%Y-%m-%d %H:%M:%S")

# Calculate age
age_hours = get_timestamp_age(now)
```

**When to use**: Artifact metadata, logging, automation workflows

---

### Get Git Information

```python
from AgentQMS.tools.utils.git import (
    get_current_branch,
    get_commit_hash,
)

branch = get_current_branch()  # "main", "feature/xyz", etc.
commit = get_commit_hash()      # "a1b2c3d4...", or fallback value
```

**Graceful fallbacks**: Returns sensible defaults if not in git repo

---

## Decision Tree (AI Should Use This)

```
Your task:
├── "Load a YAML configuration file"
│   └── USE: ConfigLoader (Tier 1 - CRITICAL)
│       ✓ ~2000x speedup
│       ✓ Automatic caching
│
├── "Find a standard project directory"
│   └── USE: paths (Tier 1 - CRITICAL)
│       ✓ No hardcoded paths
│       ✓ Maintainable
│
├── "Create/use a timestamp"
│   └── USE: timestamps (Tier 1 - CRITICAL)
│       ✓ KST timezone handling
│       ✓ Consistent formatting
│
├── "Get current branch or commit"
│   └── USE: git (Tier 1 - CRITICAL)
│       ✓ Graceful fallbacks
│       ✓ Error handling built-in
│
├── "Merge multiple config files"
│   └── USE: config (Tier 2 - Optional)
│       ✓ Hierarchical merging
│
├── "Setup application runtime"
│   └── USE: runtime (Tier 2 - Optional)
│       ✓ Path initialization
│
└── "Sync tasks to GitHub"
    └── USE: sync_github_projects (Tier 2 - Optional)
        ✓ GitHub API integration
```

## Performance Reference

| Operation | Tool | Time (First) | Time (Cached) | Speedup |
|-----------|------|--------------|---------------|---------|
| Load YAML config | ConfigLoader | ~5ms | ~0.002ms | ~2500x |
| Get path | paths | ~0.1ms | ~0.1ms | — |
| Get timestamp | timestamps | ~0.5ms | ~0.5ms | — |
| Get git info | git | ~10ms | ~10ms | — |

## Common Mistakes (Don't Do These)

❌ **BAD**: `yaml.safe_load(open('config.yaml'))`
✅ **GOOD**: `ConfigLoader().load('config.yaml')`

❌ **BAD**: `'/home/user/project/data/file.csv'`
✅ **GOOD**: `os.path.join(get_data_dir(), 'file.csv')`

❌ **BAD**: `datetime.datetime.now()`
✅ **GOOD**: `get_kst_timestamp()`

❌ **BAD**: `subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])`
✅ **GOOD**: `get_current_branch()`

## File Locations

```
AgentQMS/tools/utils/
├── config_loader.py      ← ConfigLoader class
├── paths.py              ← path functions
├── timestamps.py         ← timestamp functions
├── git.py                ← git functions
├── config.py             ← config merging (Tier 2)
├── runtime.py            ← runtime setup (Tier 2)
└── sync_github_projects.py ← GitHub integration (Tier 2)
```

## Testing

All utilities have tests in `tests/utils/`:

```bash
pytest tests/utils/test_config_loader.py
pytest tests/utils/test_paths.py
pytest tests/utils/test_timestamps.py
pytest tests/utils/test_git.py
```

## Context Bundling

This utilities bundle is integrated with AgentQMS context system.

**Auto-triggered by keywords**:
- "load yaml", "load config"
- "project root", "artifacts directory"
- "timestamp", "kst"
- "current branch", "git commit"

**Manual trigger**:
```bash
python AgentQMS/tools/utilities/suggest_context.py "your task"
```

## Additional Resources

- **Full documentation**: See `context/utility-scripts/by-category/*/`
- **Implementation examples**: In each utility's `.md` file
- **Source code**: `AgentQMS/tools/utils/`
- **Tests**: `tests/utils/`

---

**Last Updated**: 2026-01-11
**Format**: Optimized for LLM/AI agent parsing
**Status**: ✅ Ready for use
