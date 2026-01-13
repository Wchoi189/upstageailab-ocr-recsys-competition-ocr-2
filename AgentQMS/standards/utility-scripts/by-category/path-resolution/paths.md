---
title: "Paths Utility"
tier: "1-critical"
priority: "highest"
key_benefit: "No hardcoded paths = maintainable code"
ai_facing: true
---

# Paths Utility — Standard Project Directory Resolution

## Summary

**What**: Resolves standard project directories consistently
**Why**: Avoid hardcoding paths (easier to refactor)
**When**: Any time you need a project directory
**Where**: `AgentQMS/tools/utils/paths.py`

## Import

```python
from AgentQMS.tools.utils.paths import (
    get_project_root,
    get_data_dir,
    get_configs_dir,
    get_outputs_dir,
    get_models_dir,
    get_artifacts_dir,
)
```

## Standard Directories

| Function | Returns | Purpose |
|----------|---------|---------|
| `get_project_root()` | `/path/to/repo/` | Repository root |
| `get_data_dir()` | `/path/to/repo/data/` | Data directory |
| `get_configs_dir()` | `/path/to/repo/configs/` | Config files |
| `get_outputs_dir()` | `/path/to/repo/outputs/` | Output artifacts |
| `get_models_dir()` | `/path/to/repo/data/checkpoints/` | Trained models |
| `get_artifacts_dir()` | `/path/to/repo/docs/artifacts/` | Artifact documents |

## API Reference

### get_project_root() → str

Get the project repository root.

```python
from AgentQMS.tools.utils.paths import get_project_root

root = get_project_root()
# Returns: "/home/user/project/" (or absolute path)
```

**Uses**: Git `.git` directory to locate root
**Fallback**: Current working directory

---

### get_data_dir() → str

Get the data directory.

```python
data_dir = get_data_dir()
# Returns: "<project_root>/data/"
```

**Use for**: Datasets, archives, interim data, processed data

---

### get_configs_dir() → str

Get the configs directory.

```python
configs_dir = get_configs_dir()
# Returns: "<project_root>/configs/"
```

**Use for**: Hydra configs, YAML files, configuration defaults

---

### get_outputs_dir() → str

Get the outputs directory.

```python
outputs_dir = get_outputs_dir()
# Returns: "<project_root>/outputs/"
```

**Use for**: Model predictions, inference results, processed outputs

---

### get_models_dir() → str

Get the models/checkpoints directory.

```python
models_dir = get_models_dir()
# Returns: "<project_root>/data/checkpoints/"
```

**Use for**: Trained model checkpoints, weights

---

### get_artifacts_dir() → str

Get the artifacts documentation directory.

```python
artifacts_dir = get_artifacts_dir()
# Returns: "<project_root>/docs/artifacts/"
```

**Use for**: Generated artifacts, markdown docs, reports

---

## Usage Examples

### Example 1: Construct File Paths

```python
from AgentQMS.tools.utils.paths import get_data_dir
import os

data_dir = get_data_dir()
dataset_path = os.path.join(data_dir, 'raw', 'dataset.csv')
# → "<project_root>/data/raw/dataset.csv"
```

### Example 2: Load Config File

```python
from AgentQMS.tools.utils.paths import get_configs_dir
from AgentQMS.tools.utils.config_loader import ConfigLoader
import os

configs_dir = get_configs_dir()
config_path = os.path.join(configs_dir, 'train.yaml')
config = ConfigLoader().load(config_path)
```

### Example 3: Save Artifacts

```python
from AgentQMS.tools.utils.paths import get_artifacts_dir
import os

artifacts_dir = get_artifacts_dir()
output_path = os.path.join(artifacts_dir, 'report.md')

with open(output_path, 'w') as f:
    f.write("# Report\n")
```

### Example 4: Find All Models

```python
from AgentQMS.tools.utils.paths import get_models_dir
import glob

models_dir = get_models_dir()
checkpoint_files = glob.glob(os.path.join(models_dir, '*.pt'))
```

## Why Not Hardcode Paths?

### ❌ Bad (Hardcoded)

```python
data_path = "/home/user/project/data/dataset.csv"
config_path = "/home/user/project/configs/train.yaml"
```

**Problems**:
- ❌ Path breaks on different machines
- ❌ Refactoring = search-and-replace everywhere
- ❌ Not discoverable (hardcoded strings)
- ❌ CI/CD failures (wrong paths)

### ✅ Good (Using paths utility)

```python
from AgentQMS.tools.utils.paths import get_data_dir, get_configs_dir
import os

data_path = os.path.join(get_data_dir(), 'dataset.csv')
config_path = os.path.join(get_configs_dir(), 'train.yaml')
```

**Benefits**:
- ✅ Works on any machine
- ✅ Refactoring = change in one place
- ✅ Code is self-documenting
- ✅ CI/CD works without modification

## Integration Examples

### With ConfigLoader

```python
from AgentQMS.tools.utils.paths import get_configs_dir
from AgentQMS.tools.utils.config_loader import ConfigLoader
import os

loader = ConfigLoader()
config_path = os.path.join(get_configs_dir(), 'train.yaml')
config = loader.load(config_path)
```

### With Timestamps

```python
from AgentQMS.tools.utils.paths import get_artifacts_dir
from AgentQMS.tools.utils.timestamps import format_kst, get_kst_timestamp
import os

artifacts_dir = get_artifacts_dir()
timestamp = format_kst(get_kst_timestamp(), "%Y%m%d_%H%M%S")
output_file = os.path.join(artifacts_dir, f"report_{timestamp}.md")
```

### With Git Info

```python
from AgentQMS.tools.utils.paths import get_artifacts_dir
from AgentQMS.tools.utils.git import get_current_branch
import os

artifacts_dir = get_artifacts_dir()
branch = get_current_branch()
output_file = os.path.join(artifacts_dir, f"report_{branch}.md")
```

## Directory Structure Reference

```
project/
├── configs/              ← get_configs_dir()
│   ├── train.yaml
│   ├── eval.yaml
│   └── ...
├── data/                 ← get_data_dir()
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── checkpoints/      ← get_models_dir()
├── docs/
│   └── artifacts/        ← get_artifacts_dir()
├── outputs/              ← get_outputs_dir()
└── .git/                 ← Used to find get_project_root()
```

## Advanced: Custom Project Root

For special cases:

```python
import os
from AgentQMS.tools.utils.paths import get_data_dir

# Override project root detection
os.environ['PROJECT_ROOT'] = '/custom/project/path'

data_dir = get_data_dir()
# Will use custom root
```

## Testing

```bash
# Run paths utility tests
pytest tests/utils/test_paths.py -v

# Verify all directories exist
pytest tests/utils/test_paths.py::test_directories_exist -v
```

## Key Takeaways

✅ **Use paths utility** for all directory references
✅ **No hardcoded paths** in application code
✅ **Consistent across codebase** (single source of truth)
✅ **Easy refactoring** (change in paths.py only)

❌ **Don't hardcode** `/home/username/project/...`
❌ **Don't assume** relative paths work (use absolute)
❌ **Don't mix** paths utility with hardcoded strings

## Reference

**Source**: `AgentQMS/tools/utils/paths.py`
**Tests**: `tests/utils/test_paths.py`
**Status**: ✅ Production-ready
**Last Updated**: 2026-01-11
