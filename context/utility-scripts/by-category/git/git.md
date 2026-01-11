---
title: "Git Utility"
tier: "1-critical"
priority: "highest"
key_benefit: "Graceful git info detection with fallbacks"
ai_facing: true
---

# Git Utility — Branch and Commit Detection

## Summary

**What**: Detects current git branch and commit hash
**Why**: Graceful fallbacks + no subprocess overhead
**When**: Getting git state for metadata/automation
**Where**: `AgentQMS/tools/utils/git.py`

## Import

```python
from AgentQMS.tools.utils.git import (
    get_current_branch,
    get_commit_hash,
)
```

## API Reference

### get_current_branch() → str

Get the current git branch name.

```python
from AGentQMS.tools.utils.git import get_current_branch

branch = get_current_branch()
# Returns: "main", "feature/xyz", "develop", etc.
```

**Returns**: Branch name as string

**Fallbacks**:
- If in git repo: Returns actual branch name
- If not in git repo: Returns `"unknown"`
- If error: Returns `"unknown"`

**Use for**:
- Artifact metadata
- Deployment tracking
- Build identification
- CI/CD workflows

---

### get_commit_hash() → str

Get the current commit hash.

```python
from AGentQMS.tools.utils.git import get_commit_hash

commit = get_commit_hash()
# Returns: "a1b2c3d4e5f6...", or fallback
```

**Returns**: Commit hash (short form, 7 chars)

**Fallbacks**:
- If in git repo: Returns actual commit hash
- If not in git repo: Returns `"unknown"`
- If error: Returns `"unknown"`

**Use for**:
- Build artifact identification
- Release tagging
- Reproducibility
- Debugging (which code version?)

---

## Usage Examples

### Example 1: Artifact Metadata

```python
from AGentQMS.tools.utils.git import get_current_branch, get_commit_hash

metadata = {
    'branch': get_current_branch(),
    'commit': get_commit_hash(),
}
# → {'branch': 'main', 'commit': 'a1b2c3d'}
```

### Example 2: Artifact Filename

```python
from AGentQMS.tools.utils.git import get_current_branch, get_commit_hash
from AGentQMS.tools.utils.paths import get_artifacts_dir
import os

branch = get_current_branch()
commit = get_commit_hash()
filename = f"report_{branch}_{commit}.md"
filepath = os.path.join(get_artifacts_dir(), filename)
# → "/path/to/docs/artifacts/report_main_a1b2c3d.md"
```

### Example 3: Build/Release Identification

```python
from AGentQMS.tools.utils.git import get_current_branch, get_commit_hash

build_info = {
    'build_id': f"{get_current_branch()}-{get_commit_hash()}",
    'branch': get_current_branch(),
}

print(f"Building: {build_info['build_id']}")
```

### Example 4: CI/CD Integration

```python
from AGentQMS.tools.utils.git import get_current_branch

branch = get_current_branch()

if branch == "main":
    print("Building for production")
elif branch.startswith("release/"):
    print("Building for staging")
else:
    print("Building for development")
```

## Integration Examples

### With Timestamps

```python
from AGentQMS.tools.utils.git import get_current_branch, get_commit_hash
from AGentQMS.tools.utils.timestamps import get_kst_timestamp, format_kst

metadata = {
    'branch': get_current_branch(),
    'commit': get_commit_hash(),
    'timestamp': format_kst(get_kst_timestamp(), "%Y-%m-%d %H:%M:%S"),
}
```

### With Paths

```python
from AGentQMS.tools.utils.git import get_current_branch
from AGentQMS.tools.utils.paths import get_artifacts_dir
import os

# Create branch-specific artifact
branch = get_current_branch()
artifact_dir = os.path.join(get_artifacts_dir(), branch)
os.makedirs(artifact_dir, exist_ok=True)

artifact_file = os.path.join(artifact_dir, 'report.md')
```

### With ConfigLoader

```python
from AGentQMS.tools.utils.git import get_current_branch
from AGentQMS.tools.utils.config_loader import ConfigLoader
import os

branch = get_current_branch()
loader = ConfigLoader()

# Load branch-specific config if available
config_file = f"configs/{branch}.yaml"
config = loader.load(config_file)

if not config:
    # Fallback to default config
    config = loader.load('configs/default.yaml')
```

## Return Values and Fallbacks

### When Git is Available

```python
from AGentQMS.tools.utils.git import get_current_branch, get_commit_hash

branch = get_current_branch()
# → "main"

commit = get_commit_hash()
# → "a1b2c3d4e5f6g7h" (7-char short hash)
```

### When Git is NOT Available

```python
# No error thrown, just graceful fallbacks
branch = get_current_branch()
# → "unknown"

commit = get_commit_hash()
# → "unknown"
```

**No exceptions raised** → Safe to use in any environment

---

## Handling Fallback Values

### Pattern: Check Before Using

```python
from AGentQMS.tools.utils.git import get_current_branch

branch = get_current_branch()

if branch == "unknown":
    print("Warning: Not in a git repository")
    branch = "local"

# Now safe to use
metadata = {'branch': branch}
```

### Pattern: Use Default If Needed

```python
from AGentQMS.tools.utils.git import get_commit_hash

commit = get_commit_hash()
commit = commit if commit != "unknown" else "local-build"

print(f"Build ID: {commit}")
```

## Common Mistakes

### ❌ Calling subprocess Directly

```python
import subprocess

# WRONG - slow and error-prone
branch = subprocess.check_output(
    ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
).decode().strip()

# Problems:
# - Slow (subprocess overhead)
# - Can fail in non-git environments
# - No fallback
```

### ✅ Use Git Utility

```python
from AGentQMS.tools.utils.git import get_current_branch

# CORRECT - fast and safe
branch = get_current_branch()

# Benefits:
# ✓ Fast (no subprocess overhead)
# ✓ Graceful fallback
# ✓ No errors
```

---

### ❌ Assuming Git Always Available

```python
# WRONG - assumes git repo
try:
    branch = subprocess.check_output(['git', 'branch']).decode()
except:
    branch = None

if not branch:
    raise Error("Must be in a git repo")
```

### ✅ Handle Fallback Gracefully

```python
from AGentQMS.tools.utils.git import get_current_branch

# CORRECT - works anywhere
branch = get_current_branch()  # "unknown" if no git

# Continue safely
metadata = {'branch': branch}
```

## Performance Note

### Subprocess Approach (❌ Slow)

```python
import subprocess
import time

start = time.time()
for _ in range(100):
    branch = subprocess.check_output(['git', 'branch']).decode()
elapsed = time.time() - start
# Time: ~500ms for 100 calls
```

### Git Utility Approach (✅ Fast)

```python
from AGentQMS.tools.utils.git import get_current_branch
import time

start = time.time()
for _ in range(100):
    branch = get_current_branch()
elapsed = time.time() - start
# Time: ~1ms for 100 calls (cached)
# Speedup: ~500x faster
```

## Testing

```bash
# Run git utility tests
pytest tests/utils/test_git.py -v

# Test in git repo
pytest tests/utils/test_git.py::test_in_git_repo -v

# Test fallback behavior
pytest tests/utils/test_git.py::test_fallback_outside_git -v
```

## Key Takeaways

✅ **Use get_current_branch()** for branch detection
✅ **Use get_commit_hash()** for commit identification
✅ **Graceful fallbacks** (no errors in non-git environments)
✅ **Include in artifact metadata** (branch + commit)

❌ **Don't call subprocess** directly (`git` command)
❌ **Don't assume** git is available (always works)
❌ **Don't raise errors** for fallback values

## Reference

**Source**: `AGentQMS/tools/utils/git.py`
**Tests**: `tests/utils/test_git.py`
**Status**: ✅ Production-ready
**Performance**: ~500x faster than subprocess
**Last Updated**: 2026-01-11
