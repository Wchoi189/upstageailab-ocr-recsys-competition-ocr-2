# Duplicate File Detection and Split-Brain Scenarios

**Source**: Conversation analysis (follow-up session)
**Date**: 2026-01-23
**Context**: Ghost code phenomenon - duplicate file detection
**Python Manager**: uv

---

## Overview

Duplicate files are a severe manifestation of the "Ghost Code" phenomenon. When the same logic exists in multiple locations, AI agents and developers can get trapped in a cycle of "fixing" code that isn't actually being executed, leading to **split-brain scenarios** where updates to one file have no effect on runtime behavior.

---

## The Duplicate Trap

### Problem Description

Having duplicate files (e.g., `ocr/core/lightning/utils/config_utils.py` and `ocr/core/utils/config_utils.py`) creates a split-brain scenario where:

1. **Different import paths** point to different physical files
2. **Updates to one file** don't affect the other
3. **Python treats them as separate modules** (`module1 is module2` returns `False`)
4. **Debugging becomes impossible** - fixes don't take effect

### Classic Symptoms

- Code changes in one file don't affect runtime behavior
- Import statements work from both locations
- Stack traces reference the "wrong" file
- AI agents fix the same issue repeatedly without success
- Circular-sounding dependencies between utility layers

---

## Detection Techniques

### 1. The Shadow Import Test

**Purpose**: Prove that duplicate files are being treated as separate modules

**Command**:
```bash
python -c "import ocr.core.lightning.utils.config_utils as c1; \
           import ocr.core.utils.config_utils as c2; \
           print(f'C1: {c1.__file__}\nC2: {c2.__file__}'); \
           print(f'Same object? {c1 is c2}')"
```

**Expected Output** (healthy):
```
C1: /path/to/ocr/core/utils/config_utils.py
C2: /path/to/ocr/core/utils/config_utils.py
Same object? True
```

**Problem Output** (split-brain):
```
C1: /path/to/ocr/core/lightning/utils/config_utils.py
C2: /path/to/ocr/core/utils/config_utils.py
Same object? False  ← CRITICAL: Two different modules!
```

### 2. Find Duplicate Filenames

**Purpose**: Identify files with the same name in different locations

**Command**:
```bash
# Find all Python files with duplicate basenames
find ocr/ -name "*.py" -type f | \
  awk -F/ '{print $NF, $0}' | \
  sort | \
  awk '{if (prev == $1) print prev_path "\n" $0; prev = $1; prev_path = $0}' | \
  cut -d' ' -f2-
```

### 3. Content Similarity Check

**Purpose**: Find files with similar content that might be duplicates

**Command**:
```bash
# Find files with similar content (requires fdupes or similar)
fdupes -r ocr/ | grep "\.py$"

# Or use checksums
find ocr/ -name "*.py" -type f -exec md5sum {} + | \
  sort | \
  awk '{if (prev == $1) print prev_path "\n" $2; prev = $1; prev_path = $2}'
```

---

## Common Duplicate Patterns

### Pattern 1: Utility Layer Blur

**Problem**: Generic utilities duplicated in domain-specific locations

**Example**:
```
ocr/core/utils/config_utils.py          ← Source of truth
ocr/core/lightning/utils/config_utils.py ← Duplicate
```

**Root Cause**: Unclear separation between core utilities and domain-specific utilities

**Solution**: Consolidate to single location, use imports for domain-specific needs

### Pattern 2: Circular Dependencies

**Problem**: Two files importing from each other

**Example**:
```python
# ocr/core/lightning/utils/config_utils.py
from ocr.core.utils.config_utils import ensure_dict

# ocr/core/utils/config_utils.py
from ocr.core.lightning.utils.config_utils import extract_normalize_stats
```

**Root Cause**: Blurred boundaries between layers

**Solution**: Establish clear dependency hierarchy (core → domain → application)

### Pattern 3: Legacy + Refactored Versions

**Problem**: Old version kept "just in case" alongside new version

**Example**:
```
ocr/detection/models/dbnet.py      ← Legacy
ocr/domains/detection/models/dbnet.py ← New
```

**Root Cause**: Incomplete refactoring, fear of breaking changes

**Solution**: Aggressive deletion with version control safety net

---

## The "Kill-Duplicate" Plan

### Phase 1: Identification

1. **Run shadow import test** on suspected duplicates
2. **Compare file contents** to confirm duplication
3. **Trace import usage** to find which version is actually used
4. **Check for circular dependencies**

### Phase 2: Consolidation

1. **Choose source of truth**:
   - Prefer `core/` over domain-specific locations for generic utilities
   - Prefer domain-specific for domain logic
   - Follow project architecture guidelines

2. **Merge logic**:
   - Combine unique functionality from both files
   - Ensure compliance with coding standards
   - Add proper documentation

3. **Update imports**:
   - Find all import statements referencing the duplicate
   - Update to point to source of truth
   - Use AST-grep for structural search

### Phase 3: Deletion

1. **Delete duplicate file** (don't rename to `.bak`)
2. **Remove from version control**
3. **Verify no broken imports**:
   ```bash
   python scripts/audit/master_audit.py
   ```

### Phase 4: Prevention

1. **Add lint rules** to detect future duplicates
2. **Document architecture** clearly
3. **Code review checklist** for new files

---

## Example: config_utils Consolidation

### Problem Analysis

**Duplicate Files**:
- `ocr/core/utils/config_utils.py` (generic utilities)
- `ocr/core/lightning/utils/config_utils.py` (Lightning-specific)

**Issues Found**:
1. **Circular dependency**: Lightning version imports from core version
2. **Standards violation**: Uses `isinstance(cfg, dict)` instead of `is_config(cfg)`
3. **Hardcoded logic**: `extract_normalize_stats` manually iterates through transform names
4. **Silent failures**: Returns `None` if config structure changes

### Consolidation Diff

```diff
--- a/ocr/core/utils/config_utils.py
+++ b/ocr/core/utils/config_utils.py
@@ -1,11 +1,38 @@
-"""Core utility helpers for configuration."""
+"""Core utility helpers for configuration and extraction."""
 from __future__ import annotations
-from typing import Any
+from typing import Any, TypeGuard
 from omegaconf import DictConfig, ListConfig, OmegaConf
+import numpy as np

+def is_config(obj: Any) -> TypeGuard[dict | DictConfig]:
+    """Returns True if object is a dict or an OmegaConf DictConfig."""
+    return isinstance(obj, (dict, DictConfig))
+
 def ensure_dict(cfg: Any, resolve: bool = True) -> dict | list | Any:
     """Recursively converts OmegaConf objects to native Python types."""
     if isinstance(cfg, (DictConfig, ListConfig)):
         return OmegaConf.to_container(cfg, resolve=resolve)
     return cfg
+
+def extract_normalize_stats(config: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
+    """Locate normalize transform statistics using structural search."""
+    transforms_cfg = getattr(config, "transforms", None)
+    if not transforms_cfg:
+        return None, None
+
+    # Search through all standard transform slots
+    for attr in ("train_transform", "val_transform", "test_transform", "predict_transform"):
+        section = getattr(transforms_cfg, attr, None)
+        transforms = getattr(section, "transforms", None) if section else None
+
+        if isinstance(transforms, ListConfig):
+            for transform in transforms:
+                t_dict = ensure_dict(transform, resolve=True)
+                if is_config(t_dict) and t_dict.get("_target_") == "albumentations.Normalize":
+                    mean = t_dict.get("mean")
+                    std = t_dict.get("std")
+                    if mean is not None and std is not None:
+                        return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)
+
+    return None, None
```

### Post-Consolidation Steps

1. **Delete duplicate**:
   ```bash
   rm ocr/core/lightning/utils/config_utils.py
   ```

2. **Update imports**:
   ```bash
   # Find all imports of the duplicate
   grep -r "from ocr.core.lightning.utils.config_utils import" ocr/

   # Update to core version
   sed -i 's/from ocr.core.lightning.utils.config_utils/from ocr.core.utils.config_utils/g' \
     ocr/pipelines/orchestrator.py
   ```

3. **Verify**:
   ```bash
   python scripts/audit/master_audit.py
   python -c "from ocr.core.utils.config_utils import extract_normalize_stats"
   ```

---

## Doc-Sync Audit Tool

### Purpose

Ensure that **Agent Standards** (YAML) and **Python Implementation** are in perfect alignment, preventing code from violating documented standards.

### Architecture

**Components**:
1. **Pattern Extractor**: Parses `configuration-standards.yaml` for bad patterns
2. **Structural Scanner**: Uses AST-grep to find violations
3. **Cross-Referencer**: Checks compliance tier of scanned files
4. **Reporter**: Fails build if violations found in compliant files

### Implementation

**Script**: `AgentQMS/tools/compliance/doc_sync_audit.py`

```python
import yaml
import subprocess

def run_compliance_audit(standards_path: str, target_dir: str):
    with open(standards_path) as f:
        standards = yaml.safe_load(f)

    for rule in standards.get('rules', []):
        print(f"Checking Rule: {rule['id']}...")
        for pattern in rule.get('bad_patterns', []):
            # Escape the pattern for shell execution
            escaped_pattern = pattern.replace('"', '\\"')
            cmd = f"adt sg-search --pattern \"{escaped_pattern}\" --path {target_dir}"

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if "Matches: 0" not in result.stdout and result.returncode == 0:
                print(f"🚨 VIOLATION FOUND for rule {rule['id']} in {target_dir}")
                print(result.stdout)
                # In CI, we would sys.exit(1) here

if __name__ == "__main__":
    run_compliance_audit(
        "AgentQMS/standards/tier2-framework/configuration-standards.yaml",
        "ocr/"
    )
```

### Usage

**Manual Audit**:
```bash
uv run python AgentQMS/tools/compliance/doc_sync_audit.py
```

**CI/CD Integration**:
```yaml
# .github/workflows/compliance.yml
name: Code Compliance Audit

on: [push, pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Doc-Sync Audit
        run: uv run python AgentQMS/tools/compliance/doc_sync_audit.py
```

**Example Violation Detection**:
```bash
# Check for isinstance(cfg, dict) violations
adt sg-search --pattern "isinstance($CFG, dict)" --path ocr/core/lightning/utils/config_utils.py
```

---

## Prevention Strategies

### 1. The "Atomic Delete" Rule

**Rule**: When you find a duplicate, delete it immediately. Don't rename to `.bak`.

**Rationale**:
- AI agents often read `.py.bak` files and get confused
- Backup files clutter the workspace
- Version control provides safety net

**Implementation**:
```bash
# Wrong
mv duplicate.py duplicate.py.bak

# Right
git rm duplicate.py
# or
rm duplicate.py && git add duplicate.py
```

### 2. Architecture Documentation

**Requirement**: Clear documentation of module organization

**Example Structure**:
```
ocr/
├── core/           # Generic, reusable utilities
│   ├── utils/      # Pure utilities (no domain logic)
│   └── lightning/  # PyTorch Lightning specific
├── domains/        # Domain-specific logic
│   ├── detection/
│   └── recognition/
└── pipelines/      # Application orchestration
```

**Rule**: Generic utilities go in `core/utils/`, domain-specific in `domains/{domain}/utils/`

### 3. Import Linting

**AST-Grep Rule**: Detect imports from deprecated locations

```yaml
# rules/no-duplicate-imports.yaml
id: no-duplicate-imports
language: python
rule:
  any:
    - pattern: from ocr.core.lightning.utils.config_utils import $$$
    - pattern: import ocr.core.lightning.utils.config_utils
message: "Import from deprecated location. Use ocr.core.utils.config_utils instead."
severity: error
```

### 4. Pre-Commit Hooks

**Hook**: Check for duplicate filenames before commit

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Find duplicate basenames
DUPLICATES=$(find ocr/ -name "*.py" -type f | \
  awk -F/ '{print $NF}' | \
  sort | \
  uniq -d)

if [ ! -z "$DUPLICATES" ]; then
  echo "❌ Duplicate filenames found:"
  echo "$DUPLICATES"
  echo "Please consolidate before committing."
  exit 1
fi
```

---

## Troubleshooting

### Issue: Shadow import test shows same object but code still doesn't work

**Cause**: Editable install not refreshed after file deletion

**Solution**:
```bash
uv pip install -e . --force-reinstall
```

### Issue: Can't find which version is actually being used

**Cause**: Import path ambiguity

**Solution**:
```python
# Add to code temporarily
import sys
import ocr.core.utils.config_utils
print(f"Loaded from: {ocr.core.utils.config_utils.__file__}")
print(f"Module ID: {id(ocr.core.utils.config_utils)}")
```

### Issue: Deleting duplicate breaks imports

**Cause**: Some code still imports from old location

**Solution**:
```bash
# Find all imports before deleting
grep -r "from ocr.core.lightning.utils.config_utils" ocr/
grep -r "import ocr.core.lightning.utils.config_utils" ocr/

# Update all imports first, then delete
```

---

## Best Practices

### For Developers

1. **Before creating a new utility file**, check if similar functionality exists
2. **Use absolute imports** to make dependencies clear
3. **Follow architecture guidelines** for file placement
4. **Run duplicate detection** before committing

### For AI Agents

1. **Always run shadow import test** when debugging import issues
2. **Check for duplicates** before creating new files
3. **Update all imports** when consolidating files
4. **Verify with audit** after changes

### For Code Review

1. **Check for duplicate filenames** in PR
2. **Verify import paths** follow architecture
3. **Ensure deleted files** are actually deleted (not renamed)
4. **Run compliance audit** before merging

---

## See Also

- `shim_antipatterns_guide.md` - Why shims become toxic
- `migration_guard_implementation.md` - Environment validation
- `adt_usage_patterns.md` - Structural code analysis
- `../implementation_guides/auto_align_hydra_script.md` - Automated fixing
