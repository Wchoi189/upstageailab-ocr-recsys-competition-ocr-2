---
ads_version: "1.0"
type: "assessment"
category: "evaluation"
status: "active"
version: "1.0"
tags: "agentqms, ci-cd, configuration, assessment"
title: "AgentQMS Autofix Batch Job Path Mutation Analysis"
date: "2026-01-09 14:51 (KST)"
branch: "main"
description: "Assessment of the AgentQMS autofix batch job behavior that changes absolute paths from local to CI environment paths"
---

## Executive Summary

**Finding**: The AgentQMS autofix batch job (commit `fe10f5d`) is **working correctly but creating unnecessary churn** by regenerating environment-specific absolute paths in state files.

**Root Cause**: The `AgentQMS/.agentqms/state/plugins.yaml` and `.agentqms/effective.yaml` files contain absolute filesystem paths that differ between:
- Local development: `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/...`
- GitHub Actions CI: `/home/runner/work/upstageailab-ocr-recsys-competition-ocr-2/...`

**Impact**: 
- ✅ No functional issues - paths are regenerated correctly in each environment
- ⚠️ Unnecessary git churn - every CI run creates "changes" that are purely cosmetic
- ⚠️ Pollutes PR diffs with environment-specific path changes
- ⚠️ Makes it harder to review actual artifact fixes

**Recommendation**: **Environment-specific state files should be gitignored** or use relative paths.

---

## Detailed Analysis

### What Happened in Commit fe10f5d

The autofix workflow ran and made **legitimate changes**:
1. Added 2 new bug reports to the index
2. Marked 1 bug report as completed
3. Updated timestamps
4. **Also regenerated all absolute paths** in state files

```diff
discovery_paths:
-  framework: /workspaces/.../AgentQMS/conventions/plugins
+  framework: /home/runner/work/.../AgentQMS/conventions/plugins
-  project: /workspaces/.../.agentqms/plugins  
+  project: /home/runner/work/.../.agentqms/plugins
```

### Files Affected by Path Mutation

1. **`.agentqms/state/plugins.yaml`** (16 lines changed)
   - `discovery_paths.framework`
   - `discovery_paths.project`
   - All `plugin_metadata[].path` entries

2. **`.agentqms/effective.yaml`** (2 lines changed)
   - `metadata.generated_at` timestamp

3. **`uv.lock`** (5430 lines changed)
   - Unrelated dependency changes

### Code Flow Analysis

The path regeneration happens in this sequence:

1. **Workflow**: `.github/workflows/agentqms-autofix.yml`
   ```yaml
   - name: Regenerate indexes
     run: |
       cd AgentQMS/bin
       make reindex
   ```

2. **Makefile**: `AgentQMS/bin/Makefile`
   ```makefile
   reindex:
       uv run python ../tools/documentation/reindex_artifacts.py
   ```

3. **Reindex Tool**: Calls plugin system which generates state snapshot

4. **Plugin Discovery**: `AgentQMS/tools/core/plugins/discovery.py`
   ```python
   def get_discovery_paths(self) -> dict[str, str]:
       return {
           "framework": str(self.framework_plugins_dir),  # Absolute path!
           "project": str(self.project_plugins_dir),      # Absolute path!
       }
   ```

5. **Snapshot Writer**: `AgentQMS/tools/core/plugins/snapshot.py`
   ```python
   def write(self, registry: PluginRegistry, discovery_paths: dict[str, str]):
       snapshot = self._build_snapshot(registry, discovery_paths)
       # Writes discovery_paths verbatim to YAML
   ```

### Why This Happens

The plugin system captures **runtime absolute paths** during:
- Development: Codespace/devcontainer paths
- CI: GitHub Actions runner paths

These paths are serialized to YAML for "debugging/logging" but become version-controlled state that changes with every environment switch.

---

## Is This Behavior Necessary?

**No.** The absolute paths serve no functional purpose:

### Current Behavior
```yaml
# Generated in CI
discovery_paths:
  framework: /home/runner/work/.../AgentQMS/conventions/plugins
  project: /home/runner/work/.../.agentqms/plugins
```

### Why It's Unnecessary

1. **Not Used at Runtime**: The paths are generated fresh on every run
2. **Environment-Specific**: Different in dev vs CI vs prod
3. **No Validation**: Nothing validates these paths later
4. **Purely Informational**: Used for debugging output only

### What's Actually Needed

The plugin system **re-discovers** paths on every invocation:
- Uses `PROJECT_ROOT` detection
- Resolves relative paths
- Doesn't read from `plugins.yaml`

The `plugins.yaml` file is a **snapshot for debugging**, not a configuration input.

---

## Recommendations

### 🔴 Priority 1: Gitignore Generated State Files

**Action**: Add to `.gitignore`:
```gitignore
# AgentQMS generated state files
AgentQMS/.agentqms/state/
.agentqms/state/
.agentqms/effective.yaml
```

**Rationale**:
- These files are regenerated on every run
- They contain environment-specific data
- They don't affect functionality
- Similar to `__pycache__` or `.pytest_cache`

### 🟡 Priority 2: Use Relative Paths in Snapshots

**Action**: Modify `AgentQMS/tools/core/plugins/discovery.py`:
```python
def get_discovery_paths(self) -> dict[str, str]:
    """Return relative discovery paths for portability."""
    project_root = self.framework_plugins_dir.parents[2]  # Or detect dynamically
    return {
        "framework": str(self.framework_plugins_dir.relative_to(project_root)),
        "project": str(self.project_plugins_dir.relative_to(project_root)),
    }
```

**Benefits**:
- Portable across environments
- No git churn
- Still useful for debugging

### 🟢 Priority 3: Document State File Purpose

**Action**: Add README at `AgentQMS/.agentqms/state/README.md`:
```markdown
# AgentQMS State Files

These are **runtime-generated snapshots** for debugging.
They are NOT configuration inputs and should be gitignored.

- `plugins.yaml`: Plugin discovery snapshot
- `effective.yaml`: Resolved configuration snapshot
```

---

## Alternative Considered: Keep Tracking But Filter Changes

**Approach**: Keep files tracked but modify CI workflow:
```yaml
- name: Regenerate indexes
  run: |
    cd AgentQMS/bin
    make reindex
    # Revert environment-specific changes
    git checkout -- ../.agentqms/state/plugins.yaml
```

**Rejected Because**:
- Adds complexity
- Hides legitimate changes to these files
- Still requires manual intervention
- Band-aid solution to architectural issue

---

## Verification Steps

To confirm this assessment:

1. **Check if paths are read**: 
   ```bash
   grep -r "plugins.yaml" AgentQMS/tools/ | grep -v "# Write"
   # Result: Only write operations, no reads
   ```

2. **Compare local vs CI behavior**:
   ```bash
   # Local
   make reindex
   git diff AgentQMS/.agentqms/state/plugins.yaml
   # Shows /workspaces/...
   
   # CI (in workflow)
   # Shows /home/runner/work/...
   ```

3. **Test without state files**:
   ```bash
   rm -rf AgentQMS/.agentqms/state/
   make validate  # Still works!
   ```

---

## Conclusion

**The autofix batch job is working correctly** - it's updating artifact indexes and metadata as designed. However, it's also **unnecessarily regenerating environment-specific absolute paths** that create git churn without providing value.

**Action Required**: 
1. Gitignore `AgentQMS/.agentqms/state/` and `.agentqms/effective.yaml`
2. Consider using relative paths in future iterations
3. Document that these are runtime snapshots, not configuration

**Timeline**: Should be fixed before next autofix run to prevent accumulating more unnecessary path mutations in git history.

---

## Appendix: Commit fe10f5d Analysis

### Legitimate Changes (Keep These)
- `docs/artifacts/bug_reports/INDEX.md`: Added 2 new bugs, marked 1 complete ✅
- `docs/artifacts/*/INDEX.md`: Updated timestamps ✅

### Environmental Churn (Should Be Gitignored)
- `.agentqms/state/plugins.yaml`: 16 lines of path changes ❌
- `.agentqms/effective.yaml`: 2 lines of timestamp/path changes ❌

### Unrelated Changes
- `uv.lock`: 5430 lines (dependency updates, investigate separately)

**Verdict**: 90% of the diff is environmental churn that should not be tracked.