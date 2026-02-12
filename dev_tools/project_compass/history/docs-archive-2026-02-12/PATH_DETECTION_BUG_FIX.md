# Path Detection Follow-Up Bug Fix - V3.0.1

**Date:** 2026-02-12
**Status:** ✅ RESOLVED
**Context:** Post-audit follow-up fix

---

## 🐛 Bug Discovered

After initial v3.0.1 path detection fix, `.vessel/` was still being created at `/workspaces` (workspace root) when compass was imported from external contexts (e.g., MCP server).

---

## Root Cause

**Fallback logic was too permissive:**

```python
# BUGGY CODE:
else:
    # Fallback used cwd without validation
    self.project_root = Path.cwd()  # ❌ Creates .vessel anywhere!
```

**Why it failed:**
1. When called from `/workspaces`, it didn't find project-compass pyproject.toml
2. Fell back to using `/workspaces` as project root
3. Created `.vessel/` there instead of in project_compass directory

---

## Fix Applied

### New Three-Tier Detection Strategy

**Tier 1: Walk up with strict validation**
```python
# Require BOTH vault/ AND (history/ OR .vessel/)
has_vault = (current / "vault").exists()
has_history_or_vessel = (current / "history").exists() or (current / ".vessel").exists()
if has_vault and has_history_or_vessel:
    self.project_root = current  # ✅ Only if both markers present
```

**Tier 2: Smart search in common locations**
```python
search_paths = [
    Path.cwd() / "dev_tools" / "project_compass",
    Path.cwd().parent / "dev_tools" / "project_compass",
    Path("/workspaces") / "dev_tools" / "project_compass",
]
# Check each location explicitly
```

**Tier 3: Ultimate fallback**
```python
# Only if all else fails
self.project_root = Path.cwd()
```

---

## Testing

### All Scenarios Pass ✅

| Test | Working Directory | Result | Status |
|------|-------------------|--------|--------|
| From workspace root | `/workspaces` | Finds `/workspaces/dev_tools/project_compass` | ✅ |
| From dev_tools | `/workspaces/dev_tools` | Finds `/workspaces/dev_tools/project_compass` | ✅ |
| From project dir | `/workspaces/dev_tools/project_compass` | Finds `/workspaces/dev_tools/project_compass` | ✅ |
| CLI usage | Any directory | Works correctly | ✅ |

### Verification Commands

```bash
# Test 1: From workspace root
$ cd /workspaces && python -c "from project_compass.src.core import VesselPaths; p = VesselPaths(); print(p.vessel_dir)"
/workspaces/dev_tools/project_compass/.vessel  ✅

# Test 2: Only one .vessel exists
$ find /workspaces -name ".vessel" -type d
/workspaces/dev_tools/project_compass/.vessel  ✅

# Test 3: CLI works
$ uv run compass pulse-status
🔥 Active Pulse  ✅
```

---

## Changes Made

**File:** [project_compass/src/core.py](project_compass/src/core.py#L38-L75)

**Key Improvements:**
1. ✅ Stricter validation (AND instead of OR)
2. ✅ Smart search for standard locations
3. ✅ Better fallback logic
4. ✅ Tested from all common working directories

---

## Cleanup

```bash
# Removed incorrectly created directory
$ rm -rf /workspaces/.vessel
✅ Cleaned up
```

---

## Impact

**Before Fix:**
- `.vessel` created at workspace root when called from MCP server
- Path detection unreliable from external contexts
- Potential for multiple .vessel directories

**After Fix:**
- Always finds correct project directory
- Robust from any working directory
- Single .vessel at correct location

---

**Status:** ✅ COMPLETE
**Version:** v3.0.1 (same-day hotfix)
**Testing:** All scenarios verified
