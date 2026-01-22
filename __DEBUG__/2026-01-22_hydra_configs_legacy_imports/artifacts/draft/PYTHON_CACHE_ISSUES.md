# Python Cache Issues - Causes and Solutions

**Problem**: Changes to Python code don't take effect immediately
**Root Cause**: Python's bytecode caching mechanism
**Impact**: Wasted debugging time, confusion about whether fixes work

---

## Why Python Caches

### The Mechanism

1. **First Import**: Python compiles `.py` → `.pyc` bytecode
2. **Cached**: Stores in `__pycache__/` directory
3. **Subsequent Imports**: Uses cached `.pyc` if source unchanged
4. **Check**: Compares file modification time (mtime)

### The Problem

**When it fails**:
- File edited but mtime not updated (rare)
- Package installed vs source edited (your case)
- Multiple Python processes
- Docker/container file sync delays
- Network filesystems (NFS, etc.)

---

## Your Specific Issue

### What Happened

1. **Editable install**: `agent-debug-toolkit` installed via `{ path = "..." }`
2. **You edited**: `agent-debug-toolkit/src/agent_debug_toolkit/cli.py`
3. **Python loaded**: Cached bytecode from previous version
4. **Result**: Changes ignored

### Why Editable Install Didn't Help

Even with `-e` (editable), Python still caches bytecode:
- Source: `agent-debug-toolkit/src/agent_debug_toolkit/cli.py`
- Cache: `agent-debug-toolkit/src/agent_debug_toolkit/__pycache__/cli.cpython-311.pyc`

When you run `uv run python -m agent_debug_toolkit.cli`, Python checks cache first.

---

## Solutions (Ranked by Effectiveness)

### Solution 1: Disable Bytecode Caching (RECOMMENDED)

**Set environment variable permanently**:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PYTHONDONTWRITEBYTECODE=1

# Apply immediately
source ~/.bashrc
```

**Pros**:
- ✅ Prevents all caching issues
- ✅ Always uses latest source
- ✅ No manual cleanup needed

**Cons**:
- ⚠️ Slightly slower imports (negligible for dev)
- ⚠️ Larger disk usage (no `.pyc` files)

**For this session only**:
```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m agent_debug_toolkit.cli sg-search --pattern "..." --path ...
```

---

### Solution 2: Clear Cache Before Running

**Manual cleanup**:
```bash
# Clear all pycache in agent-debug-toolkit
find agent-debug-toolkit -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Then run
uv run python -m agent_debug_toolkit.cli sg-search --pattern "..." --path ...
```

**Automated cleanup script**:
```bash
#!/bin/bash
# scripts/clear-cache.sh

echo "🧹 Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "✅ Cache cleared"
```

**Pros**:
- ✅ Works immediately
- ✅ No environment changes

**Cons**:
- ❌ Must remember to run
- ❌ Tedious for frequent changes

---

### Solution 3: Use Python's `-B` Flag

**Run with bytecode writing disabled**:
```bash
uv run python -B -m agent_debug_toolkit.cli sg-search --pattern "..." --path ...
```

**Pros**:
- ✅ No environment variable needed
- ✅ Works per-command

**Cons**:
- ❌ Must remember flag every time
- ❌ Doesn't clear existing cache

---

### Solution 4: Force Reimport with PYTHONPATH

**What you discovered** (works because it bypasses installed package):
```bash
PYTHONPATH=agent-debug-toolkit/src:$PYTHONPATH uv run python -m agent_debug_toolkit.cli ...
```

**Why it works**:
- Python searches `PYTHONPATH` first
- Finds source before installed package
- Uses fresh import

**Pros**:
- ✅ Guaranteed to use latest source
- ✅ No cache issues

**Cons**:
- ❌ Long command
- ❌ Easy to forget

---

### Solution 5: Reinstall Package (LEAST EFFECTIVE)

**What you tried**:
```bash
cd agent-debug-toolkit && uv pip install -e . --force-reinstall
```

**Why it doesn't help**:
- Package reinstall doesn't clear `__pycache__`
- Bytecode cache persists
- Only helps if package metadata changed

**Don't use this for code changes**

---

## Recommended Setup

### For Development (Best Practice)

**1. Add to your shell config** (`~/.bashrc` or `~/.zshrc`):
```bash
# Disable Python bytecode caching in development
export PYTHONDONTWRITEBYTECODE=1

# Optional: Add cleanup alias
alias pyclean='find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; find . -type f -name "*.pyc" -delete 2>/dev/null'
```

**2. Apply immediately**:
```bash
source ~/.bashrc
```

**3. Verify**:
```bash
echo $PYTHONDONTWRITEBYTECODE  # Should show: 1
```

**4. Test**:
```bash
# Edit agent-debug-toolkit/src/agent_debug_toolkit/cli.py
# Then run immediately:
uv run python -m agent_debug_toolkit.cli sg-search --help
# Changes should be visible
```

---

### For Production

**Keep caching enabled**:
- Faster imports
- Smaller memory footprint
- Standard Python behavior

**Only disable in development environments**

---

## Quick Fixes (Right Now)

### Option A: One-Time Cleanup + Disable Caching
```bash
# 1. Clear existing cache
find agent-debug-toolkit -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 2. Run with caching disabled
PYTHONDONTWRITEBYTECODE=1 uv run python -m agent_debug_toolkit.cli sg-search --pattern "isinstance(\$CFG, dict)" --path ocr/core/
```

### Option B: Use PYTHONPATH (Your Working Solution)
```bash
PYTHONPATH=agent-debug-toolkit/src:$PYTHONPATH uv run python -m agent_debug_toolkit.cli sg-search --pattern "isinstance(\$CFG, dict)" --path ocr/core/
```

### Option C: Create Alias (Convenience)
```bash
# Add to ~/.bashrc
alias adt-dev='PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=agent-debug-toolkit/src:$PYTHONPATH uv run python -m agent_debug_toolkit.cli'

# Then use:
adt-dev sg-search --pattern "isinstance(\$CFG, dict)" --path ocr/
```

---

## Understanding the Cache

### What Gets Cached

```
agent-debug-toolkit/
├── src/
│   └── agent_debug_toolkit/
│       ├── cli.py                    # Your source
│       └── __pycache__/
│           └── cli.cpython-311.pyc   # Cached bytecode ← THE PROBLEM
```

### Cache Invalidation Rules

Python recompiles when:
1. ✅ Source file mtime > cache file mtime
2. ✅ Python version changes
3. ✅ Magic number changes (rare)

Python **doesn't** recompile when:
1. ❌ You edit but mtime doesn't update (filesystem issue)
2. ❌ Cache exists and mtime check passes (even if content different)
3. ❌ Running from different process/terminal

---

## Prevention Strategies

### 1. Git Ignore Pycache (Already Done)

Your `.gitignore` should have:
```
__pycache__/
*.pyc
*.pyo
*.pyd
```

### 2. Pre-commit Hook (Optional)

```bash
# .git/hooks/pre-commit
#!/bin/bash
# Clear pycache before commit
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

### 3. Development Container Config

If using devcontainer, add to `.devcontainer/devcontainer.json`:
```json
{
  "containerEnv": {
    "PYTHONDONTWRITEBYTECODE": "1"
  }
}
```

---

## Testing Your Fix Now

**After setting `PYTHONDONTWRITEBYTECODE=1`**:

```bash
# 1. Clear existing cache
find agent-debug-toolkit -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 2. Test the fix
uv run python -m agent_debug_toolkit.cli sg-search --pattern "isinstance(\$CFG, dict)" --path ocr/core/utils/config_utils.py --max 3

# 3. Verify it works
# Should show 2 matches with new --pattern syntax
```

---

## Summary

| Solution                    | Effectiveness | Effort | Recommended          |
| --------------------------- | ------------- | ------ | -------------------- |
| `PYTHONDONTWRITEBYTECODE=1` | ⭐⭐⭐⭐⭐         | Low    | ✅ YES                |
| Clear cache manually        | ⭐⭐⭐           | Medium | For one-time         |
| `python -B` flag            | ⭐⭐⭐           | Medium | If can't set env var |
| PYTHONPATH override         | ⭐⭐⭐⭐          | Low    | ✅ YES (temp)         |
| Reinstall package           | ⭐             | High   | ❌ NO                 |

**Best practice**: Set `PYTHONDONTWRITEBYTECODE=1` in your development environment permanently.
