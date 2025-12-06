# Qwen Coder + AgentQMS Integration

**Purpose:** Consolidate Qwen Coder with AgentQMS for intelligent artifact management and quality assurance.

---

## 🎯 Quick Start

### Frontmatter Fixes (Most Common)
```bash
# Fix batch 2 artifacts (20 files)
python .qwen/consolidate.py --batch 2

# Fix specific files
python .qwen/consolidate.py --files docs/artifacts/assessments/file1.md docs/artifacts/assessments/file2.md

# Preview without modifying
python .qwen/consolidate.py --batch 2 --dry-run

# Validate only
python .qwen/consolidate.py --batch 2 --validate-only
```

### Artifact Validation
```bash
# Validate all artifacts
cd AgentQMS/interface && make validate

# Check compliance
cd AgentQMS/interface && make compliance

# Both together
cd AgentQMS/interface && make validate && make compliance
```

---

## 📚 Available Tools

### 1. **Consolidate Script** (`.qwen/consolidate.py`)
**Purpose:** Batch fix artifacts using AgentQMS tools.

**What it does:**
- Detects missing or broken frontmatter
- Generates correct frontmatter using `FrontmatterGenerator`
- Validates results using `ArtifactValidator`
- Reports pass/fail status

**Usage:**
```bash
python .qwen/consolidate.py --batch N               # Fix batch N
python .qwen/consolidate.py --files <paths>        # Fix specific files
python .qwen/consolidate.py --all                   # Fix all artifacts
python .qwen/consolidate.py --dry-run               # Preview changes
python .qwen/consolidate.py --validate-only         # Just check (no fixes)
```

**Exit codes:**
- `0`: All files valid
- `1`: Invalid files or errors

---

### 2. **AgentQMS Frontmatter Generator**
**Location:** `AgentQMS/toolkit/maintenance/add_frontmatter.py`

**What it does:**
- Analyzes file paths and content
- Detects artifact type from directory structure
- Generates complete frontmatter (title, date, type, category, status, tags)
- Adds frontmatter to files missing it

**Used by:** `consolidate.py` (no need to call directly)

---

### 3. **AgentQMS Artifact Validator**
**Location:** `AgentQMS/agent_tools/compliance/validate_artifacts.py`

**What it does:**
- Checks naming conventions (YYYY-MM-DD_HHMM_type_name.md)
- Validates frontmatter structure and required fields
- Verifies date format (YYYY-MM-DD HH:MM (KST))
- Checks valid artifact types and categories
- Reports detailed validation errors

**Command line:**
```bash
cd AgentQMS/interface
make validate        # Validate all artifacts
make validate --all  # Same as above
```

---

## 🔄 Standard Workflows

### Workflow 1: Fix Batch Files

```bash
# 1. Identify violations
cd AgentQMS/interface && make validate

# 2. Fix the batch
python .qwen/consolidate.py --batch 2

# 3. Verify all are valid
cd AgentQMS/interface && make validate

# 4. Commit fixes
git add docs/artifacts/
git commit -m "AgentQMS Phase 5: batch 2 - frontmatter fixes via consolidated tooling"
```

### Workflow 2: Manual Artifact Creation (When Needed)

```bash
# Use AgentQMS Makefile commands (NOT qwen)
cd AgentQMS/interface

# Create implementation plan
make create-plan NAME=my-plan TITLE="My Plan Title"

# Create assessment
make create-assessment NAME=my-assessment TITLE="Assessment Title"

# Create bug report
make create-bug-report NAME=my-bug TITLE="Bug Description"

# Validate new artifact
make validate
```

### Workflow 3: Qwen Interactive Mode (Requires Enhancement)

```bash
# Currently not recommended - Qwen execution blocked
# In future: ./.qwen/run.sh interactive "<task>"

# For now, use AgentQMS tools directly or ask Copilot to implement fixes
```

---

## 🏗️ Architecture

### Files in This Directory

| File | Purpose | Status |
|------|---------|--------|
| `consolidate.py` | Wrapper for batch frontmatter fixes | ✅ Active |
| `run.sh` | Old script (echo-only) | ⚠️ Legacy |
| `run_improved.sh` | Improved Qwen execution script | 🔧 Not yet active |
| `settings.json` | Qwen config (validation enabled) | ✅ Enabled |
| `QWEN.md` | Qwen documentation (reference) | 📖 Reference |
| `prompts.md` | Pre-built prompts | 📖 Reference |

### Removed (Duplicates - DO NOT USE)

- ~~`fix_frontmatter.py`~~ - Use `consolidate.py` instead
- ~~`fix_batch1_batch2.py`~~ - Use `consolidate.py` instead
- ~~`final_batch_fix.py`~~ - Use `consolidate.py` instead

---

## 📋 Batch Mappings

### Batch 1 (Completed)
10 assessment artifacts (2024-11-20_1430 through 2024-11-20_1730)
- **Status:** ✅ Fixed and validated
- **Compliance:** 100% pass

### Batch 2 (In Progress)
20 artifacts: 8 assessments + 12 bug reports (2024-11-20_1800 through 2024-11-20_2420)
- **Status:** ⏳ Renamed, frontmatter needs fix
- **Command:** `python .qwen/consolidate.py --batch 2`

### Batches 3-6 (Pending)
~70 more violations
- **Status:** ❌ Not started
- **Strategy:** Use `consolidate.py` with same workflow

---

## 🔍 Troubleshooting

### Issue: `consolidate.py` ImportError

**Problem:** `ModuleNotFoundError: No module named 'toolkit.maintenance.add_frontmatter'`

**Solution:**
1. Ensure you're running from project root: `cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2`
2. Check AgentQMS exists: `ls -la AgentQMS/`
3. Run with full path: `python /workspaces/upstageailab-ocr-recsys-competition-ocr-2/.qwen/consolidate.py --batch 2`

### Issue: Validation Still Failing After Fix

**Problem:** Files show as invalid after running `consolidate.py`

**Cause:**
- Date format: Must be `YYYY-MM-DD HH:MM (KST)`, not ISO format
- Missing required fields: Check that all required fields are in frontmatter
- Invalid status/category: Must match AgentQMS allowed values

**Solution:**
1. Run with `--validate-only` to see detailed errors: `python .qwen/consolidate.py --batch 2 --validate-only`
2. Read error messages carefully (they specify what's wrong)
3. Manually check file if needed: `head -30 docs/artifacts/assessments/filename.md`

### Issue: File Not Found

**Problem:** `⏭️  SKIP: filename (not found)`

**Cause:** File path incorrect or batch mapping outdated

**Solution:**
1. Check file exists: `ls docs/artifacts/assessments/ | grep filename`
2. Update batch mapping in `consolidate.py` if needed

---

## 🎓 Key Concepts

### Frontmatter Structure
All artifacts require YAML frontmatter:
```yaml
---
title: "Artifact Title"
date: "YYYY-MM-DD HH:MM (KST)"
type: "assessment"          # or: implementation_plan, design, research, bug_report, etc.
category: "evaluation"      # or: planning, architecture, research, troubleshooting, etc.
status: "active"            # or: draft, inactive, archived
version: "1.0"
tags: ["tag1", "tag2"]
branch: "main"
---
```

### Artifact Types
- `implementation_plan` - Feature plans, milestones
- `assessment` - Analysis, reviews, validation results
- `design` - Architecture, design decisions
- `research` - Investigation results, analysis
- `bug_report` - Bug descriptions, defect tracking
- `template` - Reusable templates for other artifacts

### Status Values
- `active` - Currently in use / relevant
- `draft` - Work in progress
- `inactive` - Archived / deprecated
- `completed` - Task completed

---

## 📞 When to Use What

| Need | Tool | Command |
|------|------|---------|
| Fix batch of files | `consolidate.py` | `python .qwen/consolidate.py --batch N` |
| Check all valid | AgentQMS validate | `cd AgentQMS/interface && make validate` |
| Create new artifact | AgentQMS Makefile | `cd AgentQMS/interface && make create-*` |
| Interactive Qwen | Not recommended | Use Copilot/manual approach |
| Preview fixes | `consolidate.py` | `python .qwen/consolidate.py --batch N --dry-run` |

---

## ✅ Quality Standards

Before merging:
```bash
# 1. Fix artifacts
python .qwen/consolidate.py --batch N

# 2. Validate all
cd AgentQMS/interface && make validate && make compliance

# 3. Commit
git add docs/artifacts/
git commit -m "AgentQMS Phase 5: batch N - frontmatter fixes via consolidated tooling"
```

**No manual .md edits** - Always use tooling for consistency.

---

## 🔗 References

- **AgentQMS SST:** `AgentQMS/knowledge/agent/system.md`
- **Compliance Rules:** `AgentQMS/knowledge/agent/README.md`
- **Architecture:** `.agentqms/state/architecture.yaml`
- **Tool Catalog:** `.copilot/context/tool-catalog.md`
- **Workflow Triggers:** `.copilot/context/workflow-triggers.yaml`
