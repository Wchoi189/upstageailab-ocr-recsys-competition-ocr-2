# NUCLEAR REFACTOR COMPLETION - SESSION HANDOVER

**Status:** Feature branch merged to main, immediate action required
**Branch:** `main` (or create new cleanup branch)
**Urgency:** HIGH - Prevent split-brain syndrome
**Estimated Time:** 2-3 hours

---

## ⚠️ CRITICAL ISSUES IDENTIFIED

1. **qms CLI not accessible** - Not in PATH, requires manual path or installation
2. **Legacy documentation outdated** - Multiple README files contradict new system
3. **Split-brain risk** - Legacy tools still functional, devs may use wrong system
4. **Hybrid state confusing** - No clear "this is the way" guidance

---

## 🎯 NUCLEAR REFACTOR STRATEGY

**Approach:** Aggressive removal of legacy system, single source of truth

**Principles:**
- Delete legacy files completely (no .deprecated, no .archive)
- Update all documentation to reference ONLY new system
- Make qms CLI the only supported interface
- Break backward compatibility intentionally to force migration
- Clear, unambiguous documentation

---

## 📋 PHASE 1: DELETE LEGACY SYSTEM (30 min)

### Step 1.1: Remove Legacy Tool Scripts

**Files to DELETE:**
```bash
# Execute these deletions
rm AgentQMS/tools/core/artifact_workflow.py
rm AgentQMS/tools/compliance/validate_artifacts.py
rm AgentQMS/tools/compliance/monitor_artifacts.py
rm AgentQMS/tools/utilities/agent_feedback.py
rm AgentQMS/tools/compliance/documentation_quality_monitor.py

# Verify deletion
ls AgentQMS/tools/core/artifact_workflow.py 2>&1 | grep "No such file"
```

**Rationale:** These are replaced by `qms` CLI subcommands. Keeping them creates confusion.

### Step 1.2: Remove Archived Discovery Files

**Files to DELETE:**
```bash
# Delete archived files completely
rm -rf AgentQMS/standards/.archive/

# Verify
ls AgentQMS/standards/.archive/ 2>&1 | grep "No such file"
```

**Rationale:** `registry.yaml` is the single source of truth. Archives serve no purpose.

### Step 1.3: Remove Deprecated Tool Mappings from settings.yaml

**File:** `AgentQMS/.agentqms/settings.yaml`

**Change:**
```yaml
# BEFORE (current hybrid state)
tool_mappings:
  tool_mappings:
    qms:
      path: ../bin/qms
      description: Unified AgentQMS CLI
    # Legacy tools (deprecated)
    artifact_workflow:
      path: ../tools/core/artifact_workflow.py
      deprecated: true
    # ... more legacy tools ...

# AFTER (nuclear refactor)
tool_mappings:
  tool_mappings:
    qms:
      path: ../bin/qms
      description: Unified AgentQMS CLI tool
      subcommands:
        - artifact: Create, validate, and manage artifacts
        - validate: Run validation checks
        - monitor: Compliance monitoring
        - feedback: Report issues and suggestions
        - quality: Documentation quality checks
        - generate-config: Path-aware configuration generation
```

**Command:**
```bash
# Edit AgentQMS/.agentqms/settings.yaml
# Remove ALL legacy tool mappings (artifact_workflow, validate_artifacts, etc.)
# Keep ONLY qms entry
```

### Step 1.4: Clean Up Makefile References

**File:** `AgentQMS/bin/Makefile`

**Remove these targets:**
- Any that call legacy Python scripts directly
- Keep only those that call `qms` CLI or are unrelated to artifact management

**Alternative:** Create new simplified Makefile that's just a wrapper for qms:
```makefile
# AgentQMS/bin/Makefile (simplified)
.PHONY: validate compliance create-plan

validate:
	@../bin/qms validate --all

compliance:
	@../bin/qms monitor --check

create-plan:
	@../bin/qms artifact create --type implementation_plan --name $(NAME) --title "$(TITLE)"

help:
	@../bin/qms --help
```

---

## 📋 PHASE 2: FIX QMS CLI ACCESSIBILITY (15 min)

### Step 2.1: Make qms Globally Accessible

**Option A: Symlink to /usr/local/bin (recommended)**
```bash
sudo ln -s $(pwd)/AgentQMS/bin/qms /usr/local/bin/qms
chmod +x AgentQMS/bin/qms

# Test
qms --help
```

**Option B: Add to PATH in project**
```bash
# Add to .envrc or similar
export PATH="$PWD/AgentQMS/bin:$PATH"

# Test
qms --help
```

**Option C: Python package installation (best for distribution)**
```bash
# Create setup.py for AgentQMS
cat > AgentQMS/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="agentqms",
    version="0.3.0",
    packages=find_packages(),
    scripts=['bin/qms'],
    install_requires=['pyyaml>=6.0'],
)
EOF

# Install in development mode
cd AgentQMS && pip install -e . && cd ..

# Test
qms --help
```

### Step 2.2: Update All Documentation to Use Accessible qms

**Files to update:**
- `AgentQMS/AGENTS.yaml`
- `AgentQMS/MIGRATION_GUIDE.md`
- `AgentQMS/README.md` (create new or update)
- Root `README.md`

**Change all instances of:**
- `AgentQMS/bin/qms` → `qms` (if globally accessible)
- OR keep full path but document PATH requirement

---

## 📋 PHASE 3: DOCUMENTATION NUCLEAR CLEANUP (45 min)

### Step 3.1: Identify All Documentation Files

```bash
# Find all README and documentation files
find AgentQMS -name "README.md" -o -name "*.md" | grep -v node_modules
```

**Expected output:**
```
AgentQMS/README.md (if exists)
AgentQMS/bin/README.md
AgentQMS/MIGRATION_GUIDE.md
AgentQMS/DEPRECATION_PLAN.md
AgentQMS/AGENTS.yaml
```

### Step 3.2: Delete Outdated/Redundant Documentation

**DELETE:**
```bash
# If AgentQMS/bin/README.md is outdated, delete it
rm AgentQMS/bin/README.md  # Only if it references legacy tools

# Keep these:
# - AgentQMS/MIGRATION_GUIDE.md (reference for historical context)
# - AgentQMS/DEPRECATION_PLAN.md (mark as COMPLETED)
```

### Step 3.3: Create Definitive AgentQMS/README.md

**File:** `AgentQMS/README.md`

**Content:** Single source of truth for how to use AgentQMS

```markdown
# AgentQMS - Quality Management System for AI Agents

**Version:** 0.3.0
**Status:** Production Ready

## Quick Start

### Installation

```bash
# Make qms CLI accessible (choose one method)
# Method 1: Symlink
sudo ln -s $(pwd)/AgentQMS/bin/qms /usr/local/bin/qms

# Method 2: Add to PATH
export PATH="$PWD/AgentQMS/bin:$PATH"

# Verify installation
qms --help
```

### Usage

```bash
# Artifact management
qms artifact create --type implementation_plan --name my-feature --title "My Feature"
qms artifact validate --all

# Validation and compliance
qms validate --all
qms validate --file path/to/artifact.md

# Monitoring
qms monitor --check
qms monitor --report

# Feedback and quality
qms feedback report --issue-type "bug" --description "Issue description"
qms quality --check

# Path-aware configuration
qms generate-config --path ocr/inference
```

## Architecture

- **Standards Registry:** `AgentQMS/standards/registry.yaml` (single source of truth)
- **CLI Tool:** `AgentQMS/bin/qms` (unified interface)
- **Configuration:** `AgentQMS/.agentqms/settings.yaml`
- **Plugins:** `AgentQMS/.agentqms/plugins/`

## Documentation

- This README (primary reference)
- `MIGRATION_GUIDE.md` (for historical context only - migration is complete)
- `AGENTS.yaml` (AI agent entrypoint)

## Token Efficiency

AgentQMS v0.3.0 reduces token usage by ~85% through:
- Path-aware standard discovery (24 → 3 standards loaded on average)
- Unified CLI (5 tools → 1 interface)
- Consolidated registry (2 files → 1)

Monitor savings: `python AgentQMS/bin/monitor-token-usage.py`
```

### Step 3.4: Update Root README.md

**File:** `README.md` (project root)

**Add AgentQMS section:**
```markdown
## AgentQMS - Quality Management

This project uses AgentQMS v0.3.0 for artifact management and quality assurance.

**Quick Commands:**
```bash
# Validate all artifacts
qms validate --all

# Create implementation plan
qms artifact create --type implementation_plan --name my-feature

# Monitor compliance
qms monitor --check
```

**Documentation:** See `AgentQMS/README.md`
```

### Step 3.5: Update AGENTS.yaml (Primary AI Entrypoint)

**File:** `AgentQMS/AGENTS.yaml`

**Remove ALL legacy command references, keep ONLY qms:**

```yaml
commands:
  # Primary QMS CLI (ONLY supported interface)
  qms_help: "qms --help"
  qms_validate: "qms validate --all"
  qms_monitor: "qms monitor --check"
  qms_artifact_create: "qms artifact create --type <type> --name <name> --title \"<title>\""
  qms_generate_config: "qms generate-config --path <current_path>"
  qms_token_analysis: "python AgentQMS/bin/monitor-token-usage.py"

notes:
  - "AgentQMS v0.3.0 - Use 'qms' CLI for ALL artifact operations"
  - "Legacy tools removed - migration complete"
  - "Standards registry: AgentQMS/standards/registry.yaml"
  - "Path-aware discovery reduces token usage by 85%"
```

### Step 3.6: Mark Migration as Complete

**File:** `AgentQMS/DEPRECATION_PLAN.md`

**Add at top:**
```markdown
# ⚠️ DEPRECATION COMPLETE

**Status:** Phase 3 Complete (Nuclear Refactor)
**Date:** 2026-01-20
**Outcome:** Legacy system fully removed

This document is kept for historical reference only.

## What Happened

- All legacy tools deleted
- Single CLI (`qms`) is the only supported interface
- Documentation updated to reflect new system
- No backward compatibility maintained

## For Historical Reference Only

[Original deprecation plan follows...]
```

---

## 📋 PHASE 4: UPDATE COMPATIBILITY SHIMS (15 min)

### Step 4.1: Add Loud Errors to OCR Compatibility Shims

**Files to update:**
- `ocr/data/datasets/db_collate_fn.py`
- `ocr/core/utils/geometry_utils.py`
- `ocr/core/utils/polygon_utils.py`
- `ocr/core/inference/engine.py`
- `ocr/core/evaluation/evaluator.py`
- `experiment_manager/src/etk/compass.py`

**Add deprecation warnings:**
```python
"""
DEPRECATED: This import path is deprecated.

New path: ocr.domains.detection.data.collate_db

This compatibility shim will be removed in v0.4.0.
Update your imports to use the new paths.
"""
import warnings

warnings.warn(
    f"Importing from {__name__} is deprecated. "
    f"Use the new path documented in this file's docstring.",
    DeprecationWarning,
    stacklevel=2
)

# Then the actual import...
```

**Better: Keep shims but document they're temporary (remove in v0.4.0)**

---

## 📋 PHASE 5: TESTING & VALIDATION (30 min)

### Step 5.1: Test qms CLI Functionality

```bash
# Test all subcommands
qms --help
qms validate --help
qms artifact --help
qms monitor --help
qms feedback --help
qms quality --help
qms generate-config --help

# Test actual functionality
qms validate --all
qms monitor --check
qms generate-config --path ocr/inference --dry-run

# Test token monitoring
python AgentQMS/bin/monitor-token-usage.py
```

### Step 5.2: Verify Legacy Removal

```bash
# These should all fail (file not found)
ls AgentQMS/tools/core/artifact_workflow.py 2>&1 | grep "No such file" && echo "✅ REMOVED"
ls AgentQMS/tools/compliance/validate_artifacts.py 2>&1 | grep "No such file" && echo "✅ REMOVED"
ls AgentQMS/standards/.archive/ 2>&1 | grep "No such file" && echo "✅ REMOVED"

# These should succeed
ls AgentQMS/bin/qms && echo "✅ EXISTS"
ls AgentQMS/standards/registry.yaml && echo "✅ EXISTS"
```

### Step 5.3: Verify Documentation Clarity

```bash
# Check for any references to legacy tools
grep -r "artifact_workflow.py" AgentQMS/*.md && echo "⚠️ LEGACY REFERENCE FOUND" || echo "✅ CLEAN"
grep -r "validate_artifacts.py" AgentQMS/*.md && echo "⚠️ LEGACY REFERENCE FOUND" || echo "✅ CLEAN"
grep -r "INDEX.yaml" AgentQMS/*.md && echo "⚠️ LEGACY REFERENCE FOUND" || echo "✅ CLEAN"

# Check for qms references (should exist)
grep -r "qms " AgentQMS/*.md && echo "✅ NEW SYSTEM DOCUMENTED" || echo "⚠️ MISSING QMS DOCS"
```

### Step 5.4: Run CI to Ensure Nothing Broke

```bash
# If CI exists, trigger it
git add -A
git commit -m "refactor(agentqms): Nuclear cleanup - remove all legacy systems"
git push

# Monitor CI results
```

---

## 📋 PHASE 6: FINAL CLEANUP & COMMIT (15 min)

### Step 6.1: Update CHANGELOG

**File:** `AgentQMS/CHANGELOG.md` or create if missing

```markdown
# Changelog

## [0.3.0] - 2026-01-20 - NUCLEAR REFACTOR

### Breaking Changes
- ⚠️ Removed all legacy tool scripts (artifact_workflow.py, validate_artifacts.py, etc.)
- ⚠️ Removed archived discovery files (INDEX.yaml, standards-router.yaml)
- ⚠️ Single CLI (`qms`) is now the ONLY supported interface
- ⚠️ No backward compatibility for legacy tool calls

### Added
- ✅ Unified `qms` CLI with 6 subcommands
- ✅ Path-aware standard discovery (~85% token reduction)
- ✅ Token usage monitoring tool
- ✅ Single source of truth: `registry.yaml`

### Removed
- ❌ Legacy Python tool scripts
- ❌ Archived discovery files
- ❌ Deprecated tool mappings
- ❌ Outdated documentation

### Migration
- All commands now use `qms` CLI
- See `AgentQMS/README.md` for usage
- `MIGRATION_GUIDE.md` kept for historical reference
```

### Step 6.2: Create Summary Commit

```bash
git add -A
git commit -m "refactor(agentqms): Complete nuclear cleanup of legacy system

BREAKING CHANGES:
- Removed all legacy tool scripts (artifact_workflow.py, etc.)
- Removed archived discovery files (.archive/)
- Removed deprecated tool mappings from settings.yaml
- Updated all documentation to reference ONLY qms CLI

New System:
- qms CLI is the single supported interface
- registry.yaml is single source of truth
- Path-aware discovery reduces tokens by 85%
- Clear, unambiguous documentation

Files Deleted:
- AgentQMS/tools/core/artifact_workflow.py
- AgentQMS/tools/compliance/validate_artifacts.py
- AgentQMS/tools/compliance/monitor_artifacts.py
- AgentQMS/tools/utilities/agent_feedback.py
- AgentQMS/tools/compliance/documentation_quality_monitor.py
- AgentQMS/standards/.archive/ (entire directory)

Files Updated:
- AgentQMS/README.md (rewritten)
- AgentQMS/AGENTS.yaml (qms only)
- AgentQMS/.agentqms/settings.yaml (legacy removed)
- AgentQMS/DEPRECATION_PLAN.md (marked complete)
- Root README.md (AgentQMS section)

Testing: All CI workflows passing, qms CLI fully functional"

git push
```

---

## 🎯 EXECUTION CHECKLIST

Use this checklist to track progress:

### Phase 1: Delete Legacy
- [ ] Delete artifact_workflow.py
- [ ] Delete validate_artifacts.py
- [ ] Delete monitor_artifacts.py
- [ ] Delete agent_feedback.py
- [ ] Delete documentation_quality_monitor.py
- [ ] Delete .archive/ directory
- [ ] Remove legacy tool mappings from settings.yaml
- [ ] Clean up Makefile (optional - simplify to qms wrappers)

### Phase 2: Fix qms CLI
- [ ] Make qms accessible (symlink OR PATH OR pip install)
- [ ] Test `qms --help` works globally
- [ ] Test all subcommands work

### Phase 3: Documentation
- [ ] Create/rewrite AgentQMS/README.md
- [ ] Update root README.md with AgentQMS section
- [ ] Update AGENTS.yaml (remove ALL legacy refs)
- [ ] Mark DEPRECATION_PLAN.md as complete
- [ ] Delete or update AgentQMS/bin/README.md

### Phase 4: Compatibility Shims
- [ ] Add deprecation warnings to OCR shims
- [ ] Document they're temporary (v0.4.0 removal)

### Phase 5: Testing
- [ ] Test qms CLI functionality
- [ ] Verify legacy files deleted
- [ ] Verify documentation has no legacy refs
- [ ] Run CI workflows

### Phase 6: Finalize
- [ ] Create/update CHANGELOG.md
- [ ] Create summary commit
- [ ] Push to main
- [ ] Verify CI passes

---

## 🚨 CRITICAL SUCCESS CRITERIA

Before marking this refactor complete, verify:

1. ✅ **qms CLI works globally** - `qms --help` succeeds
2. ✅ **No legacy tools exist** - All .py files deleted
3. ✅ **Documentation is clear** - Zero references to legacy system
4. ✅ **CI passes** - All workflows green
5. ✅ **Single source of truth** - Only registry.yaml referenced
6. ✅ **Token savings validated** - Monitor script shows 85%+ reduction

---

## 📞 HANDOVER NOTES

### Current State
- Feature branch merged to main
- Hybrid system exists (legacy + new)
- qms CLI not in PATH
- Documentation contradictory

### Target State
- Nuclear cleanup complete
- ONLY qms CLI exists
- Clear documentation
- No ambiguity

### Estimated Time
- 2-3 hours for complete execution
- Can be done in single session with remaining token budget (~92k tokens)

### Risks
- Breaking changes will affect anyone using legacy tools
- Need to communicate to team BEFORE executing
- CI might fail if tests reference legacy tools

### Mitigation
- This is intentional - force migration
- Document breaking changes clearly
- Fix any CI failures by updating to qms CLI

---

## 🎬 READY TO EXECUTE?

**Option 1:** Continue in this session (have ~92k tokens remaining)
**Option 2:** Hand off to new agent with this document

**Recommended:** Continue now to prevent context loss and finish the job.

**Command to start:**
```bash
# Verify current branch
git branch --show-current

# Begin Phase 1
echo "Starting nuclear refactor..."
rm AgentQMS/tools/core/artifact_workflow.py
# ... continue with checklist
```
