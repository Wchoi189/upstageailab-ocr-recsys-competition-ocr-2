# Quick Start: Your Revolutionary Architecture System

## ✅ What's Now Installed

### 1. Architecture Guardian (Pre-commit Hook)
**Status**: ✅ Installed and active
**Location**: `.pre-commit-hooks/architecture_guardian.py`
**Activation**: `pre-commit install` (run once)

**What it catches**:
- ❌ Detection code entering `ocr/core/`
- ❌ Cross-domain imports without interfaces
- ⚠️ Files marked for migration
- ⚠️ Anti-patterns (eager registration, star imports)
- ⏱️ Slow imports (>100ms)

**Test it now**:
```bash
# See current violations
python3 .pre-commit-hooks/architecture_guardian.py

# Install hook for automatic checking
pre-commit install

# From now on, every commit will be checked automatically
git commit -m "test"  # Guardian runs automatically
```

---

## 🛠️ Tools You Have (But Maybe Didn't Know)

### QMS CLI ✅ Working!
```bash
# Validate artifacts
AgentQMS/bin/qms validate --file docs/artifacts/some-plan.md

# Create new artifact
AgentQMS/bin/qms artifact create --type implementation_plan \
  --name "my-feature" --title "My Feature Plan"

# Check quality
AgentQMS/bin/qms quality check

# Monitor compliance
AgentQMS/bin/qms monitor --check
```

**Add to PATH** (optional):
```bash
export PATH="$PATH:$PWD/AgentQMS/bin"
# Then just: qms validate ...
```

### Agent Debug Toolkit (AST Analysis) ✅ Working!
```bash
# Full analysis of a directory
uv run adt full-analysis ocr/domains/detection/ > detection_analysis.txt

# Find all detection-related code
uv run adt sg-search "class.*Detection|DetectionHead" ocr/

# Dependency graph
uv run adt analyze-dependencies ocr/core/ > core_deps.json

# Hydra config tracing
uv run adt find-hydra ocr/domains/

# Component instantiation discovery
uv run adt find-instantiations ocr/

# Complexity analysis
uv run adt analyze-complexity ocr/core/validation.py

# Smart symbol search
uv run adt intelligent-search "DBHead" ocr/
```

### MCP Context System ⚠️ Needs Repair
```bash
# Current status: Handbook index missing
# Quick fix options:

# Option 1: Use task-specific context (recommended)
cd AgentQMS/bin && make context TASK="refactoring detection code"

# Option 2: Skip handbook, use direct tools
uv run adt context-tree ocr/ > architecture_tree.txt
```

---

## 🎯 Your Immediate Next Steps

### Step 1: Activate the Guardian (2 minutes)
```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# Install pre-commit hooks
pre-commit install

# Test current violations
python3 .pre-commit-hooks/architecture_guardian.py

# Commit the guardian
git add .pre-commit-hooks/ .pre-commit-config.yaml
git add analysis/architecture-migration-2026-01-21/REVOLUTIONARY_SOLUTION.md
git commit -m "feat: Add Architecture Guardian system"
```

### Step 2: Analyze Current Architecture (10 minutes)
```bash
# Full AST analysis of detection domain
uv run adt full-analysis ocr/domains/detection/ > analysis/detection_ast_analysis.txt

# Find misplaced detection code in core/
uv run adt sg-search "polygon|DetectionHead|detection_loss" ocr/core/ > analysis/core_detection_violations.txt

# Dependency graph
uv run adt analyze-dependencies ocr/ > analysis/full_dependency_graph.json

# Complexity hotspots
uv run adt analyze-complexity ocr/core/ > analysis/core_complexity.txt
```

### Step 3: Fix Top Violations (1-2 hours)
```bash
# Run guardian to see priority violations
python3 .pre-commit-hooks/architecture_guardian.py

# For each ERROR:
# 1. Use AST tools to understand dependencies
# 2. Move file or refactor imports
# 3. Test: python3 .pre-commit-hooks/architecture_guardian.py
# 4. Commit when clean
```

### Step 4: Customize for Your Goals (30 minutes)
Edit `.pre-commit-hooks/architecture_guardian.py`:

```python
# Line 31: Add your detection-specific terms
DETECTION_KEYWORDS = {
    'polygon', 'box', 'detection', 'dbnet', 'craft',
    # ADD YOUR TERMS HERE:
    'your_custom_detection_term',
}

# Line 65: Flag files you plan to move
FLAGGED_FILES = {
    'ocr/core/your_file.py': 'Move to ocr/domains/detection/your_file.py',
}
```

Test your changes:
```bash
python3 .pre-commit-hooks/architecture_guardian.py
```

---

## 🧠 AST Tool Examples for Common Tasks

### Find All Usages of a Class
```bash
# Where is DBHead used?
uv run adt intelligent-search "DBHead" ocr/

# More specific pattern
uv run adt sg-search "class $NAME(DBHead)" ocr/domains/
```

### Analyze Import Chains
```bash
# What does validation.py import?
uv run adt analyze-imports ocr/core/validation.py

# Full dependency tree
uv run adt analyze-dependencies ocr/core/validation.py --output deps.json
```

### Find Circular Dependencies
```bash
# Analyze all of core/ for circular imports
uv run adt analyze-dependencies ocr/core/ | grep -i circular
```

### Hydra Configuration Analysis
```bash
# Find all Hydra usage
uv run adt find-hydra ocr/

# Find all instantiate() calls
uv run adt find-instantiations ocr/

# Trace config merges
uv run adt trace-merges ocr/training/trainer.py
```

### Code Complexity Analysis
```bash
# Find complex functions (candidates for refactoring)
uv run adt analyze-complexity ocr/core/ | grep "complexity: [2-9][0-9]"

# Single file analysis
uv run adt analyze-complexity ocr/core/validation.py
```

---

## 📊 Monitoring Your Progress

### Daily Check
```bash
# Quick violation check
python3 .pre-commit-hooks/architecture_guardian.py

# Lines of code in core/ (goal: decrease over time)
find ocr/core/ -name "*.py" -exec wc -l {} + | tail -1
```

### Weekly Review
```bash
# Full analysis
uv run adt full-analysis ocr/ > weekly_analysis_$(date +%F).txt

# Check complexity trends
uv run adt analyze-complexity ocr/core/ > weekly_complexity_$(date +%F).txt

# Dependency graph snapshot
uv run adt analyze-dependencies ocr/ > weekly_deps_$(date +%F).json
```

### Monthly Audit
```bash
# Re-run core audit
python analysis/architecture-migration-2026-01-21/audit_core.py

# Compare with baseline
diff analysis/core_audit_report.txt analysis/core_audit_report_baseline.txt

# Review architectural assessment progress
code analysis/architecture-migration-2026-01-21/CRITICAL_ARCHITECTURE_ASSESSMENT.md
```

---

## 🚨 Troubleshooting

### "Guardian not running on commit"
```bash
# Re-install hook
pre-commit install

# Test manually
pre-commit run --all-files
```

### "QMS command not found"
```bash
# Use full path
AgentQMS/bin/qms --help

# Or add to PATH
export PATH="$PATH:$PWD/AgentQMS/bin"
```

### "Context system errors"
```bash
# Use AST tools directly instead
uv run adt context-tree ocr/ > architecture_context.txt

# Or task-specific context
cd AgentQMS/bin && make context TASK="your task here"
```

### "Import still slow after fixes"
```bash
# Profile imports
python3 -X importtime -c "import ocr.domains.detection" 2> import_profile.txt

# Analyze with AST tools
uv run adt analyze-imports ocr/domains/detection/

# Check for eager registrations
grep -r "register_" ocr/domains/ | grep -v "lazy"
```

---

## 🎉 Success Indicators

You'll know the system is working when:

1. ✅ **Commits fail with helpful messages** (guardian catching violations)
2. ✅ **No new detection code in `ocr/core/`** (boundaries enforced)
3. ✅ **Import time decreases** (lazy loading working)
4. ✅ **You can leave for weeks** and come back to clean architecture
5. ✅ **Team members follow patterns** (guardian teaches correct way)

---

## 📚 Documentation

- **Architecture Assessment**: [CRITICAL_ARCHITECTURE_ASSESSMENT.md](./CRITICAL_ARCHITECTURE_ASSESSMENT.md)
- **Revolutionary Solution**: [REVOLUTIONARY_SOLUTION.md](./REVOLUTIONARY_SOLUTION.md)
- **Migration Complete**: [MIGRATION_COMPLETE.md](./MIGRATION_COMPLETE.md)
- **Guardian Source**: [.pre-commit-hooks/architecture_guardian.py](../../.pre-commit-hooks/architecture_guardian.py)
- **AST Tools**: `uv run adt --help`
- **QMS Tools**: `AgentQMS/bin/qms --help`

---

## 💬 Remember

> "I keep failing and lose sight of goals once conversation comes to an end"

**The guardian remembers for you.** That's the revolution.

It runs on every commit. It enforces your rules. It teaches correct patterns. It persists when conversations end.

You define the goals (edit the rules). The system enforces them (automatically). Architecture stays clean (even when you sleep).

**No more 6th refactor.** 🚀
