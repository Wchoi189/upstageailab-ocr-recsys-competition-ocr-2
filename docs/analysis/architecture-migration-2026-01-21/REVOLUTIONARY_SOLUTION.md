---
title: Revolutionary Solution to Architecture Drift
date: 2026-01-21
status: active
problem: "5 refactors failed - goals lost between conversations"
solution: "Persistent architectural guardrails that remember for you"
---

# Revolutionary Solution to Architecture Drift

## The Problem (Quoting You)

> "I've done this 5 times (v5.0 now)... I keep failing and lose sight of goals once conversation comes to an end"

> "Pre-commit hook failed catastrophically to detect architectural violations"

> "I need a revolutionary solution - not just another refactor"

**The Core Issue**: Human memory and vigilance fail. Conversations end. Goals fade. Code drifts.

## The Solution: A Memory System for Your Architecture

This isn't another manual process. This is **automated institutional memory**.

### 🏛️ Architecture Guardian (Installed)

**Location**: `.pre-commit-hooks/architecture_guardian.py`

**What It Does**:
1. **Blocks** detection code from entering `ocr/core/`
2. **Prevents** cross-domain imports without interfaces
3. **Warns** about files marked for migration
4. **Detects** anti-patterns (eager registration, star imports)
5. **Measures** import time performance

**How It Works**:
- Runs automatically on every `git commit`
- Uses AST analysis to understand code structure
- Customizable rules that match YOUR goals
- Fails fast with helpful suggestions

**Example Output**:
```
🚨 ARCHITECTURE GUARDIAN REPORT
❌ 2 ERRORS (must fix before commit):

  ocr/core/new_detection_logic.py:15
    Rule: CORE_PURITY
    ❌ File contains 8 detection-specific keywords
    💡 Move to ocr/domains/detection/ - core/ should only contain truly shared code

  ocr/domains/recognition/ocr_model.py:42
    Rule: DOMAIN_BOUNDARY
    ❌ Cross-domain import: ocr.domains.recognition → ocr.domains.detection
    💡 Use interfaces in ocr/core/interfaces/ for cross-domain communication

📖 For detailed guidance, see:
   analysis/architecture-migration-2026-01-21/CRITICAL_ARCHITECTURE_ASSESSMENT.md
```

### 📊 AST Tools You Already Have

**agent-debug-toolkit** (`uv run adt ...`):
- `adt ast dump <file>` - See Python AST structure
- `adt sg search <pattern> <path>` - Semantic code search
- `adt dependencies <path>` - Analyze dependency graphs
- `adt hydra trace <config>` - Track config merges

**MCP Unified Project** (`mcp_unified_proje_adt_meta_query`):
- Kind: `dependency_graph` - Full project dependency analysis
- Kind: `symbol_search` - Find all usages of classes/functions
- Kind: `config_flow` - Track Hydra config precedence
- Kind: `component_instantiations` - Find all Hydra instantiations

**Python Built-ins**:
```python
import ast  # Parse and analyze Python code
import jedi  # Code intelligence and refactoring
import rope  # Advanced refactoring operations
```

### 🎯 Systematic Implementation Plan

#### Phase 1: Activate Guardian (5 minutes)
```bash
# 1. Make guardian executable
chmod +x .pre-commit-hooks/architecture_guardian.py

# 2. Install pre-commit hook
pre-commit install

# 3. Test it (this will show you current violations)
python3 .pre-commit-hooks/architecture_guardian.py

# 4. Commit the guardian itself
git add .pre-commit-hooks/architecture_guardian.py .pre-commit-config.yaml
git commit -m "feat: Add Architecture Guardian to prevent drift"
```

#### Phase 2: Fix Current Violations (1-2 hours)

Run the guardian to see what's wrong:
```bash
python3 .pre-commit-hooks/architecture_guardian.py
```

For each violation:
1. **CORE_PURITY errors**: Move detection code from `ocr/core/` to `ocr/domains/detection/`
2. **DOMAIN_BOUNDARY errors**: Add interface in `ocr/core/interfaces/`, use it
3. **MIGRATION_PENDING warnings**: Schedule these for next sprint

Use AST tools to help:
```bash
# Find all detection-related code in core/
uv run adt sg search "class.*Detection|def.*polygon|DetectionHead" ocr/core/

# Check dependencies before moving
uv run adt dependencies ocr/core/validation.py
```

#### Phase 3: Customize Rules (30 minutes)

Edit `.pre-commit-hooks/architecture_guardian.py`:

```python
# Add your specific keywords
DETECTION_KEYWORDS = {
    'polygon', 'box', 'detection',  # existing
    'your_special_detection_term',  # add yours
}

# Define your domain rules
ALLOWED_IMPORTS = {
    'ocr.domains.your_new_domain': {'ocr.core', 'ocr.domains.your_new_domain'},
}

# Flag files for migration
FLAGGED_FILES = {
    'ocr/core/your_misplaced_file.py': 'Move to ocr/domains/target/your_file.py',
}
```

Commit your customizations:
```bash
git add .pre-commit-hooks/architecture_guardian.py
git commit -m "chore: Customize architectural rules"
```

#### Phase 4: Document Goals (1 hour)

Update `CRITICAL_ARCHITECTURE_ASSESSMENT.md` with:
- Your current architecture vision
- What "good" looks like
- Specific refactor phases
- Timeline and priorities

**Why**: When you come back in 3 months, the guardian remembers the rules, but YOU need to remember WHY.

### 🔄 Context Bundle System Status

**Good News**: Context bundles ARE working! Evidence:

1. You have `context/utility-scripts/utility-scripts-index.yaml`
2. Commands available: `make context TASK="..."`
3. MCP servers active: project_compass, agent_debug_toolkit

**Test It**:
```bash
# List available bundles
cd AgentQMS/bin && make context-list

# Load specific context
make context TASK="refactoring detection code"
```

**Verify MCP**:
```bash
# Check MCP servers are running
ps aux | grep mcp

# Test agent-debug-toolkit
uv run adt --help
```

### 🛠️ QMS CLI Status

**Issue**: `qms` command not in PATH

**Quick Fix**:
```bash
# Check where it's installed
find . -name "qms" -o -name "qms.py" 2>/dev/null

# Use full path for now
python AgentQMS/tools/cli.py validate

# OR add to PATH
export PATH="$PATH:$PWD/AgentQMS/bin"
```

**Permanent Fix**:
```bash
# Install as editable package
cd AgentQMS
uv pip install -e .

# Test
qms --help
```

### 📋 Maintenance Checklist

**Daily**:
- [ ] Commit triggers guardian automatically ✅
- [ ] Fix violations before they compound ✅

**Weekly**:
- [ ] Review guardian violations: `python3 .pre-commit-hooks/architecture_guardian.py`
- [ ] Update `FLAGGED_FILES` as you plan migrations

**Monthly**:
- [ ] Audit `ocr/core/` size: `du -sh ocr/core/`
- [ ] Review `CRITICAL_ARCHITECTURE_ASSESSMENT.md` progress
- [ ] Update architectural rules in guardian

**Per Refactor**:
- [ ] Update guardian rules FIRST (define boundaries)
- [ ] Let guardian prevent violations during work
- [ ] Document changes in architecture assessment
- [ ] Update flagged files list

### 🎯 Success Metrics

You'll know the revolutionary solution is working when:

1. **Commits fail fast** with helpful messages (not silent drift)
2. **Architecture stays clean** between conversations (guardian remembers)
3. **New team members** understand boundaries (violations teach correct patterns)
4. **Refactors finish** because violations are caught early
5. **You can leave** for months and come back to maintained architecture

### 🚀 Next Steps

1. **Right Now** (5 min):
   ```bash
   chmod +x .pre-commit-hooks/architecture_guardian.py
   pre-commit install
   python3 .pre-commit-hooks/architecture_guardian.py  # See current state
   ```

2. **This Session** (2 hours):
   - Fix current violations (start with ERRORS)
   - Commit with guardian active
   - Customize rules for your specific goals

3. **This Week**:
   - Move detection code from `ocr/core/` (Phase 1 of assessment)
   - Add lazy registry to fix import performance
   - Document architecture vision

4. **This Month**:
   - Complete Phases 2-4 of `CRITICAL_ARCHITECTURE_ASSESSMENT.md`
   - Train team on guardian rules
   - Celebrate no longer losing progress! 🎉

### 💡 Key Insight

**The guardian doesn't replace you - it remembers FOR you.**

You define the goals (in the rules). The guardian enforces them (at every commit). When conversations end, when months pass, when teams change - **the rules persist**.

This is how you break the cycle. Not by trying harder. By building systems that don't forget.

---

## Tools Reference

### AST Analysis Commands

```bash
# Dependency graph for entire project
uv run adt dependencies ocr/ --output graph.json

# Find all polygon-related code
uv run adt sg search "polygon" ocr/

# Check Hydra config flow
uv run adt hydra trace configs/train/detection.yaml

# AST dump for complex file
uv run adt ast dump ocr/core/validation.py > validation_ast.txt
```

### MCP Tools via Python

```python
from agent_debug_toolkit import analyze_dependencies

# Get full dependency graph
deps = analyze_dependencies('ocr/domains/detection/')

# Find circular imports
circular = [d for d in deps if d.is_circular]

# Export for visualization
deps.to_graphviz('deps.dot')
```

### Guardian Customization Examples

```python
# Detect performance regressions
MAX_FILE_SIZE = 500  # lines
if len(content.split('\n')) > MAX_FILE_SIZE:
    violations.append(...)

# Enforce naming conventions
if filepath.name.startswith('temp_'):
    violations.append(Violation(
        message="Temporary files in production code",
        severity="ERROR"
    ))

# Check for TODO/FIXME accumulation
todo_count = content.count('TODO')
if todo_count > 5:
    violations.append(...)
```

---

**Remember**: This system works even when you sleep, when conversations end, when you forget. That's the revolution. 🚀
