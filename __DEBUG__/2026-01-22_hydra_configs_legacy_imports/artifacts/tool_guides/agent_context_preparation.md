# Agent Context Preparation Guide

**Date**: 2026-01-23
**Purpose**: Outputs needed for AI agent context
**Python Manager**: uv

---

## Required Outputs for Agent Context

### 1. Context Tree (CRITICAL)

**Purpose**: Provides semantic directory structure with docstrings and exports

**Command**:
```bash
adt context-tree ocr/ --depth 3 --output json > context_tree.json
```

**What it provides**:
- Directory structure with annotations
- Module docstrings
- Exported symbols (`__all__`)
- Key class/function definitions
- File purposes and relationships

**Use when**: Agent needs to navigate codebase or understand module organization

---

### 2. Dependency Analysis

**Purpose**: Maps module dependencies and import relationships

**Command**:
```bash
adt analyze-dependencies ocr/ --output json --no-cycles > dependencies.json
```

**What it provides**:
- Import graph
- Module relationships
- Circular dependency detection
- Third-party vs local imports

**Use when**: Understanding module coupling or refactoring imports

---

### 3. Master Audit Report

**Purpose**: Identifies broken imports and Hydra targets

**Command**:
```bash
uv run python scripts/audit/master_audit.py > audit_report.txt
```

**What it provides**:
- Broken Python imports (with line numbers)
- Broken Hydra `_target_` paths
- Known problematic patterns
- Validation status

**Use when**: Systematic fixing of import/config issues

---

### 4. Hydra Configuration Analysis

**Purpose**: Analyzes Hydra usage patterns

**Command**:
```bash
adt find-hydra ocr/ --output json > hydra_usage.json
```

**What it provides**:
- `@hydra.main` decorators
- `hydra.utils.instantiate()` calls
- Config composition patterns
- Recursive instantiation risks

**Use when**: Debugging Hydra configuration issues

---

### 5. Component Instantiation Map

**Purpose**: Tracks where components are created

**Command**:
```bash
adt find-instantiations ocr/ --output json > instantiations.json
```

**What it provides**:
- `get_*_by_cfg()` factory calls
- Direct class instantiation
- Registry usage
- Component creation patterns

**Use when**: Understanding component lifecycle

---

### 6. Import Analysis

**Purpose**: Categorizes and validates imports

**Command**:
```bash
adt analyze-imports ocr/ --output json > imports.json
```

**What it provides**:
- Stdlib vs third-party vs local imports
- Potentially unused imports
- Import organization
- Missing dependencies

**Use when**: Cleaning up imports or resolving dependencies

---

### 7. Complexity Metrics

**Purpose**: Identifies complex code that needs refactoring

**Command**:
```bash
adt analyze-complexity ocr/ --threshold 10 --output json > complexity.json
```

**What it provides**:
- Cyclomatic complexity
- Nesting depth
- Lines of code
- Parameter counts

**Use when**: Identifying refactoring candidates

---

### 8. AgentQMS Context Bundle

**Purpose**: Provides standards and conventions

**Command**:
```bash
# Via Makefile
cd AgentQMS/bin
make context TASK="debug hydra config"

# Or direct
aqms generate-config --path ocr/
```

**What it provides**:
- Relevant standards for current task
- Artifact templates
- Tool catalog
- Compliance rules

**Use when**: Ensuring work follows project standards

---

## Complete Preparation Script

```bash
#!/bin/bash
# prepare_agent_context.sh

set -e

OUTPUT_DIR="agent_context_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "🔍 Generating agent context outputs..."

# 1. Context Tree
echo "  [1/8] Context tree..."
adt context-tree ocr/ --depth 3 --output json > "$OUTPUT_DIR/context_tree.json"

# 2. Dependencies
echo "  [2/8] Dependency analysis..."
adt analyze-dependencies ocr/ --output json --no-cycles > "$OUTPUT_DIR/dependencies.json"

# 3. Master Audit
echo "  [3/8] Master audit..."
uv run python scripts/audit/master_audit.py > "$OUTPUT_DIR/audit_report.txt"

# 4. Hydra Usage
echo "  [4/8] Hydra analysis..."
adt find-hydra ocr/ --output json > "$OUTPUT_DIR/hydra_usage.json"

# 5. Instantiations
echo "  [5/8] Component instantiations..."
adt find-instantiations ocr/ --output json > "$OUTPUT_DIR/instantiations.json"

# 6. Imports
echo "  [6/8] Import analysis..."
adt analyze-imports ocr/ --output json > "$OUTPUT_DIR/imports.json"

# 7. Complexity
echo "  [7/8] Complexity metrics..."
adt analyze-complexity ocr/ --threshold 10 --output json > "$OUTPUT_DIR/complexity.json"

# 8. AgentQMS Standards
echo "  [8/8] AgentQMS context..."
aqms generate-config --path ocr/ > "$OUTPUT_DIR/agentqms_standards.yaml"

echo "✅ Context outputs generated in: $OUTPUT_DIR"
echo ""
echo "📊 Summary:"
ls -lh "$OUTPUT_DIR"
```

---

## Priority Outputs

For most debugging sessions, these 3 are essential:

1. **Context Tree** - Navigate codebase
2. **Master Audit** - Identify broken imports/configs
3. **AgentQMS Standards** - Follow project conventions

The others are supplementary for specific tasks.

---

## ADT CLI Fix Required

**Issue**: `sg-search` pattern argument doesn't handle special characters properly

**Problem**:
```bash
# This fails
adt sg-search --pattern "isinstance($CFG, dict)" ocr/file.py

# Pattern is positional but shell breaks on special chars
adt sg-search "isinstance($CFG, dict)" --path ocr/file.py
```

**Root Cause**: Shell interprets `$CFG` as variable, `()` as subshell

**Solution**: Need to properly quote in CLI or escape in code

See `ADT_CLI_FIX.md` for implementation details.
