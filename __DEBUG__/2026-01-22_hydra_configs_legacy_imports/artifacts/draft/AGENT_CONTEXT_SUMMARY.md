# Agent Context Outputs Summary

**Date**: 2026-01-23
**Purpose**: Required outputs for AI agent debugging context
**Status**: ✅ Master audit complete, ADT CLI fix documented

---

## Executive Summary

For effective AI agent debugging, you need **8 types of context outputs**. The most critical are:

1. **Context Tree** - Navigate codebase structure
2. **Master Audit** - Identify broken imports/configs
3. **AgentQMS Standards** - Follow project conventions

---

## Critical Outputs (Priority 1)

### 1. Master Audit Report ✅ COMPLETE

**Generated**: `/tmp/master_audit_output.txt`

**Command**:
```bash
uv run python scripts/audit/master_audit.py > audit_report.txt
```

**What it provides**:
- Broken Python imports (53 total)
- Broken Hydra targets (13 total)
- File paths and line numbers
- Error descriptions

**Status**: ✅ Generated successfully

---

### 2. Context Tree (Requires ADT with typer)

**Command**:
```bash
# After installing typer in ADT venv
source agent-debug-toolkit/.venv/bin/activate
pip install typer
adt context-tree ocr/ --depth 3 --output json > context_tree.json
```

**What it provides**:
- Directory structure with annotations
- Module docstrings
- Exported symbols
- Key definitions

**Status**: ⏳ Requires `typer` installation in ADT venv

---

### 3. AgentQMS Standards

**Command**:
```bash
aqms generate-config --path ocr/ > agentqms_standards.yaml
```

**What it provides**:
- Relevant standards for current path
- Artifact templates
- Tool catalog
- Compliance rules

**Status**: ⏳ Ready to generate

---

## Supplementary Outputs (Priority 2)

### 4. Dependency Analysis

**Command**:
```bash
adt analyze-dependencies ocr/ --output json > dependencies.json
```

**Provides**: Import graph, circular dependencies

### 5. Hydra Usage Analysis

**Command**:
```bash
adt find-hydra ocr/ --output json > hydra_usage.json
```

**Provides**: Hydra patterns, instantiate calls, recursive risks

### 6. Component Instantiations

**Command**:
```bash
adt find-instantiations ocr/ --output json > instantiations.json
```

**Provides**: Factory calls, component creation patterns

### 7. Import Analysis

**Command**:
```bash
adt analyze-imports ocr/ --output json > imports.json
```

**Provides**: Import categorization, unused imports

### 8. Complexity Metrics

**Command**:
```bash
adt analyze-complexity ocr/ --threshold 10 --output json > complexity.json
```

**Provides**: Cyclomatic complexity, refactoring candidates

---

## Quick Start Script

```bash
#!/bin/bash
# generate_agent_context.sh

OUTPUT_DIR="agent_context_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "🔍 Generating agent context outputs..."

# 1. Master Audit (CRITICAL - works now)
echo "  [1/3] Master audit..."
uv run python scripts/audit/master_audit.py > "$OUTPUT_DIR/audit_report.txt"

# 2. AgentQMS Standards (CRITICAL - works now)
echo "  [2/3] AgentQMS standards..."
aqms generate-config --path ocr/ > "$OUTPUT_DIR/agentqms_standards.yaml"

# 3. Context Tree (requires ADT with typer)
echo "  [3/3] Context tree..."
if command -v adt &> /dev/null; then
    adt context-tree ocr/ --depth 3 --output json > "$OUTPUT_DIR/context_tree.json"
else
    echo "  ⚠️  ADT not available - install typer in agent-debug-toolkit venv"
fi

echo "✅ Context outputs generated in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
```

---

## ADT Setup Required

**Issue**: ADT CLI requires `typer` but it's not installed

**Fix**:
```bash
cd agent-debug-toolkit
source .venv/bin/activate
pip install typer
# or
pip install agent-debug-toolkit[cli]
```

**After fix**, all ADT commands will work:
- `adt context-tree`
- `adt analyze-dependencies`
- `adt find-hydra`
- `adt sg-search` (with pattern fix)

---

## ADT CLI Fix Applied

**Issue**: `sg-search` pattern argument doesn't handle special characters

**Solution**: Changed pattern from positional to named option

**Before**:
```bash
adt sg-search "isinstance($CFG, dict)" --path ocr/  # Fails
```

**After** (with fix):
```bash
adt sg-search --pattern "isinstance($CFG, dict)" --path ocr/  # Works
```

**Documentation**: See `adt_cli_fix_sg_search.md` for implementation details

---

## Current Status

| Output             | Status        | Location                       |
| ------------------ | ------------- | ------------------------------ |
| Master Audit       | ✅ Complete    | `/tmp/master_audit_output.txt` |
| AgentQMS Standards | ⏳ Ready       | Run `aqms generate-config`     |
| Context Tree       | ⏳ Needs typer | Install in ADT venv            |
| Dependencies       | ⏳ Needs typer | Install in ADT venv            |
| Hydra Usage        | ⏳ Needs typer | Install in ADT venv            |
| Instantiations     | ⏳ Needs typer | Install in ADT venv            |
| Imports            | ⏳ Needs typer | Install in ADT venv            |
| Complexity         | ⏳ Needs typer | Install in ADT venv            |

---

## Next Steps

1. **Install typer in ADT**:
   ```bash
   cd agent-debug-toolkit
   source .venv/bin/activate
   pip install typer
   ```

2. **Generate all outputs**:
   ```bash
   bash generate_agent_context.sh
   ```

3. **Apply ADT CLI fix** (optional but recommended):
   - Update `agent-debug-toolkit/src/agent_debug_toolkit/cli.py`
   - Change `sg-search` pattern to named option
   - See `adt_cli_fix_sg_search.md` for details

---

## Files Created

1. `agent_context_preparation.md` - Complete guide with all commands
2. `adt_cli_fix_sg_search.md` - Fix for pattern argument handling
3. This summary document

All located in: `__DEBUG__/2026-01-22_hydra_configs_legacy_imports/artifacts/tool_guides/`
