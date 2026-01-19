# Configuration Standards Compliance Audit Guide

## Overview

This guide provides a systematic approach to audit the `ocr/` module for compliance with [configuration-standards.yaml](../../AgentQMS/standards/tier2-framework/configuration-standards.yaml).

**Last Updated:** January 20, 2026
**Module:** `ocr/`
**Standard:** `AgentQMS/standards/tier2-framework/configuration-standards.yaml`

---

## Quick Start

### 1. Run the Automated Audit

```bash
python scripts/audit_config_compliance.py
```

This generates a report at `docs/reports/config_compliance_audit.txt`

### 2. Use AST Analyzer Tools (MCP or CLI)

#### Via MCP (in VS Code with Copilot):
```
Use the unified_proje_adt_meta_query tool with:
- kind: "config_access"
- target: "ocr/"
```

#### Via CLI:
```bash
uv run adt analyze-config ocr/ > ocr_config_analysis.json
```

---

## Critical Compliance Rules

### ❌ Rule 1: `no-isinstance-dict` (CRITICAL)

**Problem:** `isinstance(cfg, dict)` returns `False` for OmegaConf's `DictConfig`, causing silent failures.

**Bad:**
```python
if isinstance(cfg, dict):
    process(cfg)  # Never runs for DictConfig!
```

**Good:**
```python
from ocr.core.utils.config_utils import is_config

if is_config(cfg):
    process(cfg)  # Works for both dict and DictConfig
```

### ❌ Rule 2: `use-ensure-dict` (CRITICAL)

**Problem:** `dict(cfg)` or `OmegaConf.to_container(cfg)` can have inconsistent behavior with variable interpolation.

**Bad:**
```python
native_dict = dict(cfg)  # May not resolve ${...} variables
native_dict = OmegaConf.to_container(cfg)  # Inconsistent API
```

**Good:**
```python
from ocr.core.utils.config_utils import ensure_dict

native_dict = ensure_dict(cfg)  # Handles resolution properly
```

---

## Audit Tools & Existing Outputs

### Your Pre-Existing Scan Outputs

#### ✅ `project_compass/config_access.txt` - **HIGHLY VALUABLE**

This 14,478-line JSON file contains:
- Every `.get()` call on config objects
- File path, line number, column
- Code context and snippet
- Categorization by access type

**How to Use:**
```bash
# Find isinstance violations
grep "isinstance.*dict" project_compass/config_access.txt

# Find to_container usage
grep "to_container" project_compass/config_access.txt

# Extract specific file's violations
jq '.results[] | select(.file | contains("ocr/data/"))' project_compass/config_access.txt
```

#### ❌ `project_compass/config_deps.txt` - Empty
Re-run the dependency analyzer to populate this.

#### ℹ️ `project_compass/context_tree.txt` & `dependency_graph.txt`
These provide structural context but aren't directly used for rule checking.

---

## Systematic Audit Workflow

### Phase 1: Automated Scan (5 minutes)

1. **Run the compliance audit:**
   ```bash
   python scripts/audit_config_compliance.py
   ```

2. **Review the report:**
   ```bash
   cat docs/reports/config_compliance_audit.txt
   ```

3. **Current findings:**
   - **8 violations** of `no-isinstance-dict`
   - **2 violations** of `use-ensure-dict`
   - **Priority files:**
     - `ocr/data/lightning_data.py` (6 violations)
     - `ocr/core/utils/config_utils.py` (1 violation - ironic!)
     - `ocr/command_builder/overrides.py` (1 violation)

### Phase 2: Deep Analysis with AST Tools (10-15 minutes)

#### Using MCP Tools (Recommended)

**Available via `mcp_unified_proje_adt_meta_query`:**

```yaml
# Find all config accesses
kind: config_access
target: ocr/
options:
  output: json
```

```yaml
# Find Hydra instantiate() patterns
kind: hydra_usage
target: ocr/
```

```yaml
# Trace config merge precedence
kind: merge_order
target: ocr/core/lightning/
```

```yaml
# Build dependency graph
kind: dependency_graph
target: ocr/
options:
  output: json
```

#### Using CLI (`adt`)

**1. Find all config accesses:**
```bash
uv run adt analyze-config ocr/ > ocr_config_access_full.json
```

**2. Find Hydra usage patterns:**
```bash
uv run adt find-hydra ocr/
```

**3. Trace merge precedence issues:**
```bash
uv run adt trace-merges ocr/core/lightning/lightning_module.py
```

**4. Generate context tree for navigation:**
```bash
uv run adt context-tree ocr/ --depth 3 --output markdown > ocr_tree.md
```

**5. Search for specific symbols:**
```bash
# Find all uses of ensure_dict
uv run adt intelligent-search "ensure_dict" --root ocr/

# Find DictConfig imports
uv run adt intelligent-search "DictConfig"
```

### Phase 3: Manual Review (30-60 minutes)

**Priority files to review:**

1. **`ocr/data/lightning_data.py`** (6 violations)
   - Lines around 31-32
   - Redundant `isinstance(x, dict) or OmegaConf.is_dict(x)` checks

2. **`ocr/core/utils/config_utils.py`** (1 violation)
   - Line 35: Inside `ensure_dict()` itself
   - This is acceptable (internal implementation), but could use `is_config()`

3. **`ocr/command_builder/overrides.py`** (1 violation)
   - Line 19: Profile handling

4. **`ocr/data/datasets/__init__.py`** (2 violations)
   - Line 57: Using `OmegaConf.to_container()` directly

### Phase 4: Validation (5 minutes)

After fixes, re-run:

```bash
# Re-scan
python scripts/audit_config_compliance.py

# Run tests to ensure nothing broke
pytest tests/core/utils/test_config_utils.py -v

# Validate with AgentQMS
cd AgentQMS/bin && make validate
```

---

## MCP Tool Reference

### Available Tools

From `mcp_unified_proje_adt_meta_query`:

| Kind | Purpose | Example Target |
|------|---------|----------------|
| `config_access` | Find cfg.X, config['X'] patterns | `ocr/` |
| `merge_order` | Debug OmegaConf.merge() precedence | `ocr/core/lightning/` |
| `hydra_usage` | Find @hydra.main, instantiate() | `ocr/` |
| `component_instantiations` | Track factory patterns | `ocr/core/models/` |
| `dependency_graph` | Build import/call graph | `ocr/` |
| `imports` | Analyze import structure | `ocr/` |
| `complexity` | Check cyclomatic complexity | `ocr/` |
| `context_tree` | Semantic directory tree | `ocr/` |
| `symbol_search` | Find symbols by name/path | `"ensure_dict"` |

### Example MCP Queries

**Check all config accesses in recognition module:**
```python
mcp_unified_proje_adt_meta_query(
    kind="config_access",
    target="ocr/recognition/",
    options={"output": "json"}
)
```

**Find all Hydra instantiate calls:**
```python
mcp_unified_proje_adt_meta_query(
    kind="hydra_usage",
    target="ocr/",
    options={}
)
```

**Build dependency graph:**
```python
mcp_unified_proje_adt_meta_query(
    kind="dependency_graph",
    target="ocr/core/",
    options={"output": "json"}
)
```

---

## Common Violations & Fixes

### Violation 1: `isinstance(cfg, dict)`

**Location:** `ocr/data/lightning_data.py:32`

**Current Code:**
```python
if isinstance(self.config, dict) or OmegaConf.is_dict(self.config):
    self.collate_cfg = self.config.get("collate_fn")
```

**Fixed Code:**
```python
from ocr.core.utils.config_utils import is_config

if is_config(self.config):
    self.collate_cfg = self.config.get("collate_fn")
```

### Violation 2: `OmegaConf.to_container()`

**Location:** `ocr/data/datasets/__init__.py:57`

**Current Code:**
```python
predict_config = OmegaConf.create(
    OmegaConf.to_container(datasets_config.predict_dataset, resolve=True)
)
```

**Fixed Code:**
```python
from ocr.core.utils.config_utils import ensure_dict

predict_config = OmegaConf.create(
    ensure_dict(datasets_config.predict_dataset, resolve=True)
)
```

---

## Integration with AgentQMS

### Compliance Checking

```bash
cd AgentQMS/bin && make compliance
```

This runs the full AgentQMS compliance suite, which includes configuration standards.

### Auto-Fix (if available)

```bash
cd AgentQMS/bin && make compliance-fix-ai LIMIT=10
```

This uses Grok AI to automatically fix compliance issues.

---

## Statistics

### Current State (Jan 20, 2026)

| Metric | Count |
|--------|-------|
| Total violations | 10 |
| Critical violations | 10 |
| Files with violations | 4 |
| Files importing config_utils | 23 |
| Config access patterns found | 1,024 |

### Priority Files

1. `ocr/data/lightning_data.py` - 6 violations
2. `ocr/data/datasets/__init__.py` - 2 violations
3. `ocr/core/utils/config_utils.py` - 1 violation
4. `ocr/command_builder/overrides.py` - 1 violation

---

## Next Steps

1. **Fix high-priority violations** (4 files, ~30 minutes)
2. **Re-scan** to verify fixes
3. **Update tests** if config_utils behavior changes
4. **Document** any legitimate exceptions (e.g., internal implementation in config_utils.py)
5. **Set up CI check** to prevent future violations

---

## Additional Resources

- **Configuration Standards:** [configuration-standards.yaml](../../AgentQMS/standards/tier2-framework/configuration-standards.yaml)
- **Utility Functions:** [ocr/core/utils/config_utils.py](../../ocr/core/utils/config_utils.py)
- **AST Tool Documentation:** [agent-debug-toolkit/AI_USAGE.yaml](../../agent-debug-toolkit/AI_USAGE.yaml)
- **MCP Schema:** [AgentQMS/mcp_schema.yaml](../../AgentQMS/mcp_schema.yaml)

---

## Automation Options

### Add to Pre-Commit Hook

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: config-compliance
      name: Check Config Compliance
      entry: python scripts/audit_config_compliance.py
      language: system
      pass_filenames: false
      always_run: true
```

### Add to CI/CD

```yaml
# .github/workflows/compliance.yml
- name: Check Configuration Standards
  run: |
    python scripts/audit_config_compliance.py
    if grep -q "CRITICAL" docs/reports/config_compliance_audit.txt; then
      exit 1
    fi
```

---

## Questions?

For issues or questions about configuration standards, see:
- [AGENTS.md](../../AGENTS.md) for overall project guidance
- [AgentQMS documentation](../../AgentQMS/standards/INDEX.yaml) for standards index
- Run `cd AgentQMS/bin && make help` for available tools
