---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: "ast,hydra,config,phase2.5,implementation,plan"
title: "AST Analyzer Phase 2.5: Hydra Config Resolver"
date: "2026-01-07 19:18 (KST)"
branch: "refactor/hydra"
description: "Implementation plan for Phase 2.5: Hydra Config Resolver tool, enabling static mapping of Hydra configs to Python files. Includes Symbol Table Service requirement."
---

# Implementation Plan - AST Analyzer Phase 2.5: Hydra Config Resolver

## Goal

Implement the **Hydra Config Resolver** tool ("Phase 2.5") to statically map Hydra configuration `_target_` keys to their corresponding Python file definitions. This requires a new **Symbol Table Service** to track cross-file definitions.

This tool directly addresses the "Audit Survey" request for visibility into Hydra's config-to-code mapping and serves as a prerequisite for future automated refactoring.

---

## Technical Approach

### 1. Symbol Table Service (`precomputes/symbol_table.py`)
A reusable service effectively implementing the "Two-Pass Analysis" pattern:
- **Pass 1 (Indexing):** Walk the project directory, parsing files with `ast` to find all class and function definitions.
- **Pass 2 (Resolution):** Build a mapping of `FullyQualifiedName -> (FilePath, LineNumber)`.

**Schema:**
```python
@dataclass
class SymbolDefinition:
    name: str          # "ResNet"
    full_name: str     # "ocr.models.encoders.ResNet"
    file_path: str     # "/abs/path/ocr/models/encoders.py"
    line_number: int   # 45
    kind: str          # "class" | "function"
```

### 2. Hydra Config Resolver (`analyzers/hydra_config_resolver.py`)
An analyzer that parses YAML configs and uses the Symbol Table to resolve `_target_` keys.

**Resolution Strategy (Per Research/Design):**
- **Static Paths (`ocr.models.ResNet`)**: Direct lookup in Symbol Table.
- **Interpolated Paths (`${module}.${name}`)**: Flag as `UNRESOLVED_DYNAMIC`.
- **Relative Paths**: Flag as `UNRESOLVED_RELATIVE` (for now, unless simple to resolve).

---

## Proposed Changes

### Core Logic

#### [NEW] `src/agent_debug_toolkit/precomputes/symbol_table.py`
```python
class SymbolTable:
    def __init__(self, root_path: str): ...
    def build(self): ...
    def lookup(self, full_name: str) -> Optional[SymbolDefinition]: ...
```

#### [NEW] `src/agent_debug_toolkit/analyzers/hydra_config_resolver.py`
```python
class HydraConfigResolver(BaseAnalyzer):
    def resolve_target(self, target_string: str) -> ResolutionResult: ...
    def analyze_config_dir(self, config_dir: str) -> List[ConfigMapping]: ...
```

### Interfaces

#### [MODIFY] `src/agent_debug_toolkit/cli.py`
Add `resolve-configs` command:
```bash
adt resolve-configs conf/ --output markdown --module-root src/
```

#### [MODIFY] `src/agent_debug_toolkit/mcp_server.py`
Add `resolve_hydra_configs` tool:
```json
{
  "name": "resolve_hydra_configs",
  "description": "Map Hydra config _target_ keys to Python files",
  "inputSchema": {
    "type": "object",
    "properties": {
      "config_dir": {"type": "string"},
      "module_root": {"type": "string"}
    }
  }
}
```

---

## Verification Plan

### Automated Tests
- [ ] **Unit Test: Symbol Table**
  - Create temporary generic project structure.
  - Verify `SymbolTable` correctly finds classes and functions.
  - Verify it handles duplicate names (favors last found or lists duplicates).

- [ ] **Unit Test: Config Resolver**
  - Create mock YAML configs with:
    - Valid static target (`pkg.mod.Class`) -> Expect RESOLVED
    - Dynamic target (`${var}.Class`) -> Expect DYNAMIC
    - Non-existent target -> Expect NOT_FOUND
  - Verify analyzer output matches expectations.

- [ ] **Integration Test**
  - Run against `ocr/` directory codebase.
  - `uv run pytest tests/test_hydra_resolver.py`

### Manual Verification
- [ ] Run `adt resolve-configs` on the actual `conf/` directory.
- [ ] Check a few known mappings (e.g., the Experiment Registry or Model definitions) to ensure they point to the correct files.
