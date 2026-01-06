---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation', 'plan', 'ast', 'analyzer', 'low-risk', 'phase1']
title: "AST Analyzer Phase 1: Low Risk Features"
date: "2026-01-07 04:48 (KST)"
branch: "refactor/hydra"
priority: "high"
estimated_effort: "4-6 days"
---

# Implementation Plan - AST Analyzer Phase 1: Low Risk Features

## Goal

Extend `agent-debug-toolkit` with foundational analysis capabilities for AI agent codebase navigation. Focus on well-understood AST patterns with minimal risk.

## Features

| Feature                     | Purpose                                        | Effort   |
| --------------------------- | ---------------------------------------------- | -------- |
| Dependency Graph Analyzer   | Trace module/class dependencies for navigation | 2-3 days |
| Import Tracker              | Map all imports for dead code detection        | 1 day    |
| Complexity Metrics Analyzer | Cyclomatic complexity, nesting depth, LOC      | 1-2 days |

---

## Proposed Changes

### New Analyzers

#### [NEW] `analyzers/dependency_graph.py`

```yaml
class: DependencyGraphAnalyzer
extends: BaseAnalyzer
methods:
  - visit_Import: Track import dependencies
  - visit_ImportFrom: Track from...import dependencies
  - visit_Call: Track function/class instantiation calls
  - build_graph: Construct dependency DAG
  - find_cycles: Detect circular dependencies
  - get_dependents: Find what depends on a module
output:
  - nodes: List of modules/classes
  - edges: Dependency relationships
  - cycles: Circular dependency warnings
```

#### [NEW] `analyzers/import_tracker.py`

```yaml
class: ImportTracker
extends: BaseAnalyzer
patterns:
  - import module
  - from module import name
  - from module import name as alias
  - dynamic: __import__(), importlib.import_module()
output:
  - stdlib_imports: Set of standard library imports
  - third_party_imports: Set of external dependencies
  - local_imports: Set of project-local imports
  - unused_candidates: Imports not referenced (heuristic)
```

#### [NEW] `analyzers/complexity_metrics.py`

```yaml
class: ComplexityMetricsAnalyzer
extends: BaseAnalyzer
metrics:
  - cyclomatic_complexity: McCabe complexity per function
  - nesting_depth: Max nesting level
  - lines_of_code: Logical lines per function
  - param_count: Number of parameters
  - return_statements: Count of return points
thresholds:
  cyclomatic_high: 10
  nesting_high: 4
  params_high: 5
```

---

### CLI Commands

```yaml
new_commands:
  - name: analyze-dependencies
    path_arg: required
    options: [--format json|markdown|mermaid, --include-stdlib, --detect-cycles]

  - name: analyze-imports
    path_arg: required
    options: [--format json|markdown, --show-unused, --categorize]

  - name: analyze-complexity
    path_arg: required
    options: [--format json|markdown, --threshold INT, --sort-by metric]
```

### MCP Server Tools

```yaml
new_tools:
  - analyze_dependency_graph:
      description: Build module dependency graph
      params: [path, include_stdlib, detect_cycles]

  - analyze_imports:
      description: Categorize and analyze imports
      params: [path, show_unused]

  - analyze_complexity:
      description: Calculate code complexity metrics
      params: [path, threshold]
```

---

## Verification Plan

### Automated Tests

```bash
# Unit tests for new analyzers
uv run pytest tests/test_dependency_graph.py -v
uv run pytest tests/test_import_tracker.py -v
uv run pytest tests/test_complexity_metrics.py -v

# Integration test: run on actual codebase
uv run adt analyze-dependencies ocr/models/ --format json > /dev/null
uv run adt analyze-complexity ocr/models/ --threshold 10
```

### Manual Verification

- [ ] CLI `--help` shows new commands
- [ ] MCP tools visible in tool listing
- [ ] JSON/Markdown output correctly formatted
- [ ] Run on `ocr/` directory produces valid results

---

## Dependencies

- None (pure Python AST, stdlib only)
- Extends existing `BaseAnalyzer` infrastructure
