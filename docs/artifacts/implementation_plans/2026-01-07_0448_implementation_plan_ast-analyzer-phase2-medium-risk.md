---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation', 'plan', 'ast', 'analyzer', 'medium-risk', 'phase2']
title: "AST Analyzer Phase 2: Medium Risk Features"
date: "2026-01-07 04:48 (KST)"
branch: "refactor/hydra"
priority: "medium"
estimated_effort: "10-14 days"
depends_on: "ast-analyzer-phase1-low-risk"
---

# Implementation Plan - AST Analyzer Phase 2: Medium Risk Features

## Goal

Implement advanced analyzers requiring deeper AST traversal, scope tracking, and heuristic matching. Medium risk due to potential false positives and scope complexity.

## Features

| Feature                 | Purpose                                | Effort   | Risk Factor            |
| ----------------------- | -------------------------------------- | -------- | ---------------------- |
| Duplication Detector    | Find copy-paste code patterns          | 2-3 days | Tuning sensitivity     |
| Type Inference Analyzer | Track variable types for validation    | 3-5 days | Cross-scope complexity |
| Security Scanner        | Detect unsafe patterns in config usage | 5-7 days | False positive rate    |

---

## Proposed Changes

### New Analyzers

#### [NEW] `analyzers/duplication_detector.py`

```yaml
class: DuplicationDetector
extends: BaseAnalyzer
algorithm:
  1. Normalize AST (remove comments, whitespace, variable names)
  2. Hash normalized subtrees (functions, code blocks)
  3. Compare hashes to find duplicates
  4. Report similar code with line ranges
config:
  min_lines: 5  # Minimum lines for duplicate detection
  similarity_threshold: 0.85  # 85% similarity
output:
  - duplicate_groups: [[file:line, file:line], ...]
  - similarity_score: Float 0-1
  - suggested_action: "Extract to function" | "Create shared module"
```

#### [NEW] `analyzers/type_inference.py`

```yaml
class: TypeInferenceAnalyzer
extends: BaseAnalyzer
tracks:
  - assignment_types: x = 5 → int, y = "foo" → str
  - function_return_types: Inferred from return statements
  - type_hints: Explicit annotations
  - type_narrowing: if isinstance(x, Foo): ...
limitations:
  - No cross-file analysis (single-file scope)
  - No runtime type inference
output:
  - variables: {name: inferred_type}
  - functions: {name: {params: types, return: type}}
  - type_conflicts: Assignments with conflicting types
```

#### [NEW] `analyzers/security_scanner.py`

```yaml
class: SecurityScanner
extends: BaseAnalyzer
patterns:
  hydra_specific:
    - unsafe_instantiate: instantiate() with user input
    - dynamic_target: _target_ from untrusted source
    - config_exec: cfg values passed to eval/exec
  general:
    - hardcoded_secrets: API keys, passwords in code
    - path_traversal: os.path.join with unchecked input
    - shell_injection: subprocess with shell=True + config
severity_levels: [critical, high, medium, low, info]
output:
  - findings: [{pattern, severity, file, line, recommendation}]
  - summary: Count by severity
```

---

### CLI Commands

```yaml
new_commands:
  - name: detect-duplicates
    path_arg: required
    options: [--min-lines INT, --threshold FLOAT, --format json|markdown]

  - name: infer-types
    path_arg: required
    options: [--format json|markdown, --show-conflicts]

  - name: security-scan
    path_arg: required
    options: [--severity LEVEL, --format json|markdown|sarif]
```

### MCP Server Tools

```yaml
new_tools:
  - detect_code_duplicates:
      description: Find duplicate/similar code blocks
      params: [path, min_lines, threshold]

  - infer_types:
      description: Infer variable and function types
      params: [path]

  - security_scan:
      description: Scan for security vulnerabilities in config usage
      params: [path, severity_filter]
```

---

## Verification Plan

### Automated Tests

```bash
# Unit tests with known patterns
uv run pytest tests/test_duplication_detector.py -v
uv run pytest tests/test_type_inference.py -v
uv run pytest tests/test_security_scanner.py -v

# False positive testing
uv run adt security-scan ocr/ --severity critical 2>&1 | grep -c "Finding"
```

### Manual Verification

- [ ] Duplication detector finds known copy-paste in codebase
- [ ] Type inference handles OmegaConf `DictConfig` correctly
- [ ] Security scanner flags `eval(cfg.command)` patterns
- [ ] No false positives on clean utility files

---

## Risk Mitigation

| Risk                                | Mitigation                                            |
| ----------------------------------- | ----------------------------------------------------- |
| False positives in security scanner | Start with Hydra-specific patterns only, expand later |
| Type inference across scopes        | Document single-file limitation clearly               |
| Duplication sensitivity             | Configurable thresholds, conservative defaults        |

---

## Dependencies

- Phase 1 complete (shared infrastructure)
- May need `typing` module integration for type hints
