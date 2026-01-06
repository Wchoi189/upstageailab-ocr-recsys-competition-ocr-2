---
ads_version: "1.0"
type: "implementation_plan"
category: "research"
status: "deferred"
version: "1.0"
tags: ['implementation', 'plan', 'ast', 'refactoring', 'conceptual', 'phase3']
title: "AST Analyzer Phase 3: Conceptual Plan for Automated Refactoring"
date: "2026-01-07 04:48 (KST)"
branch: "refactor/hydra"
priority: "low"
estimated_effort: "TBD - requires research"
depends_on: "ast-analyzer-phase2-medium-risk"
---

# Conceptual Plan - Automated Refactoring Tools

## Problem Statement

> "I have a very big code base... bloat, inconsistencies, obsolete, deprecated, broken, outdated, superseded, incomplete, overlapping, conflicting, abandoned migrations, invisible legacy code, mysterious behaviors... Agents are not able to navigate quickly and this itself is a compute burden."

The core problem is **cognitive overhead for AI agents** when navigating complex, legacy-laden codebases. Automated refactoring could reduce this burden by:

1. **Detecting** problematic patterns
2. **Suggesting** concrete fixes
3. **Applying** safe transformations automatically

---

## Use Case Categories

### 1. Dead Code Elimination

```yaml
name: dead_code_eliminator
detects:
  - Unused imports (from Phase 1)
  - Unreachable code (after return/raise)
  - Unused functions/classes (no callers)
  - Commented-out code blocks
  - TODO/FIXME markers > 6 months old
actions:
  report: List all dead code with confidence scores
  preview: Show diff of proposed removal
  apply: Remove dead code (requires confirmation)
risk: LOW (removals are reversible via git)
```

### 2. Configuration Consolidation

```yaml
name: config_consolidator
detects:
  - Duplicate config definitions
  - Conflicting default values
  - Scattered config access patterns
  - Legacy/deprecated config keys
actions:
  map: Create config usage map (key → files)
  suggest: Recommend canonical config location
  migrate: Generate migration script
risk: MEDIUM (semantic changes possible)
```

### 3. Import Optimization

```yaml
name: import_optimizer
detects:
  - Unused imports
  - Duplicate imports
  - Circular imports (from Phase 1)
  - Import * usage
  - Relative vs absolute import inconsistency
actions:
  sort: Apply isort-compatible ordering
  remove_unused: Remove unused imports
  convert_star: Expand star imports to explicit
risk: LOW (import changes rarely break semantics)
```

### 4. Function Extraction

```yaml
name: function_extractor
detects:
  - Long functions (>50 lines)
  - High cyclomatic complexity (from Phase 1)
  - Deeply nested blocks
  - Repeated code patterns (from Phase 2)
actions:
  suggest: Propose extraction points
  preview: Show extracted function signature
  apply: Extract and replace inline code
risk: HIGH (semantic preservation is complex)
```

### 5. Legacy Migration Assistant

```yaml
name: legacy_migrator
detects:
  - Deprecated API usage (from deprecation warnings)
  - Old-style class definitions
  - Python 2 compatibility code
  - Abandoned feature flags
  - Dead experiment branches
actions:
  audit: List all legacy patterns with age
  suggest: Recommend modern equivalents
  batch_migrate: Apply safe migrations
risk: MEDIUM-HIGH (requires domain knowledge)
```

---

## Technical Approach

### AST-to-AST Transformation

```python
# Conceptual architecture
class RefactoringEngine:
    def __init__(self):
        self.transformers: list[ASTTransformer] = []

    def analyze(self, path: Path) -> RefactoringReport:
        """Detect refactoring opportunities."""

    def preview(self, report: RefactoringReport) -> str:
        """Generate unified diff of proposed changes."""

    def apply(self, report: RefactoringReport, dry_run: bool = True):
        """Apply transformations (with safety checks)."""
```

### Safety Guarantees

| Guarantee            | Implementation                  |
| -------------------- | ------------------------------- |
| Syntax preservation  | Use `ast.unparse()` or `libcst` |
| Semantic equivalence | Run tests before/after          |
| Reversibility        | Commit changes atomically       |
| Human-in-loop        | Require explicit confirmation   |

---

## Research Questions

1. **AST unparsing fidelity**: Python's `ast.unparse()` loses formatting. Should we use `libcst` for concrete syntax preservation?

2. **Cross-file refactoring**: How to handle renames that span multiple files?

3. **Test coverage integration**: How to verify refactoring safety via existing tests?

4. **Agent interaction model**: Should refactoring be fully automatic, or assistant-guided?

5. **Rollback strategy**: How to handle failed refactorings mid-batch?

---

## Recommended Next Steps

1. **Spike**: Build minimal dead code eliminator (unused imports only)
2. **Evaluate**: Test on isolated module, measure accuracy
3. **Research**: Evaluate `libcst` vs `ast` for transformation fidelity
4. **Design**: Define agent-facing API for refactoring suggestions

---

## Related Tools to Evaluate

| Tool        | Purpose                           | Notes                               |
| ----------- | --------------------------------- | ----------------------------------- |
| `rope`      | Python refactoring library        | Mature, but slow on large codebases |
| `libcst`    | Concrete syntax tree manipulation | Preserves formatting                |
| `bowler`    | AST-based codemod tool            | By Facebook/Meta                    |
| `vulture`   | Dead code detection               | Good baseline comparison            |
| `autoflake` | Unused import removal             | Simple, well-tested                 |

---

## Decision: DEFERRED

This phase requires significant research and is deferred pending:

- [ ] Phase 1 and Phase 2 completion
- [ ] User feedback on priority use cases
- [ ] Spike implementation of dead code eliminator
- [ ] Evaluation of `libcst` integration effort
