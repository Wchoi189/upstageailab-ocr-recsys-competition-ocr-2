---
ads_version: '2.0'
id: 'FW-FRAMEWORK_SPECS.SPEC'
type: 'rule_set'
tier: 2
priority: 'high'
spec_version: '1.0.0'
updated: '2026-02-03'
description: 'Framework Specifications for framework tier'
---

# Framework Specifications

**Tier**: 2 (Framework)
**Scope**: Reference Patterns, Anti-Patterns, and API Contracts.

## 1. Anti-Patterns (Do NOT Do)

| Pattern | Why Bad? | Fix |
| :--- | :--- | :--- |
| **Monolithic Configs** | Hard to maintain, merge conflicts. | **Hydra Domain-First** (Split by domain). |
| **Hardcoded Paths** | Breaks on different machines. | Use `get_project_root()`. |
| **God Classes** | > 500 lines, too many responsibilities. | Split into Components + Orchestrator. |
| **Silent Failures** | `try: ... except: pass` hides bugs. | Log stack trace + Raise. |

## 2. API Contracts & Interfaces

**Principles**:
*   **TypedDict / Dataclasses**: Prefer over raw Dicts for interface boundaries.
*   **Immutability**: Config objects passed to functions should be immutable.
*   **Explicit Returns**: Functions must return `Result` objects, not just `True/False`.

### Hydra Patterns Reference
> [!NOTE]
> See `configuration.spec.md` for full Hydra rules.
*   **Self-Mounting**: Components define their own `@package`.
*   **Atomic Architecture**: Models Only contain layers, no training logic.
