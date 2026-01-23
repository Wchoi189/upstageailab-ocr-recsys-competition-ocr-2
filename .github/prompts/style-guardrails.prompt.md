---
name: Guardrails Style
description: Enforces strict, dependency-first guardrails for complex migrations.
---

<!-- To prevent the agent from deviating during this massive fix, present the information in a **Dependency-First Roadmap**. -->


**Project State:** Post-Refactor Migration.
**Current Goal:** Resolve 18 Broken Hydra Targets.

**Phase 1: Environment Lock (Guardrail)**
* Execute `python scripts/migration_guard.py`.
* **Hard Stop:** If "Ghost Code" or "Site-Packages" is detected, stop and report.

**Phase 2: Target Resolution**
* For each entry in `broken_targets.json`:
1. If target is dynamic (`${...}`), use `yq` to find the variable source.
2. Use `adt intelligent-search` to find the current physical location of the symbol.
3. Propose a `yq` command to update the YAML.

**Phase 3: Verification**
* Re-run `master_audit.py`. The "Kill List" must be 0.

