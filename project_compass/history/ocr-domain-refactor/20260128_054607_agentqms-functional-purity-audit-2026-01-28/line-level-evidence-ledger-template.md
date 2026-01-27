# Line‑Level Evidence Ledger Template

Date: 2026-01-28
Status: Draft

## Purpose
Capture line‑level evidence for every boundary violation. This ledger is required for all refactor actions (move/split/merge).

## Ledger Fields
- File: (path)
- Section: (heading or YAML key path)
- Line Range: (Lx–Ly)
- Current Category: (Reference / Constraint / Discovery / Runtime)
- Intended Category: (Reference / Constraint / Discovery / Runtime)
- Evidence: (quote or short summary)
- Proposed Action: (move / split / merge / delete / keep)
- Target Path: (if moving/splitting)
- Registry Update Needed: (Yes/No)
- Validation Gate: (list required checks)

## Ledger Template (Repeat per finding)

### Entry N
- File:
- Section:
- Line Range:
- Current Category:
- Intended Category:
- Evidence:
- Proposed Action:
- Target Path:
- Registry Update Needed:
- Validation Gate:

## Notes
- Every refactor must reference one or more ledger entries.
- Off‑category content >20% is mandatory split.
- Use this ledger before any code or file movement.

## References
- [project_compass/pulse_staging/artifacts/implementation_plan.md](project_compass/pulse_staging/artifacts/implementation_plan.md)
- [__DEBUG__/functional-purity-audit-standards.md](__DEBUG__/functional-purity-audit-standards.md)
