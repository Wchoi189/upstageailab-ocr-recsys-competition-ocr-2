# Refactor Batch Plan — Session E (Readiness & Validation)

Date: 2026-01-28
Status: Draft
Scope: Convert overlap findings into executable refactor batches with gates
Inputs:
- project_compass/pulse_staging/artifacts/implementation_plan.md
- project_compass/pulse_staging/artifacts/tier2-line-ledger.md
- project_compass/pulse_staging/artifacts/tier1-3-4-line-ledger.md
- project_compass/pulse_staging/artifacts/overlap-matrix-session-d.md
- project_compass/pulse_staging/artifacts/refactor-map-session-d.md
- AgentQMS/standards/registry.yaml

## Session E Principles (Must Hold)
- No passive refactors: every change references a ledger entry.
- Registry updates are atomic with moves/splits.
- Discovery routing must be validated after every batch.

## Pre‑Checks (Before Any Batch)
1. Confirm AG-006 and FW-037 registry mismatch resolution plan is attached to Batch 1.
2. Ensure discovery routing sources are centralized (tool catalog + workflow detector).
3. Validate taxonomy targets (specs/constraints/discovery/runtime) for each file.

## Batch 1 — Discovery Routing Consolidation
**Scope:** tier1-sst discovery files + workflow triggers
**Entries:** C‑1, C‑12, C‑15, D‑workflow triggers

Pre‑Checks:
- Identify all task→bundle mappings currently referenced by agents.

Move/Split Steps:
- Move discovery-rules.yaml, workflow-detector.yaml, manifest.yaml to tier2-framework/discovery/.
- Extract experiment-workflow.yaml triggers to tier2-framework/discovery/experiment-workflow-triggers.yaml.

Registry Updates:
- Fix FW-037 path.
- Add new discovery entries for workflow-detector, utility-scripts-manifest, experiment-workflow-triggers.

Validation Gates:
- Registry sync passes.
- Discovery routing still resolves task→bundle mappings.

Rollback Criteria:
- Any missing task mapping or registry ambiguity.

## Batch 2 — SST Schema vs Validation Split
**Scope:** frontmatter-master + artifact-types
**Entries:** C‑10, C‑11

Move/Split Steps:
- Split frontmatter schema vs validation.
- Split artifact type catalog vs validation.

Registry Updates:
- Replace SC-002/SC-005 paths or add new SC ids.

Validation Gates:
- Schema reference contains no enforcement rules.

Rollback Criteria:
- Compliance checker fails to locate schema fields.

## Batch 3 — Workflow Policies and Runbooks
**Scope:** validation protocols + workflow requirements + middleware policies
**Entries:** C‑13, C‑14, C‑8

Move/Split Steps:
- Move middleware-policies.yaml to tier2 constraints.
- Split validation/protocol requirements into tier1 constraints + tier4 runbook.

Registry Updates:
- Update WF-002 tier/path; add runbook WF entry.

Validation Gates:
- Tier 4 contains execution steps only.

Rollback Criteria:
- Policy and runtime steps still co‑located.

## Batch 4 — Agent Persona Isolation
**Scope:** tier3 persona configs
**Entries:** C‑16, C‑5

Move/Split Steps:
- Extract generic policy from tier3 config to tier1 constraints.
- Ensure Tier 3 retains identity‑only content.

Registry Updates:
- Update AG-002/AG-004 paths.

Validation Gates:
- Tier 3 contains no global policy or workflow commands.

Rollback Criteria:
- Persona content still references generic constraints.

## Batch 5 — Agent Workflows to Tier 4
**Scope:** multi‑agent, VLM tools, bloat detection
**Entries:** C‑3, C‑4, C‑6

Move/Split Steps:
- Move runbooks to tier4 workflows; move constraints to tier2.

Registry Updates:
- Retire or repoint AG-001/AG-003/AG-005 and add new FW/WF entries.

Validation Gates:
- Tier 3 has no runtime commands.

Rollback Criteria:
- Workflow commands still in tier3.

## Batch 6 — Compliance Reporting Hygiene
**Scope:** compliance reporting tooling and outputs
**Entries:** C‑18, C‑9

Move/Split Steps:
- Relocate report output to docs/artifacts/compliance_reports.
- Relocate script to AgentQMS/tools with uv run python usage.

Registry Updates:
- Remove generated output from standards routing.

Validation Gates:
- No generated outputs under standards.

Rollback Criteria:
- Standards tree still contains generated outputs.

## Pruning Candidates (Consolidation‑First)
- Redundant discovery catalogs: discovery-rules.yaml vs tool catalog keywords.
- Repeated validation workflows: validation-protocols.yaml vs pre-commit hooks.
- Persona quick references that restate global SST rules.

## Change Logging
- Each move/split must cite the ledger entry (Session B/C).
- Registry updates are atomic with file moves.

## Next Session Trigger
Proceed to execution only after explicit approval of this batch plan.
