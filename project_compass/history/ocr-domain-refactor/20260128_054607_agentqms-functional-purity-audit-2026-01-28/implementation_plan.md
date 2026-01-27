# Implementation Plan — Multi‑Session Audit & Refactor Readiness

## Objective
Establish a surgical, multi‑session audit and refactor readiness plan to restore clear architectural boundaries within AgentQMS standards. The plan must prevent passive refactors by enforcing line‑level evidence, explicit change gates, and registry integrity checks.

## Scope
- Target: AgentQMS/standards (all YAML files across tiers)
- Inputs: Prior audit report [__DEBUG__/functional-purity-audit-standards.md](__DEBUG__/functional-purity-audit-standards.md), registry map [AgentQMS/standards/registry.yaml](AgentQMS/standards/registry.yaml)
- Constraint: Documentation is AI‑facing only (no user docs)

## Principles (Non‑negotiable)
1. One file, one function (Reference / Constraint / Discovery / Runtime).
2. All refactors must be justified by line‑level audit notes.
3. Registry integrity is a gate: changes must be mirrored in registry and discovery rules.
4. No passive refactors: every change must tie back to a logged overlap or boundary violation.

## Audit Strategy (Multi‑Session)
### Session A — Baseline & Taxonomy Freeze
- Freeze taxonomy definition and naming/placement rules.
- Create canonical folder map (specs/constraints/discovery/runtime) and file archetypes.
- Produce “Line‑Level Evidence Ledger” template (required for all later sessions).

### Session B — Tier‑2 Deep Audit (Line‑Level)
- Full read of tier2‑framework YAMLs.
- Annotate every section with functional category.
- Identify sections >20% off‑category and mark for move/split.

### Session C — Tier‑1/3/4 Audits
- Apply same line‑level method to tier1‑sst, tier3‑agents, tier4‑workflows.
- Flag boundary leaks into runtime or constraint tiers.

### Session D — Cross‑Tier Overlap Synthesis
- Build overlap matrix across all tiers.
- Identify redundancy clusters and propose merge/split candidates.
- Prepare refactor map (file‑level move/split plan).

### Session E — Refactor Readiness & Validation Plan
- Convert audit results into refactor batches.
- For each batch: specify pre‑checks, move/split steps, registry updates, and validation gates.
- Define rollback criteria and change logging.

### Session F — Post‑Refactor Verification
- Re‑scan standards for overlap reduction.
- Confirm registry sync and discovery routing work.
- Finalize “steady‑state” structure and maintenance rules.

## Required Artifacts (Per Session)
- Inventory tables with scores and overlap flags.
- Line‑level evidence ledger (source line → category → action).
- Refactor map (move/split/merge plan with acceptance checks).
- Registry update checklist and discovery rule updates.

## Milestones (Recommended)
1. **M1 — Taxonomy Freeze & Ledger Template**
   - Output: Canonical folder map + line‑level ledger template.
2. **M2 — Tier‑2 Line Audit Complete**
   - Output: Tier‑2 evidence ledger + overlap matrix.
3. **M3 — Tier‑1/3/4 Line Audit Complete**
   - Output: Full‑tier evidence ledger.
4. **M4 — Cross‑Tier Overlap Resolution Plan**
   - Output: Consolidation map with merge/split decisions.
5. **M5 — Refactor Batches Defined**
   - Output: Batch plan with gates, registry updates, rollback points.
6. **M6 — Post‑Refactor Validation**
   - Output: Updated inventory + reduced overlap metrics.

## Validation Gates (Must Pass)
- Registry sync passes without ambiguity.
- Discovery rules map new file paths.
- No file has >20% off‑category content.
- Every change references a line‑level ledger entry.

## Risks & Mitigations
- **Scope creep** → lock taxonomy before refactor.
- **Passive refactor** → require ledger evidence and acceptance checks.
- **Registry drift** → apply registry updates in every batch.

## Continuation Prompt (Next Session)
Use this prompt to resume the audit series:

“Continue the AgentQMS standards multi‑session audit. Start from the canonical taxonomy and ledger template created in M1. Use the baseline audit in [__DEBUG__/functional-purity-audit-standards.md](__DEBUG__/functional-purity-audit-standards.md) and verify against [AgentQMS/standards/registry.yaml](AgentQMS/standards/registry.yaml). Perform the next planned session, update the line‑level evidence ledger, and produce any required artifacts (inventory tables, overlap matrix, refactor map). Enforce AI‑facing documentation only. Do not refactor yet unless the session explicitly targets a refactor batch.”

## Status
- Created: 2026-01-28T02:54:07.021039
- Tool: Project Compass v2
- Status: Draft (updated with detailed plan)
