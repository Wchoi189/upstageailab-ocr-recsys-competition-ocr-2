# Taxonomy Freeze — AgentQMS Standards

Date: 2026-01-28
Status: Draft

## Objective
Freeze the standards taxonomy to eliminate ambiguity before refactors. This is the authoritative classification scheme used for audits, ledgers, and registry updates.

## Canonical Functional Taxonomy
1. **Reference (Specs)**
   - Purpose: describe interfaces, contracts, schemas, and canonical patterns.
   - Must not include enforcement language (rules, validation, SLAs).
   - Typical verbs: “defines”, “describes”, “enumerates”.

2. **Constraint (Validation/Rules)**
   - Purpose: enforce requirements, prohibitions, validations, SLAs.
   - Must not include narrative or architecture overviews except minimal context for enforcement.
   - Typical verbs: “must”, “shall”, “prohibit”, “validate”.

3. **Discovery (Routing/Indexing)**
   - Purpose: keyword maps, registries, and routing metadata.
   - Must not include spec or enforcement content.

4. **Runtime (Orchestration/Execution)**
   - Purpose: control flow, orchestration steps, runtime sequencing, lifecycle procedures.
   - Must not include policy enforcement except minimal preconditions.

## Canonical Folder Map
- specs/ → Reference
- constraints/ → Constraint
- discovery/ → Discovery
- runtime/ → Runtime

## File Archetypes (Naming Patterns)
- *-spec.yaml → Reference
- *-contracts.yaml → Reference
- *-patterns.yaml → Reference
- *-rules.yaml → Constraint
- *-validation.yaml → Constraint
- *-slas.yaml → Constraint
- *-index.yaml / *-registry.yaml / *-manifest.yaml → Discovery
- *-workflow.yaml / *-orchestration.yaml → Runtime

## Boundary Rules
- Any section with >20% off‑category content must be split or relocated.
- Each file must declare its functional category in the header (future enforcement).
- Registry updates are mandatory for any move or split.

## Acceptance Criteria (Freeze Complete)
- All tiers agree on this taxonomy.
- Ledger template is published and used for all audits.
- No refactor begins without a line‑level ledger entry.

## References
- [__DEBUG__/functional-purity-audit-standards.md](__DEBUG__/functional-purity-audit-standards.md)
- [AgentQMS/standards/registry.yaml](AgentQMS/standards/registry.yaml)
