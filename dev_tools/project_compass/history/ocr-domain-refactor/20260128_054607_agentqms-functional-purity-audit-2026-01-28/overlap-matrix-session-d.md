# Overlap Matrix — Session D (Cross‑Tier Synthesis)

Date: 2026-01-28
Status: Draft
Scope: Tier 1/3/4 overlaps (Session C outputs) + cross‑checks with baseline audit
Sources:
- Session C ledger: project_compass/pulse_staging/artifacts/tier1-3-4-line-ledger.md
- Tier‑2 ledger: project_compass/pulse_staging/artifacts/tier2-line-ledger.md
- Baseline audit: __DEBUG__/functional-purity-audit-standards.md
- Registry: AgentQMS/standards/registry.yaml

## Overlap Matrix (Clusters)

| Cluster | Files (current) | Overlap Type | Dominant Function | Off‑Category Evidence | Proposed Split/Move | Registry Impact |
|---|---|---|---|---|---|---|
| SST Discovery Routing | tier1-sst/discovery-rules.yaml; tier1-sst/workflow-detector.yaml; tier1-sst/manifest.yaml | Discovery in SST | Discovery | Keyword routing, task mapping, utility registry | Move to tier2-framework/discovery/* | Update FW-037 path; add discovery entries |
| SST Workflow Policy | tier1-sst/workflow-requirements.yaml; tier1-sst/validation-protocols.yaml | Runtime procedures in SST | Constraint | Command sequences, runbooks, validation flow | Split policies vs runbooks | Add tier4 workflow runbook; adjust SC-008/SC-010 |
| SST Schema vs Validation | tier1-sst/frontmatter-master.yaml; tier1-sst/artifact-types.yaml | Reference + Constraint mix | Reference | Validation rules embedded in schema catalog | Split reference schema vs validation constraints | Update SC-002/SC-005 paths and new constraint ids |
| Agent Persona vs Policy | tier3-agents/config.yaml; tier3-agents/quick-reference.yaml | Persona + global policy | Constraint | Validation commands, prohibited actions | Extract generic policy to tier1 constraints; keep persona identity only | Update AG-002/AG-004 and add new SC entry |
| Agent Workflows in Tier 3 | tier3-agents/multi-agent-system.yaml; tier3-agents/vlm-tools.yaml; tier3-agents/bloat-detection-rules.yaml | Runtime/Workflow in Tier 3 | Runtime | CLI commands, automation steps | Move runbooks to tier4 workflows; move constraints to tier2 | Update AG-001/AG-003/AG-005; add new WF/FW entries |
| Workflow Policies in Tier 4 | tier4-workflows/middleware-policies.yaml | Constraint inside workflow tier | Constraint | Enforcement rules (must/never) | Move to tier2 constraints | Update WF-002 path and tier |
| Workflow Triggers in Tier 4 | tier4-workflows/experiment-workflow.yaml | Discovery triggers embedded | Runtime | Pattern triggers + policy rules inside workflow definition | Split triggers to tier2 discovery, rules to tier1 constraints | Update WF-001; add discovery and constraint entries |
| Generated Artifacts in Standards | tier4-workflows/compliance-reporting/latest-report.txt | Output in standards tree | Runtime output | Generated report stored in standards | Move to docs/artifacts/compliance_reports | Remove from standards routing |
| Compliance Script Policy Drift | tier4-workflows/compliance-reporting/generate-compliance-report.py | Tooling inside standards | Runtime tooling | Uses sys.path insert + python3 | Move to AgentQMS/tools + workflow spec YAML | Update workflow entry + tooling references |

## Cross‑Tier Redundancy Hotspots
1. **Discovery routing** duplicated across SST and Tier 2 tool catalog (discovery-rules.yaml vs tool-catalog.yaml).
2. **Validation workflows** embedded in SST and Tier 4 scripts (validation-protocols.yaml + pre‑commit hooks).
3. **Agent policy duplication** in Tier 3 quick references vs SST rules (naming/placement/validation).
4. **Workflow policy overlap** between workflow definitions and SST constraints (manual creation bans, validation mandates).

## Risk Notes
- Registry tier/path mismatches already exist (AG-006, FW-037). Must be corrected in same batch as moves.
- Any move requires registry sync and discovery routing update gates.
- Do not refactor content until Session E batch plan is approved.
