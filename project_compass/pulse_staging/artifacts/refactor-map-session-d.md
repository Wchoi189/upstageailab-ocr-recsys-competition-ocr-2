# Refactor Map — Session D (File‑Level Move/Split Plan)

Date: 2026-01-28
Status: Draft
Scope: Tier 1/3/4 moves and splits derived from Session C ledger and overlap matrix
Inputs:
- project_compass/pulse_staging/artifacts/tier1-3-4-line-ledger.md
- project_compass/pulse_staging/artifacts/tier2-line-ledger.md
- project_compass/pulse_staging/artifacts/overlap-matrix-session-d.md
- AgentQMS/standards/registry.yaml

## Batch 1 — Discovery Routing Consolidation
**Goal:** Centralize discovery in tier2-framework/discovery.

Moves:
- tier1-sst/discovery-rules.yaml → tier2-framework/discovery/discovery-rules.yaml
- tier1-sst/workflow-detector.yaml → tier2-framework/discovery/workflow-detector.yaml
- tier1-sst/manifest.yaml → tier2-framework/discovery/utility-scripts-manifest.yaml
- tier4-workflows/experiment-workflow.yaml (triggers section only) → tier2-framework/discovery/experiment-workflow-triggers.yaml

Registry updates:
- Update FW-037 path + tier
- Add new discovery entries for workflow-detector, utility-scripts-manifest, experiment-workflow-triggers

Validation gates:
- Registry sync passes
- Discovery routing still resolves task→bundle mappings

## Batch 2 — SST Schema vs Validation Split
**Goal:** Separate schemas (reference) from enforcement (constraints).

Splits:
- frontmatter-master.yaml → specs/frontmatter-schema.yaml + constraints/frontmatter-validation.yaml
- artifact-types.yaml → specs/artifact-types-reference.yaml + constraints/artifact-type-validation.yaml

Registry updates:
- Replace SC-002/SC-005 with new paths (or create new SC ids)

Validation gates:
- No enforcement rules remain in specs
- Compliance checker schema still resolves required fields

## Batch 3 — Workflow Policies and Runbooks
**Goal:** Keep Tier 4 for execution; move policies to constraints.

Moves/Splits:
- tier4-workflows/middleware-policies.yaml → tier2-framework/constraints/middleware-policies.yaml
- tier1-sst/validation-protocols.yaml → tier1-sst/constraints/validation-policies.yaml + tier4-workflows/validation-runbook.yaml
- tier1-sst/workflow-requirements.yaml → tier1-sst/constraints/workflow-requirements.yaml + tier4-workflows/validation-runbook.yaml (merge)
- tier4-workflows/experiment-workflow.yaml (rules section) → tier1-sst/constraints/experiment-workflow-policies.yaml

Registry updates:
- Update WF-002 to tier2 constraints
- Add new runbook entry (WF id)

Validation gates:
- Tier 4 contains only execution steps
- Tier 1 constraints contain no procedural steps

## Batch 4 — Agent Persona Isolation
**Goal:** Keep Tier 3 identity‑only; move policy/flow elsewhere.

Moves/Splits:
- tier3-agents/config.yaml → tier3-agents/qwen-identity.yaml + tier1-sst/constraints/agent-policy.yaml
- tier3-agents/quick-reference.yaml → update to persona‑specific guidance or deprecate and move general rules to SST

Registry updates:
- Update AG-002/AG-004 paths

Validation gates:
- Tier 3 contains persona identity only

## Batch 5 — Agent Workflows to Tier 4
**Goal:** Move operational runbooks out of Tier 3.

Moves/Splits:
- tier3-agents/multi-agent-system.yaml → tier4-workflows/multi-agent-runbook.yaml + tier2-framework/agent-infra/multi-agent-architecture.yaml
- tier3-agents/vlm-tools.yaml → tier4-workflows/vlm-tools-workflow.yaml + tier2-framework/tool-catalog.yaml reference
- tier3-agents/bloat-detection-rules.yaml → tier2-framework/constraints/bloat-detection-rules.yaml + tier4-workflows/bloat-detection-workflow.yaml

Registry updates:
- Update AG-001/AG-003/AG-005 to new locations (or deprecate AG ids, add FW/WF ids)

Validation gates:
- Tier 3 has no runtime commands
- Constraints vs workflows split cleanly

## Batch 6 — Compliance Reporting Hygiene
**Goal:** Remove generated outputs from standards; isolate tooling.

Moves:
- tier4-workflows/compliance-reporting/latest-report.txt → docs/artifacts/compliance_reports/latest-report.txt
- tier4-workflows/compliance-reporting/generate-compliance-report.py → AgentQMS/tools/compliance/generate_report.py

Changes:
- Replace sys.path insertion with paths utility
- Replace `python3` invocations with `uv run python` in workflow spec

Registry updates:
- Remove generated report from standards routing
- Add workflow YAML spec if missing

Validation gates:
- No generated outputs under standards
- Policy compliance: no sys.path manipulation, no raw python

## Risk Notes — Session E Readiness

### Registry Tier/Path Mismatches (Pre‑Batch Fix)
- **AG-006**: Registry lists `tier: 3` but points to `tier2-framework/agent-infra/ollama-models.yaml`.
- **FW-037**: Registry lists Tier 2 discovery but points to `tier1-sst/discovery-rules.yaml`.

**Required Action (Session E Gate):** Correct these entries in the same batch as the corresponding moves; do not allow registry drift between batches.

### Mandatory Gates for Any Move
- **Registry sync** must pass immediately after each move/split.
- **Discovery routing** must be re‑verified for task→bundle mappings (workflow detector + tool catalog).

### Refactor Hold
- **No content refactors** (rewrites) until the Session E batch plan is approved. Only move/split operations tied to ledger entries.

## Rollback Criteria
- Registry sync failures
- Discovery routing misses required tasks
- Any file exceeds 20% off‑category after move/split

## Change Logging Requirements
- Each move/split must reference a ledger entry (Session B/C)
- Update registry atomically with any file path change
- Record acceptance checks per batch
