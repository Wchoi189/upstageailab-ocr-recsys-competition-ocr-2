# Functional Purity Audit: AgentQMS Standards

Date: 2026-01-28

## Scope
- Target: AgentQMS/standards (all YAML files across tiers)
- Files audited: 48 YAML files
- Taxonomy: Reference (Specs), Constraint (Validation/Rules), Discovery (Indexing/Keywords), Runtime (Orchestration/Execution)

## Methodology (Manual report, heuristic pre-scoring)
- Inventory generated from full YAML scan.
- Functional distribution inferred from keyword density across keys and values.
- Scoring rubric (0–100) applied as requested:
  - Clarity of Purpose (40%)
  - Taxonomy Alignment (30%)
  - Uniqueness (20%)
  - AI-Readability (10%)
- Overlap flag raised if dominant function share < 0.80.

> Note: This report is manually compiled, but uses a heuristic scan for baseline scoring. Files marked with overlap require deep, section-by-section review before any final move/merge decisions.

---

## 1) Inventory & Pre-Scoring (All Files)

| Path | Tier | Dominant | Score | Overlap |
|---|---|---|---:|:---:|
| AgentQMS/standards/registry.yaml | root | Runtime | 55 | Yes |
| AgentQMS/standards/tier1-sst/artifact-types.yaml | tier1-sst | Reference | 58 | Yes |
| AgentQMS/standards/tier1-sst/artifact_rules.yaml | tier1-sst | Reference | 86 | No |
| AgentQMS/standards/tier1-sst/discovery-rules.yaml | tier1-sst | Reference | 59 | Yes |
| AgentQMS/standards/tier1-sst/file-placement-rules.yaml | tier1-sst | Constraint | 86 | No |
| AgentQMS/standards/tier1-sst/frontmatter-master.yaml | tier1-sst | Reference | 54 | Yes |
| AgentQMS/standards/tier1-sst/manifest.yaml | tier1-sst | Discovery | 78 | Yes |
| AgentQMS/standards/tier1-sst/naming-conventions.yaml | tier1-sst | Constraint | 55 | Yes |
| AgentQMS/standards/tier1-sst/prohibited-actions.yaml | tier1-sst | Constraint | 91 | No |
| AgentQMS/standards/tier1-sst/system-architecture.yaml | tier1-sst | Constraint | 59 | Yes |
| AgentQMS/standards/tier1-sst/validation-protocols.yaml | tier1-sst | Constraint | 78 | Yes |
| AgentQMS/standards/tier1-sst/workflow-detector.yaml | tier1-sst | Runtime | 56 | Yes |
| AgentQMS/standards/tier1-sst/workflow-requirements.yaml | tier1-sst | Constraint | 58 | Yes |
| AgentQMS/standards/tier2-framework/agent-infra/ollama-models.yaml | tier2-framework | Constraint | 76 | Yes |
| AgentQMS/standards/tier2-framework/configuration/config-externalization-checklist.yaml | tier2-framework | Reference | 55 | Yes |
| AgentQMS/standards/tier2-framework/configuration/config-externalization-index.yaml | tier2-framework | Reference | 52 | Yes |
| AgentQMS/standards/tier2-framework/configuration/config-externalization-pattern.yaml | tier2-framework | Reference | 59 | Yes |
| AgentQMS/standards/tier2-framework/configuration/configuration-standards.yaml | tier2-framework | Reference | 77 | Yes |
| AgentQMS/standards/tier2-framework/configuration/hydra-configuration-architecture.yaml | tier2-framework | Runtime | 82 | Yes |
| AgentQMS/standards/tier2-framework/configuration/hydra-v5-rules.yaml | tier2-framework | Reference | 80 | Yes |
| AgentQMS/standards/tier2-framework/constraints/performance-slas.yaml | tier2-framework | Constraint | 88 | No |
| AgentQMS/standards/tier2-framework/constraints/pydantic-validation.yaml | tier2-framework | Constraint | 80 | Yes |
| AgentQMS/standards/tier2-framework/constraints/testing-standards.yaml | tier2-framework | Constraint | 79 | Yes |
| AgentQMS/standards/tier2-framework/core-infra/agent-architecture.yaml | tier2-framework | Reference | 56 | Yes |
| AgentQMS/standards/tier2-framework/core-infra/agent-feedback-protocol.yaml | tier2-framework | Runtime | 60 | Yes |
| AgentQMS/standards/tier2-framework/core-infra/python-core.yaml | tier2-framework | Reference | 57 | Yes |
| AgentQMS/standards/tier2-framework/debugging-sessions.yaml | tier2-framework | Runtime | 58 | Yes |
| AgentQMS/standards/tier2-framework/ml-frameworks.yaml | tier2-framework | Constraint | 52 | Yes |
| AgentQMS/standards/tier2-framework/ocr-engine/model-management.yaml | tier2-framework | Reference | 55 | Yes |
| AgentQMS/standards/tier2-framework/ocr-engine/pipeline-contracts.yaml | tier2-framework | Reference | 56 | Yes |
| AgentQMS/standards/tier2-framework/ocr-engine/postprocessing-logic.yaml | tier2-framework | Reference | 52 | Yes |
| AgentQMS/standards/tier2-framework/ocr-engine/preprocessing-logic.yaml | tier2-framework | Constraint | 54 | Yes |
| AgentQMS/standards/tier2-framework/runtime/coordinate-transforms.yaml | tier2-framework | Constraint | 56 | Yes |
| AgentQMS/standards/tier2-framework/runtime/image-loading.yaml | tier2-framework | Constraint | 68 | Yes |
| AgentQMS/standards/tier2-framework/runtime/orchestration-flow.yaml | tier2-framework | Runtime | 80 | Yes |
| AgentQMS/standards/tier2-framework/runtime/viz-engine.yaml | tier2-framework | Reference | 55 | Yes |
| AgentQMS/standards/tier2-framework/runtime/workflow-rules.yaml | tier2-framework | Runtime | 56 | Yes |
| AgentQMS/standards/tier2-framework/specs/anti-patterns.yaml | tier2-framework | Reference | 77 | Yes |
| AgentQMS/standards/tier2-framework/specs/api-contracts.yaml | tier2-framework | Constraint | 55 | Yes |
| AgentQMS/standards/tier2-framework/specs/component-interfaces.yaml | tier2-framework | Reference | 82 | Yes |
| AgentQMS/standards/tier2-framework/specs/hydra-v5-patterns-reference.yaml | tier2-framework | Reference | 80 | Yes |
| AgentQMS/standards/tier3-agents/bloat-detection-rules.yaml | tier3-agents | Reference | 53 | Yes |
| AgentQMS/standards/tier3-agents/config.yaml | tier3-agents | Constraint | 58 | Yes |
| AgentQMS/standards/tier3-agents/multi-agent-system.yaml | tier3-agents | Runtime | 56 | Yes |
| AgentQMS/standards/tier3-agents/quick-reference.yaml | tier3-agents | Constraint | 80 | Yes |
| AgentQMS/standards/tier3-agents/vlm-tools.yaml | tier3-agents | Constraint | 58 | Yes |
| AgentQMS/standards/tier4-workflows/experiment-workflow.yaml | tier4-workflows | Runtime | 78 | Yes |
| AgentQMS/standards/tier4-workflows/middleware-policies.yaml | tier4-workflows | Constraint | 79 | Yes |

---

## 2) Overlap & Purity Findings (Tier-first)

### Tier 2 (Most Mixed)
Primary overlap clusters:
1. **Hydra configuration cluster**
   - hydra-configuration-architecture.yaml (Runtime) and hydra-v5-rules.yaml (Constraints/Rules) live in configuration/ but mix architecture + enforcement.
   - hydra-v5-patterns-reference.yaml includes both patterns and anti-patterns.
2. **Config externalization cluster**
   - config-externalization-index/checklist/pattern all mix reference and discovery; unclear ownership and redundancy.
3. **OCR engine logic cluster**
   - preprocessing-logic/postprocessing-logic/model-management/pipeline-contracts blend runtime behavior with constraints.
4. **Runtime folder mixing reference**
   - coordinate-transforms, image-loading, viz-engine read like reference/constraint guidance rather than runtime orchestration.
5. **Specs folder mixing constraint**
   - anti-patterns and api-contracts include enforcement rules; overlap with constraints tier.

### Tier 1 (Core SST)
- artifact-types vs artifact_rules vs frontmatter-master show mixed reference + constraint + discovery.
- discovery-rules + workflow-detector + manifest overlap in discovery/registry function.
- workflow-requirements contains runtime guidance; overlap with workflow-rules in runtime tier.

### Tier 3 (Agent-facing)
- quick-reference vs config vs multi-agent-system blur reference vs runtime.
- bloat-detection-rules is enforcement content in a reference-looking location.

### Tier 4 (Workflows)
- experiment-workflow includes both runtime steps and policy constraints.
- middleware-policies lives in workflows but is primarily constraint (policy enforcement).

---

## 3) Refinement Proposals (Moves / Merges / Splits)

### High Priority (Score < 70)
1. **registry.yaml**
   - Reclassify as Discovery (registry index). Move to tier1-sst/discovery/ or create a dedicated discovery folder.
2. **frontmatter-master.yaml**
   - Split into: frontmatter-schema (Reference) + frontmatter-validation (Constraint).
3. **artifact-types.yaml**
   - Split into: artifact-type-reference (Reference) + artifact-type-rules (Constraint).
4. **workflow-detector.yaml**
   - Move to Discovery (keyword/routing). Keep only detection heuristics; push enforcement to constraints.
5. **tier2 config-externalization** (checklist/index/pattern)
   - Merge into a single Reference doc: config-externalization-guide.
   - Extract keyword routing into Discovery (config-externalization-discovery).
6. **tier2 specs/anti-patterns.yaml**
   - Move to constraints/ as anti-pattern-enforcement; keep a small reference stub in specs (examples only).
7. **tier2 specs/api-contracts.yaml**
   - Reframe as Reference; move enforcement to constraints if present.
8. **tier2 runtime/viz-engine.yaml, coordinate-transforms.yaml, image-loading.yaml**
   - Split into runtime flow (runtime/) and validation rules (constraints/). Any interface details should move to specs/.
9. **tier2 ocr-engine/preprocessing-logic.yaml**
   - Split runtime steps vs validation rules; move validation rules to constraints.
10. **tier3 bloat-detection-rules.yaml**
    - Move to constraints/ or rename to bloat-detection-criteria and ensure it is explicitly enforcement-only.

### Medium Priority (Score 70–79)
- validation-protocols.yaml: separate runtime flow vs enforcement protocols.
- workflow-requirements.yaml: move runtime steps to tier4 workflows; keep constraints in tier1.
- quick-reference.yaml (tier3): keep as reference; move any rules into constraints.
- middleware-policies.yaml: move to constraints tier (policy enforcement).

### Low Priority (>=80)
- file-placement-rules.yaml, prohibited-actions.yaml, performance-slas.yaml: already pure constraints.
- component-interfaces.yaml: mostly reference; keep but strip enforcement rules if any.

---

## 4) Cross-File Consolidation Targets

1. **Hydra guidance consolidation**
   - Merge hydra-v5-patterns-reference + hydra-configuration-architecture into a single Reference spec.
   - Keep hydra-v5-rules as constraints in constraints/.

2. **Config externalization consolidation**
   - Merge index + checklist + pattern into one Reference doc; move routing keywords to discovery rules.

3. **OCR runtime + contracts**
   - Move pipeline-contracts to specs/ (Reference). Keep runtime flow in runtime/.
   - Extract validation constraints from preprocessing/postprocessing into constraints/.

4. **Workflow governance**
   - Merge workflow-rules + workflow-requirements into a single constraints doc; route runtime flow to tier4 workflows.

---

## 5) Validation Phase (Planned)
- Registry sync simulation could not be executed due to CLI import error (AgentQMS.tools.utils.runtime missing). 
- Recommendation: run registry sync after refactor and re-scan to verify overlap reduction.

---

## 6) Priority Action List

### Priority 0 (Blocking)
- Split frontmatter-master.yaml and artifact-types.yaml to isolate schema vs enforcement.

### Priority 1 (Tier 2 cleanup)
- Move anti-patterns and hydra rules into constraints.
- Consolidate config-externalization documents.
- Split OCR runtime vs constraints.

### Priority 2 (Tier 3 & Tier 4)
- Relocate middleware policies to constraints.
- Normalize quick-reference to pure reference.

---

## 7) AI Impact Summary
- Current overlap ratio is high (most files flagged). This increases ambiguity for agent routing.
- After proposed splits and relocations, most files should reach >= 80 score by construction.
- Expected outcome: faster context detection and fewer misrouted agent workflows.
