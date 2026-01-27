# Tier-1/3/4 Line-Level Evidence Ledger (Session C)

Date: 2026-01-28
Status: In Progress
Scope: tier1-sst, tier3-agents, tier4-workflows

## Inventory Summary (Tier 1/3/4)

| Path | Tier | Dominant | Overlap | Note |
|---|---|---|---|---|
| AgentQMS/standards/tier1-sst/artifact-types.yaml | 1 | Reference | Yes | Reference + enforcement + validation commands mixed |
| AgentQMS/standards/tier1-sst/artifact_rules.yaml | 1 | Reference | No | Mostly placement rules (constraint-lite) |
| AgentQMS/standards/tier1-sst/discovery-rules.yaml | 1 | Reference | Yes | Discovery routing stored in SST |
| AgentQMS/standards/tier1-sst/file-placement-rules.yaml | 1 | Constraint | No | Pure placement rules |
| AgentQMS/standards/tier1-sst/frontmatter-master.yaml | 1 | Reference | Yes | Schema + validation rules mixed |
| AgentQMS/standards/tier1-sst/manifest.yaml | 1 | Discovery | Yes | Utility discovery registry |
| AgentQMS/standards/tier1-sst/naming-conventions.yaml | 1 | Constraint | Yes | Enforcement + examples |
| AgentQMS/standards/tier1-sst/prohibited-actions.yaml | 1 | Constraint | No | Pure enforcement |
| AgentQMS/standards/tier1-sst/system-architecture.yaml | 1 | Constraint | Yes | Runtime inventory in SST |
| AgentQMS/standards/tier1-sst/validation-protocols.yaml | 1 | Constraint | Yes | Runtime commands embedded |
| AgentQMS/standards/tier1-sst/workflow-detector.yaml | 1 | Runtime | Yes | Discovery + workflow routing |
| AgentQMS/standards/tier1-sst/workflow-requirements.yaml | 1 | Constraint | Yes | Runtime steps embedded |
| AgentQMS/standards/tier3-agents/bloat-detection-rules.yaml | 3 | Reference | Yes | Constraints + workflow automation |
| AgentQMS/standards/tier3-agents/config.yaml | 3 | Constraint | Yes | Global policy in persona config |
| AgentQMS/standards/tier3-agents/multi-agent-system.yaml | 3 | Runtime | Yes | Operational runbook |
| AgentQMS/standards/tier3-agents/quick-reference.yaml | 3 | Constraint | Yes | Generic compliance checklist |
| AgentQMS/standards/tier3-agents/vlm-tools.yaml | 3 | Constraint | Yes | CLI workflow guide |
| AgentQMS/standards/tier4-workflows/experiment-workflow.yaml | 4 | Runtime | Yes | Includes discovery/constraints |
| AgentQMS/standards/tier4-workflows/middleware-policies.yaml | 4 | Constraint | Yes | Policy rules inside workflow tier |

## Entry 1
- File: AgentQMS/standards/tier1-sst/discovery-rules.yaml
- Section: file header + keyword mappings
- Line Range: L1–L118
- Current Category: Discovery catalog (tool catalog) stored in Tier 1
- Intended Category: Discovery (Tier 2) — routing metadata
- Evidence: Metadata declares `type: tool_catalog`, `tier: 2` and the body lists keyword-to-bundle mappings, which are discovery/routing rules, not constitutional SST guidance.
- Proposed Action: Move the file to tier2-framework/discovery/ and keep SST for constitutional naming/placement only.
- Target Path: AgentQMS/standards/tier2-framework/discovery/discovery-rules.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + discovery routing coverage check

## Entry 2
- File: AgentQMS/standards/tier1-sst/system-architecture.yaml
- Section: applications + inference_engine + config_system
- Line Range: L9–L60
- Current Category: Runtime/infrastructure inventory embedded in Tier 1 quick reference
- Intended Category: Reference/Discovery in Tier 2 (infrastructure overview) with workflows separated to Tier 4
- Evidence: Lists active apps, hydra config directories, and runner syntax (`python runners/train.py domain=X`) which are operational details rather than SST-level principles.
- Proposed Action: Relocate operational inventory to tier2-framework (architecture reference) and keep only constitutional principles in SST. Split any runnable commands into a Tier 4 workflow runbook.
- Target Path: AgentQMS/standards/tier2-framework/specs/system-architecture.yaml (reference) + tier4-workflows/ocr-config-runbook.yaml (workflow commands)
- Registry Update Needed: Yes
- Validation Gate: Registry sync + no runtime commands in SST

## Entry 10
- File: AgentQMS/standards/tier1-sst/artifact-types.yaml
- Section: artifact_type_system + validation
- Line Range: L35–L109
- Current Category: Mixed (Reference + Constraint)
- Intended Category: Split (Reference catalog vs Constraint validation)
- Evidence: File includes allowed/prohibited type registry (reference) plus validation commands and enforcement rules.
- Proposed Action: Split into a reference catalog (allowed/prohibited types) and a constraints file for validation/enforcement.
- Target Path: AgentQMS/standards/tier1-sst/specs/artifact-types-reference.yaml + AgentQMS/standards/tier1-sst/constraints/artifact-type-validation.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + enforcement isolated from reference catalog

## Entry 11
- File: AgentQMS/standards/tier1-sst/frontmatter-master.yaml
- Section: fields + validation
- Line Range: L9–L131
- Current Category: Mixed (Schema reference + validation rules)
- Intended Category: Split (Reference schema vs Constraint validation)
- Evidence: Field definitions are schema reference; validation modes and required fields are enforcement rules.
- Proposed Action: Extract validation rules to tier1-sst/constraints/frontmatter-validation.yaml; keep schema spec in tier1-sst/specs/frontmatter-schema.yaml.
- Target Path: AgentQMS/standards/tier1-sst/specs/frontmatter-schema.yaml + AgentQMS/standards/tier1-sst/constraints/frontmatter-validation.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + schema reference contains no enforcement

## Entry 12
- File: AgentQMS/standards/tier1-sst/workflow-detector.yaml
- Section: task_types + artifact_triggers + command_templates
- Line Range: L9–L105
- Current Category: Discovery/routing logic stored in SST
- Intended Category: Discovery (Tier 2) with workflow routing metadata
- Evidence: Maps tasks to context bundles and provides command templates; this is routing logic, not SST-level law.
- Proposed Action: Move to tier2-framework/discovery/workflow-detector.yaml and keep SST minimal.
- Target Path: AgentQMS/standards/tier2-framework/discovery/workflow-detector.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + discovery routing centralized

## Entry 13
- File: AgentQMS/standards/tier1-sst/workflow-requirements.yaml
- Section: required_workflows + discovery_workflow
- Line Range: L12–L43
- Current Category: Runtime workflow commands embedded in Tier 1 constraints
- Intended Category: Split (Tier 1 constraint requirements + Tier 4 workflow runbook)
- Evidence: Contains exact command invocations (make create-*, make discover/status) and runtime sequences.
- Proposed Action: Move command execution sequences to tier4-workflows/validation-runbook.yaml; keep only constraint requirements in Tier 1.
- Target Path: AgentQMS/standards/tier4-workflows/validation-runbook.yaml + tier1-sst/constraints/workflow-requirements.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + Tier 1 contains rules only

## Entry 14
- File: AgentQMS/standards/tier1-sst/validation-protocols.yaml
- Section: validation_commands + validation_workflow
- Line Range: L11–L61
- Current Category: Runtime command procedures stored as constraints
- Intended Category: Split (constraints vs workflow procedures)
- Evidence: Command sequences and failure handling are operational workflows, not SST rules.
- Proposed Action: Move procedural steps to tier4-workflows/validation-runbook.yaml; keep policy statements in tier1 constraints.
- Target Path: AgentQMS/standards/tier4-workflows/validation-runbook.yaml + tier1-sst/constraints/validation-policies.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + no runtime steps in Tier 1

## Entry 15
- File: AgentQMS/standards/tier1-sst/manifest.yaml
- Section: utility registry + decision_tree
- Line Range: L6–L243
- Current Category: Discovery registry stored in SST
- Intended Category: Discovery (Tier 2) utility registry
- Evidence: File is a discovery manifest (paths, utilities, decision tree) rather than constitutional constraints.
- Proposed Action: Move to tier2-framework/discovery/utility-scripts-manifest.yaml; keep Tier 1 for cross-cutting laws only.
- Target Path: AgentQMS/standards/tier2-framework/discovery/utility-scripts-manifest.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + discovery routing aligned

## Entry 16
- File: AgentQMS/standards/tier3-agents/config.yaml
- Section: critical_protocols + tool_access + prohibited_actions
- Line Range: L18–L67
- Current Category: Constraint/workflow policy embedded in agent persona
- Intended Category: Split (persona identity vs global constraints)
- Evidence: Validation commands, prohibited actions, and tool access are global operational policies rather than Qwen-specific identity.
- Proposed Action: Extract generic policies to tier1-sst constraints; keep only agent_identity and persona-specific parameters in Tier 3.
- Target Path: AgentQMS/standards/tier1-sst/constraints/agent-policy.yaml (generic) + tier3-agents/qwen-identity.yaml (persona)
- Registry Update Needed: Yes
- Validation Gate: Registry sync + Tier 3 remains identity-only

## Entry 3
- File: AgentQMS/standards/tier3-agents/bloat-detection-rules.yaml
- Section: detection_criteria + automated_scanning + manual_review
- Line Range: L10–L193
- Current Category: Code-quality constraint/workflow stored under Agents (Tier 3)
- Intended Category: Constraint (Tier 2 coding) + Workflow (Tier 4 automation)
- Evidence: Contains thresholds, CLI invocations (radon, wily, adt, uv run), and GitHub Actions steps — these are code-quality enforcement/workflows, not agent persona configuration.
- Proposed Action: Move constraint thresholds to tier2-framework/coding/constraints/bloat-detection-rules.yaml and the scanning/automation steps to tier4-workflows/bloat-detection-workflow.yaml; keep only agent-persona configs in Tier 3.
- Target Path: AgentQMS/standards/tier2-framework/constraints/bloat-detection-rules.yaml and AgentQMS/standards/tier4-workflows/bloat-detection-workflow.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + separation of constraints vs workflows

## Entry 4
- File: AgentQMS/standards/tier3-agents/multi-agent-system.yaml
- Section: core_components + usage_pattern
- Line Range: L20–L49
- Current Category: Runtime orchestration guide inside Agents tier
- Intended Category: Workflow (Tier 4) or Framework reference (Tier 2 infra)
- Evidence: Provides RabbitMQ transport details and runnable commands (`python -m ocr.agents.coordinator_agent`, dispatch_task) which are operational playbooks, not persona identity.
- Proposed Action: Move the orchestration instructions to tier4-workflows/multi-agent-runbook.yaml; keep any shared architecture reference in tier2-framework/agent-infra/.
- Target Path: AgentQMS/standards/tier4-workflows/multi-agent-runbook.yaml (workflow) + tier2-framework/agent-infra/multi-agent-architecture.yaml (reference)
- Registry Update Needed: Yes
- Validation Gate: Registry sync + no runtime commands left in Tier 3 persona files

## Entry 5
- File: AgentQMS/standards/tier3-agents/quick-reference.yaml (AG-004)
- Section: mandatory_rules + common_commands
- Line Range: L1–L26
- Current Category: Global compliance checklist stored as an agent quick reference
- Intended Category: SST constraints (Tier 1) or replace with agent-specific persona guidance
- Evidence: Content restates naming/placement/validation rules (make create-*, make validate) but contains no Gemini- or agent-specific behavior, so it is not a persona configuration.
- Proposed Action: Either relocate the general rules to tier1-sst/validation-protocols.yaml or rewrite the file with actual Gemini persona settings and move generic rules to SST.
- Target Path: AgentQMS/standards/tier1-sst/validation-protocols.yaml (for generic rules) OR retain in Tier 3 only after adding persona-specific content
- Registry Update Needed: Yes (if relocated)
- Validation Gate: Registry sync + persona files must be identity-specific

## Entry 6
- File: AgentQMS/standards/tier3-agents/vlm-tools.yaml
- Section: core_command + analysis_modes + usage_pattern
- Line Range: L1–L83
- Current Category: Runtime CLI workflow for VLM tooling inside Agents tier
- Intended Category: Workflow (Tier 4) with tool reference in Tier 2
- Evidence: Specifies CLI invocations (`uv run python -m AgentQMS.vlm.cli.analyze_image_defects ...`), modes, and experiment integration paths—this is a workflow guide, not agent persona data.
- Proposed Action: Move the CLI workflow to tier4-workflows/vlm-tools-workflow.yaml and keep any static tool reference in tier2-framework/tool-catalog.yaml.
- Target Path: AgentQMS/standards/tier4-workflows/vlm-tools-workflow.yaml (workflow) + tier2-framework/tool-catalog.yaml (reference link)
- Registry Update Needed: Yes
- Validation Gate: Registry sync + no runtime workflows under Tier 3

## Entry 7
- Files: AgentQMS/standards/tier3-agents/claude/validation.sh; .../copilot/validation.sh; .../cursor/validation.sh; .../gemini/validation.sh
- Section: shell scripts (self-validation)
- Line Range: L1–L28 (each)
- Current Category: Executable runtime scripts stored under Agents standards
- Intended Category: Workflow tooling (Tier 4) or tools/validation scripts outside standards
- Evidence: Bash scripts invoke compliance-checker and dependency checks; these are operational validators, not agent persona definitions, and they live alongside missing config/quick-reference files they expect.
- Proposed Action: Move scripts to tier4-workflows/pre-commit-hooks/agents/ or AgentQMS/tools/validation/, update registry to stop treating them as standards, and ensure agent personas use YAML-only definitions.
- Target Path: AgentQMS/standards/tier4-workflows/pre-commit-hooks/agents/validation.sh (per agent) or AgentQMS/tools/validation/
- Registry Update Needed: Yes (remove from standards routing; add workflow location)
- Validation Gate: Registry sync + no executable scripts in Tier 3 standards

## Entry 8
- File: AgentQMS/standards/tier4-workflows/middleware-policies.yaml
- Section: items (Redundancy Prevention, Python Execution Compliance, Documentation Standards, File System Integrity)
- Line Range: L1–L31
- Current Category: Constraint rule set stored under Workflows
- Intended Category: Constraint (Tier 2 framework/server) with any execution hooks in Tier 4
- Evidence: File codifies enforcement rules (“Must not use plain python”, “frontmatter must include ads_version”) rather than procedural workflow steps; it functions as policy, not a workflow definition.
- Proposed Action: Relocate policy rules to tier2-framework/constraints/middleware-policies.yaml and keep Tier 4 for execution scripts/hooks referencing those policies.
- Target Path: AgentQMS/standards/tier2-framework/constraints/middleware-policies.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + policies separated from workflows

## Entry 17
- File: AgentQMS/standards/tier4-workflows/experiment-workflow.yaml
- Section: triggers + rules
- Line Range: L34–L44
- Current Category: Discovery/Constraint content inside workflow definition
- Intended Category: Split (Discovery triggers + constraint policies)
- Evidence: Trigger routing and policy rules (“Never create artifacts manually”) are not workflow steps.
- Proposed Action: Move trigger routing to tier2-framework/discovery/experiment-workflow-triggers.yaml and enforcement rules to tier1 constraints.
- Target Path: AgentQMS/standards/tier2-framework/discovery/experiment-workflow-triggers.yaml + tier1-sst/constraints/experiment-workflow-policies.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + workflow file contains only execution steps

## Entry 18
- File: AgentQMS/standards/tier4-workflows/compliance-reporting/generate-compliance-report.py
- Section: sys.path insertion + python3 execution path
- Line Range: L13–L121
- Current Category: Workflow implementation script stored in standards (runtime logic)
- Intended Category: Tooling (AgentQMS/tools) with workflow spec in YAML
- Evidence: Uses `sys.path.insert` (prohibited by Tier 1) and invokes `python3` directly instead of `uv run python`.
- Proposed Action: Move script to AgentQMS/tools/compliance/generate_report.py; replace sys.path manipulation with path utilities; update tier4 workflow YAML to call `uv run python`.
- Target Path: AgentQMS/tools/compliance/generate_report.py + tier4-workflows/compliance-reporting.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + policy compliance (no sys.path, use uv run python)

## Entry 9
- File: AgentQMS/standards/tier4-workflows/compliance-reporting/latest-report.txt
- Section: generated compliance dashboard output
- Line Range: L1–L118
- Current Category: Generated artifact stored inside standards
- Intended Category: Artifact output (docs/artifacts) or reports directory outside standards
- Evidence: File is a generated report (timestamped 2026-01-28) summarizing compliance metrics; storing generated outputs in standards creates drift and violates “no passive refactors”.
- Proposed Action: Move report to docs/artifacts/compliance_reports/ (or experiment_manager outputs) and keep standards tree source-only; add ignore rule to prevent regenerated files in standards.
- Target Path: docs/artifacts/compliance_reports/latest-report.txt
- Registry Update Needed: Yes (remove report from standards routing; add report location to artifacts index if needed)
- Validation Gate: Registry sync + standards tree remains source-only (no generated outputs)

## Entry 19
- File: AgentQMS/standards/registry.yaml
- Section: AG-006 and FW-037 registry entries
- Line Range: L27–L34 and L158–L164
- Current Category: Registry integrity mismatch
- Intended Category: Discovery registry with accurate tier/path alignment
- Evidence: AG-006 is marked `tier: 3` but points to a Tier 2 framework path; FW-037 points to tier1-sst/discovery-rules.yaml while tagged as Tier 2 discovery.
- Proposed Action: Align registry tiers and paths after relocations from Entries 1/12/15/16.
- Target Path: AgentQMS/standards/registry.yaml updates
- Registry Update Needed: Yes
- Validation Gate: Registry sync + no tier/path mismatches
