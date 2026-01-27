# Tier‑2 Line‑Level Evidence Ledger (Session B)

Date: 2026-01-28
Status: In Progress
Scope: tier2-framework/configuration (initial subset)

> This ledger is partial and will be extended in subsequent Session B passes.

## Entry 1
- File: AgentQMS/standards/tier2-framework/configuration/config-externalization-checklist.yaml
- Section: artifact_templates_externalization (steps + validation)
- Line Range: L6–L65
- Current Category: Mixed (Runtime + Constraint)
- Intended Category: Split (Runtime steps, Constraint validation)
- Evidence: Implementation steps, time estimates, and test/validation checklist are execution and enforcement content inside configuration references.
- Proposed Action: Split into two files:
  - Runtime: config-externalization-workflow.yaml (steps/time)
  - Constraint: config-externalization-validation.yaml (validation checks)
- Target Path: AgentQMS/standards/tier2-framework/runtime/ (workflow), AgentQMS/standards/tier2-framework/constraints/ (validation)
- Registry Update Needed: Yes
- Validation Gate: Registry sync + discovery routing update + no off‑category content remains

## Entry 2
- File: AgentQMS/standards/tier2-framework/configuration/config-externalization-index.yaml
- Section: config_externalization.identified_configs
- Line Range: L6–L63
- Current Category: Discovery (index of config targets) but embedded in rule_set
- Intended Category: Discovery
- Evidence: The section is an inventory of externalized config targets and locations.
- Proposed Action: Move inventory to discovery/config-externalization-index.yaml and keep only index metadata there.
- Target Path: AgentQMS/standards/tier2-framework/discovery/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + discovery routing update

## Entry 3
- File: AgentQMS/standards/tier2-framework/configuration/config-externalization-index.yaml
- Section: implementation + pattern
- Line Range: L74–L91
- Current Category: Runtime/Reference (implementation steps + pattern)
- Intended Category: Split (Runtime steps, Reference pattern)
- Evidence: Implementation steps and pattern guidance mixed into index.
- Proposed Action: Move implementation steps to runtime workflow; move pattern guidance to specs reference.
- Target Path: runtime/config-externalization-workflow.yaml; specs/config-externalization-patterns.yaml
- Registry Update Needed: Yes
- Validation Gate: Registry sync + no off‑category content remains in index

## Entry 4
- File: AgentQMS/standards/tier2-framework/configuration/config-externalization-pattern.yaml
- Section: validation
- Line Range: L56–L62
- Current Category: Constraint inside reference pattern
- Intended Category: Constraint
- Evidence: Validation checklist belongs to enforcement, not reference.
- Proposed Action: Extract validation list to constraints/config-externalization-validation.yaml.
- Target Path: AgentQMS/standards/tier2-framework/constraints/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + constraint checks in place

## Entry 5
- File: AgentQMS/standards/tier2-framework/configuration/hydra-configuration-architecture.yaml
- Section: override_rules + defaults_ordering
- Line Range: L92–L128
- Current Category: Constraint inside architecture reference
- Intended Category: Constraint
- Evidence: Explicit enforcement rules and ordering constraints.
- Proposed Action: Move rule sections to constraints/hydra-v5-rules.yaml (or a constraints/hydra-override-rules.yaml).
- Target Path: AgentQMS/standards/tier2-framework/constraints/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + no enforcement rules remain in architecture reference

## Entry 6
- File: AgentQMS/standards/tier2-framework/configuration/hydra-configuration-architecture.yaml
- Section: validation + troubleshooting
- Line Range: L150–L205
- Current Category: Constraint/Runtime inside reference
- Intended Category: Constraint (validation) + Runtime (troubleshooting workflows)
- Evidence: Test commands and troubleshooting procedures are enforcement/execution content.
- Proposed Action: Split validation into constraints/hydra-validation.yaml and troubleshooting into runtime/hydra-troubleshooting.yaml.
- Target Path: AgentQMS/standards/tier2-framework/constraints/ and runtime/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + no execution steps remain in reference

## Entry 7
- File: AgentQMS/standards/tier2-framework/configuration/hydra-v5-rules.yaml
- Section: design_patterns
- Line Range: L184–L201
- Current Category: Reference inside constraints
- Intended Category: Reference
- Evidence: Patterns describe architecture guidance rather than enforcement.
- Proposed Action: Move design_patterns to specs/hydra-v5-patterns-reference.yaml or a dedicated specs/hydra-design-patterns.yaml.
- Target Path: AgentQMS/standards/tier2-framework/specs/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + patterns no longer embedded in rule set

## Entry 8
- File: AgentQMS/standards/tier2-framework/configuration/hydra-v5-rules.yaml
- Section: directory_responsibilities
- Line Range: L96–L183
- Current Category: Mixed (Reference taxonomy + Constraint rules)
- Intended Category: Reference (architecture) with constraints isolated
- Evidence: Directory responsibilities define architecture structure and include allowed/forbidden rules.
- Proposed Action: Split into specs/hydra-directory-architecture.yaml (structure) and constraints/hydra-directory-rules.yaml (forbidden/allowed enforcement).
- Target Path: AgentQMS/standards/tier2-framework/specs/ and constraints/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + off‑category content removed

## Entry 9
- File: AgentQMS/standards/tier2-framework/core-infra/agent-architecture.yaml
- Section: rules
- Line Range: L49–L53
- Current Category: Constraint embedded in architecture reference
- Intended Category: Constraint
- Evidence: Explicit “MUST” rules (enforcement) are bundled into an architecture description.
- Proposed Action: Move rules to constraints/agent-architecture-rules.yaml; keep components and protocol structure in specs.
- Target Path: AgentQMS/standards/tier2-framework/constraints/ and specs/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + architecture spec remains reference-only

## Entry 10
- File: AgentQMS/standards/tier2-framework/core-infra/agent-feedback-protocol.yaml
- Section: mechanisms + protocol
- Line Range: L10–L37
- Current Category: Mixed (Constraint enforcement + Runtime protocol)
- Intended Category: Split (Constraint enforcement, Runtime response steps)
- Evidence: Compliance rules are enforcement; protocol steps are execution flow.
- Proposed Action: Split into constraints/agent-feedback-rules.yaml and runtime/agent-feedback-protocol.yaml.
- Target Path: AgentQMS/standards/tier2-framework/constraints/ and runtime/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + separate enforcement vs flow

## Entry 11
- File: AgentQMS/standards/tier2-framework/ocr-engine/model-management.yaml
- Section: critical_logic
- Line Range: L16–L51
- Current Category: Reference (component spec) but stored outside specs
- Intended Category: Reference (specs)
- Evidence: Describes component responsibilities and logic; no runtime orchestration.
- Proposed Action: Move file to specs/ocr-model-management.yaml.
- Target Path: AgentQMS/standards/tier2-framework/specs/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + specs-only placement

## Entry 12
- File: AgentQMS/standards/tier2-framework/ocr-engine/pipeline-contracts.yaml
- Section: data_contract.types
- Line Range: L19–L83
- Current Category: Reference (contracts)
- Intended Category: Reference (specs)
- Evidence: Pure data contracts and interface definitions.
- Proposed Action: Move file to specs/ocr-pipeline-contracts.yaml.
- Target Path: AgentQMS/standards/tier2-framework/specs/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + contracts only in specs

## Entry 13
- File: AgentQMS/standards/tier2-framework/ocr-engine/postprocessing-logic.yaml
- Section: critical_logic
- Line Range: L15–L48
- Current Category: Reference (algorithm spec) but stored outside specs
- Intended Category: Reference (specs)
- Evidence: Algorithm description and inputs/outputs; not orchestration.
- Proposed Action: Move file to specs/ocr-postprocessing-spec.yaml.
- Target Path: AgentQMS/standards/tier2-framework/specs/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + specs-only placement

## Entry 14
- File: AgentQMS/standards/tier2-framework/ocr-engine/preprocessing-logic.yaml
- Section: critical_logic
- Line Range: L15–L42
- Current Category: Reference (algorithm spec) but stored outside specs
- Intended Category: Reference (specs)
- Evidence: Algorithm steps and configuration details; no runtime orchestration.
- Proposed Action: Move file to specs/ocr-preprocessing-spec.yaml.
- Target Path: AgentQMS/standards/tier2-framework/specs/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + specs-only placement

## Entry 15
- File: AgentQMS/standards/tier2-framework/runtime/coordinate-transforms.yaml
- Section: critical_logic
- Line Range: L15–L34
- Current Category: Reference (component spec) inside runtime
- Intended Category: Reference (specs)
- Evidence: Describes algorithm behavior and data contract, not orchestration.
- Proposed Action: Move file to specs/ocr-coordinate-transforms.yaml.
- Target Path: AgentQMS/standards/tier2-framework/specs/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + runtime folder reserved for orchestration

## Entry 16
- File: AgentQMS/standards/tier2-framework/runtime/image-loading.yaml
- Section: critical_logic
- Line Range: L15–L38
- Current Category: Reference/Constraint mix inside runtime
- Intended Category: Split (spec vs constraint)
- Evidence: Format support and EXIF correction are spec; “Must reject” and limits are constraints.
- Proposed Action: Split into specs/ocr-image-loading-spec.yaml and constraints/ocr-image-loading-rules.yaml.
- Target Path: AgentQMS/standards/tier2-framework/specs/ and constraints/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + no enforcement in specs

## Entry 17
- File: AgentQMS/standards/tier2-framework/runtime/viz-engine.yaml
- Section: critical_logic
- Line Range: L15–L35
- Current Category: Reference (component spec) inside runtime
- Intended Category: Reference (specs)
- Evidence: Describes drawing logic and output format; not orchestration.
- Proposed Action: Move file to specs/ocr-viz-engine-spec.yaml.
- Target Path: AgentQMS/standards/tier2-framework/specs/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + runtime folder reserved for orchestration

## Entry 18
- File: AgentQMS/standards/tier2-framework/runtime/workflow-rules.yaml
- Section: task_types + artifact_triggers + command_templates
- Line Range: L7–L103
- Current Category: Discovery/Constraint (routing and triggers) in runtime
- Intended Category: Discovery
- Evidence: Maps task types to context bundles and triggers; no runtime flow.
- Proposed Action: Move to discovery/workflow-rules.yaml and mark as routing metadata.
- Target Path: AgentQMS/standards/tier2-framework/discovery/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + discovery routing updated

## Entry 19
- File: AgentQMS/standards/tier2-framework/specs/anti-patterns.yaml
- Section: enforcement + code_review_checklist
- Line Range: L34–L293
- Current Category: Constraint inside specs
- Intended Category: Constraint
- Evidence: Enforcement tools, CI checks, and fail conditions are rule enforcement.
- Proposed Action: Move to constraints/anti-patterns.yaml; keep any descriptive reference in specs if needed.
- Target Path: AgentQMS/standards/tier2-framework/constraints/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + constraints-only enforcement

## Entry 20
- File: AgentQMS/standards/tier2-framework/specs/component-interfaces.yaml
- Section: critical_logic + data_contract.implementation_rules
- Line Range: L17–L57
- Current Category: Mixed (Reference + Constraint)
- Intended Category: Split
- Evidence: “MUST NOT” and inheritance rules are enforcement embedded in interface description.
- Proposed Action: Move enforcement rules to constraints/component-interface-rules.yaml; keep interface descriptions in specs.
- Target Path: AgentQMS/standards/tier2-framework/specs/ and constraints/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + specs-only narrative

## Entry 21
- File: AgentQMS/standards/tier2-framework/specs/hydra-v5-patterns-reference.yaml
- Section: validation_checklist
- Line Range: L253–L273
- Current Category: Constraint inside reference
- Intended Category: Constraint
- Evidence: Validation checklist is enforcement content embedded in reference guide.
- Proposed Action: Move checklist to constraints/hydra-v5-validation.yaml.
- Target Path: AgentQMS/standards/tier2-framework/constraints/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + validation isolated

## Entry 22
- File: AgentQMS/standards/tier2-framework/debugging-sessions.yaml
- Section: session_structure + workflow
- Line Range: L9–L88
- Current Category: Runtime workflow
- Intended Category: Runtime (workflows)
- Evidence: Defines phases, steps, and execution flow for bug sessions.
- Proposed Action: Move to runtime/debugging-session-workflow.yaml (or tier4 workflows if standardized there).
- Target Path: AgentQMS/standards/tier2-framework/runtime/ or tier4-workflows/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + workflow placement clarified

## Entry 23
- File: AgentQMS/standards/tier2-framework/ml-frameworks.yaml
- Section: heavy_modules
- Line Range: L6–L20
- Current Category: Discovery (catalog list) but stored in framework root
- Intended Category: Discovery
- Evidence: Pure list used for detection/routing of heavy imports.
- Proposed Action: Move to discovery/heavy-modules.yaml.
- Target Path: AgentQMS/standards/tier2-framework/discovery/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + discovery rules updated

## Entry 24
- File: AgentQMS/standards/tier2-framework/configuration/configuration-standards.yaml
- Section: rules
- Line Range: L12–L41
- Current Category: Constraint (rules) inside configuration
- Intended Category: Constraint
- Evidence: All content is enforcement (MUST/NEVER rules and sanctioned patterns).
- Proposed Action: Move to constraints/omegaconf-handling-rules.yaml; leave configuration folder for reference architecture only.
- Target Path: AgentQMS/standards/tier2-framework/constraints/
- Registry Update Needed: Yes
- Validation Gate: Registry sync + configuration folder contains no enforcement rules
