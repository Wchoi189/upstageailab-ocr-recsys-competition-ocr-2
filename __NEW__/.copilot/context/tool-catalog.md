# AgentQMS Tool Catalog

Auto-generated tool registry for AI agent discovery.

## Tools by Category

### Audit

| Tool | Description | CLI | Usage |
|---|---|---|---|
| **audit_generator** | Shim for the audit document generator. | ✓ | `AgentQMS/agent_tools/audit/audit_generator.py` |
| **audit_validator** | Shim for the audit document validator. | ✓ | `AgentQMS/agent_tools/audit/audit_validator.py` |
| **checklist_tool** | Shim for the audit checklist tool. | ✓ | `AgentQMS/agent_tools/audit/checklist_tool.py` |
### Compliance

| Tool | Description | CLI | Usage |
|---|---|---|---|
| **monitor_artifacts** | Artifact Monitoring and Compliance System | ✓ | `AgentQMS/agent_tools/compliance/monitor_artifacts.py` |
| **validate_artifacts** | Artifact Validation Script for AI Agents | ✓ | `AgentQMS/agent_tools/compliance/validate_artifacts.py` |
| **validate_boundaries** | Validate AgentQMS framework boundaries. | ✓ | `AgentQMS/agent_tools/compliance/validate_boundaries.py` |
### Core

| Tool | Description | CLI | Usage |
|---|---|---|---|
| **artifact_templates** | Shim for artifact templates. | ✗ | `AgentQMS/agent_tools/core/artifact_templates.py` |
| **artifact_workflow** | AI Agent Artifact Workflow Integration | ✓ | `AgentQMS/agent_tools/core/artifact_workflow.py` |
| **context_bundle** | Context Bundle Generator Core | ✓ | `AgentQMS/agent_tools/core/context_bundle.py` |
| **discover** | Agent Tools Discovery Helper | ✓ | `AgentQMS/agent_tools/core/discover.py` |
| **plugin_loader** | Plugin Loader Shim | ✓ | `AgentQMS/agent_tools/core/plugin_loader.py` |
| **workflow_detector** | Workflow Detection and Auto-Suggestion | ✓ | `AgentQMS/agent_tools/core/workflow_detector.py` |
### Documentation

| Tool | Description | CLI | Usage |
|---|---|---|---|
| **auto_generate_index** | Shim for handbook index generation/validation. | ✓ | `AgentQMS/agent_tools/documentation/auto_generate_index.py` |
| **validate_links** | Shim for documentation link validation. | ✓ | `AgentQMS/agent_tools/documentation/validate_links.py` |
| **validate_manifest** | Shim for AI handbook manifest validation. | ✓ | `AgentQMS/agent_tools/documentation/validate_manifest.py` |
### Utilities

| Tool | Description | CLI | Usage |
|---|---|---|---|
| **adapt_project** | Canonical wrapper for the project adaptation script. | ✓ | `AgentQMS/agent_tools/utilities/adapt_project.py` |
| **cli** | Canonical CLI entrypoint for development/debug tracking and experiments. | ✓ | `AgentQMS/agent_tools/utilities/tracking/cli.py` |
| **config** | Framework configuration loader for AgentQMS. | ✗ | `AgentQMS/agent_tools/utils/config.py` |
| **get_context** | get_context tool | ✓ | `AgentQMS/agent_tools/utilities/get_context.py` |
| **paths** | Path resolution helpers for AgentQMS (agent_tools canonical). | ✗ | `AgentQMS/agent_tools/utils/paths.py` |
| **runtime** | Runtime helpers for initializing AgentQMS tooling (agent_tools canonical). | ✗ | `AgentQMS/agent_tools/utils/runtime.py` |

## Workflows (Makefile Targets)

| Workflow | Description | Category |
|---|---|---|
| `help` | Show agent help message | workflow |
| `discover` | Discover all available agent tools | workflow |
| `status` | Check overall system status | workflow |
| `create-plan` | Create implementation plan (usage: make create-plan NAME=my-plan TITLE="My Plan") | artifact_creation |
| `create-assessment` | Create assessment (usage: make create-assessment NAME=my-assessment TITLE="My Assessment") | artifact_creation |
| `create-design` | Create design document (usage: make create-design NAME=my-design TITLE="My Design") | artifact_creation |
| `create-research` | Create research document (usage: make create-research NAME=my-research TITLE="My Research") | artifact_creation |
| `create-template` | Create template (usage: make create-template NAME=my-template TITLE="My Template") | artifact_creation |
| `create-bug-report` | Create bug report (usage: make create-bug-report NAME=my-bug TITLE="My Bug Report") | artifact_creation |
| `validate` | Validate all artifacts | validation |
| `validate-file` | Validate specific file (usage: make validate-file FILE=path/to/file.md) | validation |
| `validate-naming` | Check naming conventions only | validation |
| `boundary` | Validate framework/project boundaries | workflow |
| `compliance` | Check compliance status | validation |
| `compliance-fix` | Fix compliance issues automatically (DEPRECATED) | validation |
| `compliance-dashboard` | Open enhanced compliance dashboard (DEPRECATED) | validation |
| `docs-generate` | Generate documentation index | documentation |
| `docs-regenerate` | Regenerate all documentation | documentation |
| `docs-update-indexes` | Update artifact indexes | documentation |
| `docs-validate-links` | Validate documentation links | documentation |
| `docs-validate-manifest` | Validate documentation manifest | documentation |
| `docs-validate-metadata` | Validate documentation metadata | documentation |
| `context` | Generate context bundle for task (usage: make context TASK="task description") | context_loading |
| `context-development` | Get development context bundle | context_loading |
| `context-docs` | Get documentation context bundle | context_loading |
| `context-debug` | Get debugging context bundle | context_loading |
| `context-plan` | Get planning context bundle | context_loading |
| `context-list` | List all available context bundles | context_loading |
| `audit-init` | Initialize audit (usage: make audit-init FRAMEWORK="Framework Name" DATE="2025-11-09" SCOPE="Scope description") | audit |
| `audit-validate` | Validate audit documents | audit |
| `audit-checklist-generate` | Generate checklist for phase (usage: make audit-checklist-generate PHASE="discovery") | audit |
| `audit-checklist-report` | Generate audit progress report | audit |
| `docs-validate-templates` | Validate documentation templates | documentation |
| `docs-validate-ui-schema` | Validate UI schema documentation | documentation |
| `docs-validate-ai` | Validate AI documentation (DEPRECATED) | documentation |
| `feedback-report` | Generate feedback report | workflow |
| `feedback-issue` | Report documentation issue (usage: make feedback-issue ISSUE="description" FILE="path") | workflow |
| `feedback-suggest` | Suggest improvement (usage: make feedback-suggest AREA="area" CURRENT="current" CHANGE="suggested" RATIONALE="reason") | workflow |
| `quality-check` | Check documentation quality | workflow |
| `quality-check-output` | Check documentation quality and save to file | workflow |
| `ast-analyze` | Run AST analysis on codebase (usage: make ast-analyze [TARGET=path/to/code]) | workflow |
| `ast-generate-tests` | Generate test scaffolds using AST analysis (usage: make ast-generate-tests TARGET=path/to/module.py) | workflow |
| `ast-extract-docs` | Extract documentation from code using AST (usage: make ast-extract-docs TARGET=path/to/module.py) | workflow |
| `ast-check-quality` | Check code quality using AST analysis (usage: make ast-check-quality [TARGET=path/to/code]) | workflow |
| `context-bundle` | Get specific context bundle (usage: make context-bundle BUNDLE=bundle-name) | context_loading |
| `workflow-create` | Complete workflow: create artifact, validate, update indexes | workflow |
| `workflow-validate` | Complete validation workflow: validate, fix, monitor | workflow |
| `workflow-docs` | Complete documentation workflow: generate, validate, update | workflow |
| `track-init` | Initialize tracking SQLite DB | workflow |
| `plan-new` | Create plan (usage: make plan-new TITLE="..." OWNER="me" [KEY=slug]) | workflow |
| `exp-new` | Create experiment (usage: make exp-new TITLE="..." OBJECTIVE="..." OWNER="me" [KEY=slug]) | workflow |
| `exp-export` | Export experiment runs to CSV (usage: make exp-export OUT=data/ops/experiment_runs.csv) | workflow |
| `clean` | Clean up temporary files and caches | workflow |
| `clean-feedback` | Clean up old feedback files | workflow |
| `dev-setup` | Setup development environment | workflow |
| `dev-test` | Test all agent tools | workflow |
| `info-tools` | Show detailed tool information | workflow |
| `info-stats` | Show project statistics | workflow |
| `tool-catalog` | Generate tool catalog and scripts index | workflow |
| `export` | Export framework to project-agnostic package (usage: make export OUTPUT=export_package/) | workflow |
| `export-dry-run` | Dry run export (see what would be exported) (usage: make export-dry-run OUTPUT=export_package/) | workflow |
| `export-validate` | Export framework with validation (usage: make export-validate OUTPUT=export_package/) | workflow |
| `version-bump` | Bump project version (usage: make version-bump VERSION=0.8.0 NOTES="..." DATE=YYYY-MM-DDTHH:MM:SSZ) | workflow |
| `docs-deprecate` | Deprecate docs based on project_version.yaml (add --dry-run with DRY_RUN=1) | documentation |
| `changelog-draft` | Generate changelog draft from tracking DB and git log | workflow |
| `changelog-preview` | Preview changelog draft without writing file | workflow |
| `version` | Show system version information | workflow |