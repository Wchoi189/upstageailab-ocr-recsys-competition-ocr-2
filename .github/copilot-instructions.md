## AgentQMS Artifact Generation Workflow

This project uses AgentQMS (Agent Quality Management System) for structured artifact generation. You MUST integrate artifact creation into your workflow when appropriate.

### When to Generate Artifacts

- **implementation_plan**: When planning or implementing new features, changes, or refactoring. Create before starting work.
- **assessment**: When evaluating system aspects, performance, or conducting reviews.
- **bug_report**: When documenting bugs, root causes, and resolutions.
- **data_contract**: When defining or changing data interfaces, schemas, or validation rules.

### How to Generate Artifacts

1. Use the `create_artifact` tool from `agent_qms.toolbelt.core` to generate artifacts based on templates and schemas.
2. Always validate artifacts using `validate_artifact` after creation.
3. Follow validation rules: Use timestamped filenames (YYYY-MM-DD_HHMM_name.md) and include timestamp (YYYY-MM-DD HH:MM KST) and branch fields in frontmatter.

### Seamless Integration Steps

- **Before Implementation**: Generate `implementation_plan` artifact outlining the plan.
- **During Assessment**: Create `assessment` artifacts for evaluations.
- **Bug Resolution**: Document with `bug_report` artifacts.
- **Data Changes**: Define with `data_contract` artifacts.
- **Post-Task**: Validate and store artifacts in designated locations.

Use terminal commands to run the Python toolbelt scripts, e.g., `python -m agent_qms.toolbelt.core create_artifact --type implementation_plan --name my_plan`.
