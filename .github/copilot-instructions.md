# AgentQMS Copilot Instructions

You are working in an AgentQMS-enabled project.

**Requirements:**
- Generate artifacts using AgentQMS tools
- Follow project conventions
- Reference existing documentation
- AI-only documentation: ultra-concise, machine-parseable, no tutorials
- Flag project pain points (standardization gaps, inconsistencies, unclear processes)

**Execution Mode:**
- Direct answers only
- Assume domain expertise
- Execute instructions without conversation
- Provide solutions, not explanations

## Critical Rules
1. **Discovery**: Read `.ai-instructions/tier3-agents/copilot/config.yaml` and `.ai-instructions/tier2-framework/tool-catalog.yaml`.
2. **Artifacts**: NEVER create `docs/artifacts/*` files manually.
   - Use: `cd AgentQMS/interface && make create-plan` (or similar)
3. **Safety**: Run `make validate` before asking the user to review.

## Context
- AI Documentation Standard: `.ai-instructions/schema/ads-v1.0-spec.yaml`
- Tool Catalog: `.ai-instructions/tier2-framework/tool-catalog.yaml`
- Critical Rules: `.ai-instructions/tier1-sst/*.yaml`

## Safe State Management
- **State Files**: Experiment state files are now in YAML format (`state.yml`) instead of JSON for better safety and readability.
- **Safe Editing**: Use `experiment-tracker/scripts/safe_state_manager.py` for all state file operations to prevent corruption.
- **Commands**:
  - Validate: `python scripts/safe_state_manager.py <path>/state.yml --validate`
  - Get section: `python scripts/safe_state_manager.py <path>/state.yml --get <section>`
  - Set section: `python scripts/safe_state_manager.py <path>/state.yml --set <key> <value>`
- **ETK Tool**: Use `etk sync --all` to keep state files and metadata in sync.
- **Migration**: Existing JSON files have been migrated to YAML with backups.
