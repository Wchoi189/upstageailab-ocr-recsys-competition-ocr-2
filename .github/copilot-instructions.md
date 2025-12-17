# AgentQMS Copilot Instructions

You are working in an AgentQMS-enabled project.

## Critical Rules
1. **Discovery**: Read `.ai-instructions/tier3-agents/copilot/config.yaml` and `.ai-instructions/tier2-framework/tool-catalog.yaml`.
2. **Artifacts**: NEVER create `docs/artifacts/*` files manually.
   - Use: `cd AgentQMS/interface && make create-plan` (or similar)
3. **Safety**: Run `make validate` before asking the user to review.

## Context
- AI Documentation Standard: `.ai-instructions/schema/ads-v1.0-spec.yaml`
- Tool Catalog: `.ai-instructions/tier2-framework/tool-catalog.yaml`
- Critical Rules: `.ai-instructions/tier1-sst/*.yaml`
