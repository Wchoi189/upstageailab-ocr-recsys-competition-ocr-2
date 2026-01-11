---
ads_version: "1.0"
type: agent_instructions
agent: copilot
tier: 1
priority: critical
memory_footprint: 40
date: "2026-01-12"
status: active
---

> [!IMPORTANT]
> **Always read `utilities://quick-reference` first** — provides 2500x faster config loading and standard path utilities.

# AgentQMS Copilot Instructions

## Environment
- **System**: AgentQMS (Quality Management System)
- **Standards**: AgentQMS/standards/
- **Tools**: AgentQMS/bin/make (preferred) or uv run
- **Artifacts**: docs/artifacts/ (auto-validated)

## Agent Constraints
1. Artifacts → Always use `AgentQMS/bin/make create-*` (not manual)
2. Validation → Run before completion: `make validate`
3. Standards → Read AgentQMS/standards/INDEX.yaml (single source)

## Tool Routing
| Task | Command | Location |
|------|---------|----------|
| Create artifact | make create-plan/assessment/design | AgentQMS/bin |
| Validate | make validate | AgentQMS/bin |
| Check compliance | make compliance | AgentQMS/bin |
| Discover tools | make discover | AgentQMS/bin |
| Get context | suggest_context.py "task" | AgentQMS/tools/utilities |

## Utilities (Preferred Imports)
| Use | Import | Note |
|-----|--------|------|
| YAML | ConfigLoader from AgentQMS.tools.utils.config_loader | 2000x cache speedup |
| Paths | get_project_root, get_data_dir from AgentQMS.tools.utils.paths | No hardcoding |
| Time | get_kst_timestamp, format_kst from AgentQMS.tools.utils.timestamps | KST only |
| Git | get_current_branch, get_commit_hash from AgentQMS.tools.utils.git | No subprocess |

## Workflow
1. **Discover** → Run suggest_context.py or make context TASK="..."
2. **Plan** → Use implementation plan artifact type
3. **Execute** → Follow standards, use utilities
4. **Validate** → make validate before completion
5. **Report** → If needed, use assessment/audit artifact types

## Key Policies
- ❌ Manual artifact creation in docs/artifacts/
- ❌ Hardcoded paths or subprocess calls
- ✅ Use ConfigLoader for YAML
- ✅ Use paths utilities for directories
- ✅ Auto-inject utility context when available
- ✅ Follow lowercase-kebab-case naming

## Resources (Priority Order)
1. **FIRST**: `utilities://quick-reference` (Tier 1 utilities — config, paths, timestamps, git)
2. Standards: AgentQMS/standards/INDEX.yaml
3. Tool Catalog: AgentQMS/standards/tier2-framework/tool-catalog.yaml
4. Utilities Index: context/utility-scripts/utility-scripts-index.yaml
5. Project Context: AGENTS.md
