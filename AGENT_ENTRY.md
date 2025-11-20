# AI Agent Entry Point

**🚨 READ THIS FIRST** - This is the single entry point for all AI agents working on this project.

## Single Source of Truth

👉 **[`docs/agents/system.md`](docs/agents/system.md)** - **READ THIS FIRST**

All rules, instructions, and guidelines are in `system.md`. This file is just a navigation guide.

## 🚨 Critical Documentation - MUST READ Before Code Changes

**REQUIRED reading before modifying code:**
- **Data Contracts**: [`docs/pipeline/data_contracts.md`](docs/pipeline/data_contracts.md) - Prevents shape errors
- **API Contracts**: [`docs/api/pipeline-contract.md`](docs/api/pipeline-contract.md) - Prevents API violations
- **Coding Standards**: [`docs/agents/protocols/development.md#coding-standards`](docs/agents/protocols/development.md) - Type hints, formatting, conventions
- **State Tracking**: [`.agentqms/USAGE_GUIDE.md`](.agentqms/USAGE_GUIDE.md) - Agent state persistence

## Quick Navigation

### Core Documentation
- **System Instructions**: [`docs/agents/system.md`](docs/agents/system.md) - Single source of truth
- **Documentation Map**: [`docs/agents/index.md`](docs/agents/index.md) - Quick reference
- **Cursor Rules**: [`.cursor/rules/prompts-artifacts-guidelines.mdc`](.cursor/rules/prompts-artifacts-guidelines.mdc) - Always-applied rules

### By Task Type
- **Development Tasks**: [`docs/agents/protocols/`](docs/agents/protocols/) - Concise instructions
- **Reference Information**: [`docs/agents/references/`](docs/agents/references/) - Key facts
- **Detailed Context**: [`docs/maintainers/`](docs/maintainers/) - For humans

## ✅ Pre-Commit Checklist

Before committing, verify:
- [ ] Artifact frontmatter validated: `python scripts/agent_tools/documentation/validate_manifest.py`
- [ ] Code formatted: `uv run ruff format .`
- [ ] Code checked: `uv run ruff check . --fix`
- [ ] Data contracts reviewed (if pipeline changes)
- [ ] API contracts reviewed (if API changes)

## 📊 Status Updates

**When to provide status updates:**
- Every 5 major tasks completed
- When encountering blockers
- On major milestones

**Format**: See [`docs/agents/protocols/status-update.md`](docs/agents/protocols/status-update.md)

## Tools

- **Unified CLI** (recommended):
  - List: `python -m scripts.agent_tools list`
  - Prune links: `python -m scripts.agent_tools prune-links --root docs`
  - Update indexes: `python -m scripts.agent_tools update-artifact-indexes`
- **Advanced**:
  - Tool Discovery: `python scripts/agent_tools/core/discover.py --list`
  - Artifact Guide: `python scripts/agent_tools/core/artifact_guide.py`

---

**Remember**: All rules and instructions are in [`docs/agents/system.md`](docs/agents/system.md). This file is just a navigation guide.
