# AI Agent Entry Point

**🚨 READ THIS FIRST** - This is the single entry point for all AI agents working on this project.

## Single Source of Truth

👉 **[`docs/agents/system.md`](docs/agents/system.md)** - **READ THIS FIRST**

All rules, instructions, and guidelines are in `system.md`. This file is just a navigation guide.

## Quick Navigation

### Core Documentation
- **System Instructions**: [`docs/agents/system.md`](docs/agents/system.md) - Single source of truth (all rules and instructions)
- **Documentation Map**: [`docs/agents/index.md`](docs/agents/index.md) - Quick reference for finding docs
- **Cursor Rules**: [`.cursor/rules/prompts-artifacts-guidelines.mdc`](.cursor/rules/prompts-artifacts-guidelines.mdc) - Always-applied rules

### By Task Type
- **Development Tasks**: [`docs/agents/protocols/`](docs/agents/protocols/) - Concise instructions
- **Reference Information**: [`docs/agents/references/`](docs/agents/references/) - Key facts
- **Detailed Context**: [`docs/maintainers/`](docs/maintainers/) - Detailed documentation for humans

### Tools
- Unified CLI (recommended):
  - List: `python -m scripts.agent_tools list`
  - Prune links: `python -m scripts.agent_tools prune-links --root docs`
  - Update artifact indexes: `python -m scripts.agent_tools update-artifact-indexes`
  - Generate sitemap: `python -m scripts.agent_tools sitemap --root docs`
- Advanced:
  - Tool Discovery: `python scripts/agent_tools/core/discover.py --list`
  - Artifact Guide: `python scripts/agent_tools/core/artifact_guide.py`
  - Index Generator: `python scripts/agent_tools/documentation/auto_generate_index.py`

---

**Remember**: All rules and instructions are in [`docs/agents/system.md`](docs/agents/system.md). This file is just a navigation guide.
