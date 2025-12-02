# AgentQMS Context

Last Updated: 2025-11-30 01:17 (KST)

Python-based QMS (Quality Management System) agent.
Directory: `/workspaces/upstageailab-ocr-recsys-competition-ocr-2`
Source: `AgentQMS/`
Config: `pyproject.toml`, `requirements.txt`, `pytest.ini`
Python project with tests and pre-commit hooks.

## Agent Tools Reference

### Agent-Only Interface
- Directory: `AgentQMS/interface/` (AI agents only - humans should NOT use)
- Implementation: `AgentQMS/agent_tools/` (canonical Python packages)

### Key Agent Commands
- `cd AgentQMS/interface/` - Navigate to agent interface
- `make help` - Show available agent commands
- `make discover` - List all available tools
- `make status` - Check system status
- `make validate` - Validate all artifacts
- `make compliance` - Check compliance status

### Artifact Creation
- `make create-plan NAME=title TITLE="Plan Title"`
- `make create-assessment NAME=title TITLE="Assessment Title"`
- `make create-audit NAME=title TITLE="Audit Title"`
- `make create-design NAME=title TITLE="Design Title"`
- `make create-research NAME=title TITLE="Research Title"`

### Context Bundles
- `make context TASK="task description"`
- `make context-development` - Get development context
- `make context-docs` - Get documentation context
- `make context-debug` - Get debugging context
- `make context-plan` - Get planning context

### Quality & Validation
- `make ast-analyze` - Analyze codebase structure
- `make ast-check-quality` - Check code quality
- `make ast-generate-tests TARGET=file.py` - Generate test scaffolds
- `make ast-extract-docs TARGET=file.py` - Extract documentation

### Feedback & Issues
- `make feedback-issue ISSUE="description" FILE="path"`
- `make feedback-suggest AREA="area" CURRENT="current" CHANGE="suggested" RATIONALE="reason"`
- `make quality-check` - Check documentation quality
