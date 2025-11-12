# Tools Reference

**Purpose:** Quick reference for agent tools. For detailed context, see `docs/maintainers/`.

## Tool Discovery

```bash
# List all tools
python scripts/agent_tools/core/discover.py --list

# Discover tools
python scripts/agent_tools/core/discover.py
```

## Tool Categories

**Core:**
- `discover.py` - Tool discovery
- `artifact_workflow.py` - Artifact creation (legacy)

**Documentation:**
- `validate_manifest.py` - Validate documentation manifest
- `auto_generate_index.py` - Generate documentation index

**OCR:**
- `next_run_proposer.py` - Propose next training run
- OCR-specific tools

**Utilities:**
- `get_context.py` - Get context bundles
- `export_framework.py` - Export framework

**Maintenance:**
- `reorganize_files.py` - Reorganize files
- `fix_naming_conventions.py` - Fix naming

**Compliance:**
- `validate_artifacts.py` - Validate artifacts
- `monitor_artifacts.py` - Monitor artifacts
- `fix_artifacts.py` - Fix artifacts

## Context Bundles

```bash
# List bundles
uv run python scripts/agent_tools/utilities/get_context.py --list-bundles

# Get bundle
uv run python scripts/agent_tools/utilities/get_context.py --bundle streamlit-maintenance
```

## AgentQMS Toolbelt

**Import:**
```python
from agent_qms.toolbelt import AgentQMSToolbelt
```

**Usage:**
```python
toolbelt = AgentQMSToolbelt()
artifact_path = toolbelt.create_artifact(
    artifact_type="assessment",  # or "implementation_plan"
    title="My Artifact",
    content="## Summary\n...",
    author="ai-agent",
    tags=["tag1", "tag2"]
)
```

**Artifact Types:**
- `assessment` → `artifacts/assessments/`
- `implementation_plan` → `artifacts/implementation_plans/`

