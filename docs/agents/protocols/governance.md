# Governance Protocols

**Purpose:** Concise instructions for governance tasks. For detailed context, see `docs/maintainers/protocols/governance/`.

## Artifact Management

**AgentQMS Toolbelt (Preferred):**
```python
from agent_qms.toolbelt import AgentQMSToolbelt

toolbelt = AgentQMSToolbelt()
artifact_path = toolbelt.create_artifact(
    artifact_type="assessment",  # or "implementation_plan"
    title="My Artifact",
    content="## Summary\n...",
    author="ai-agent",
    tags=["tag1", "tag2"]
)
```

**Artifact Locations:**
- Implementation plans: `artifacts/implementation_plans/`
- Assessments: `artifacts/assessments/`
- Bug reports: `docs/bug_reports/` (AgentQMS-managed)

**Legacy Method:**
```bash
python scripts/agent_tools/core/artifact_workflow.py create --type <TYPE> --name <NAME> --title "<TITLE>"
```

## Implementation Plan Protocol

**Blueprint Template:**
- Always use Blueprint Protocol Template (PROTO-GOV-003)
- AgentQMS toolbelt uses template automatically
- See `docs/maintainers/protocols/governance/03_blueprint_protocol_template.md`

**Required Sections:**
- Objective, Approach, Implementation Steps
- Testing Strategy, Success Criteria

## Documentation Governance

**Update Protocol:**
- Update indexes when adding new docs
- Follow documentation style guide
- Link from relevant sections

**Documentation Style:**
- Use concise instructions (not tutorials)
- Bullet points, not paragraphs
- Minimal examples (correct/incorrect patterns)

## Bug Fix Protocol

**⚠️ CRITICAL: Bug reports require bug ID generation before creation.**

**Process:**
1. Generate bug ID: `uv run python scripts/bug_tools/next_bug_id.py` (generates `BUG-YYYYMMDD-###`)
2. Create bug report using AgentQMS toolbelt (preferred) or manual process
3. **Embed bug ID in code changes** (see Code Indexing below)
4. Create code changes document: `docs/bug_reports/BUG-YYYYMMDD-###-code-changes.md`
5. Update changelog
6. Update related documentation

**⚠️ CRITICAL: Code Indexing with Bug ID**
When making changes to core project files, **ALWAYS embed the bug ID in the code**:
- Add bug ID to function docstrings
- Add bug ID comments at change locations
- Index at function level, not just file level (functions survive refactoring)
- See `docs/agents/protocols/development.md` for detailed protocol and examples

**Bug Report Creation (Two Workflows):**

**Primary Workflow: Create Empty Template → Fill In Manually**
```bash
# Step 1: Generate bug ID
uv run python scripts/bug_tools/next_bug_id.py  # Returns BUG-YYYYMMDD-###

# Step 2: Create empty template with placeholders
uv run python scripts/agent_tools/core/artifact_workflow.py create \
  --type bug_report \
  --name "bug-description" \
  --title "Bug Description" \
  --bug-id "BUG-YYYYMMDD-###" \
  --severity "High" \
  --tags "bug,issue"

# Step 3: Fill in template manually
# Edit docs/bug_reports/BUG-YYYYMMDD-###_bug-description.md
```

**Alternative Workflow: Create with Full Content (Programmatic)**
```bash
# Step 1: Generate bug ID
uv run python scripts/bug_tools/next_bug_id.py  # Returns BUG-YYYYMMDD-###

# Step 2: Create with full content from file
uv run python scripts/agent_tools/core/artifact_workflow.py create \
  --type bug_report \
  --name "bug-description" \
  --title "Bug Description" \
  --content-file path/to/bug_report_content.md \
  --bug-id "BUG-YYYYMMDD-###" \
  --severity "High" \
  --tags "bug,issue"
```

**Or Using AgentQMS Toolbelt Directly:**
```python
from agent_qms.toolbelt import AgentQMSToolbelt
import subprocess

# Generate bug ID first
bug_id = subprocess.check_output(
    ["uv", "run", "python", "scripts/bug_tools/next_bug_id.py"],
    text=True
).strip()

toolbelt = AgentQMSToolbelt()
artifact_path = toolbelt.create_artifact(
    artifact_type="bug_report",
    title="Bug Description",
    content="## Summary\n...",  # Full markdown content
    author="ai-agent",
    tags=["bug", "issue"],
    bug_id=bug_id,
    severity="High"
)
```

**Note:** Templates render with default placeholders. User content (if provided) is appended after the template.

**Bug Report Template:**
- Location: `docs/bug_reports/BUG_REPORT_TEMPLATE.md` (reference template)
- AgentQMS template: `agent_qms/templates/bug_report.md` (used automatically)
- Required: Summary, Environment, Steps, Root Cause, Resolution

## Streamlit Maintenance

**Protocol:**
- Test in browser after changes
- Check logs for errors
- Update dependencies: `uv sync`
- Validate all pages work

**Common Issues:**
- Duplicate element keys → Use unique keys
- Import deadlocks → Check import order
- Threading issues → Use proper async patterns

## Compliance Validation

**Checks:**
```bash
# Code quality
uv run ruff check <files>
uv run ruff format <files>

# Type checking
uv run mypy <files>

# Template validation
python scripts/agent_tools/documentation/validate_manifest.py
```

