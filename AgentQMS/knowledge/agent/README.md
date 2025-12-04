# Agent Knowledge Index

Entry point to agent instructions.

## Primary Instructions

### 1. System (Core Rules)
**File**: `system.md`
**Read**: ALWAYS read first
**Contains**: Core rules, artifact creation, tool discovery

### 2. OCR Experiment Agent
**File**: `ocr_experiment_agent.md`
**Read**: When working with OCR experiments
**Quick Start**: `../../experiment-tracker/docs/quickstart.md`

### 3. Tracking CLI
**File**: `tracking_cli.md`
**Purpose**: Tracking database CLI commands

### 4. Tool Catalog
**File**: `tool_catalog.md`
**Purpose**: Automation tools and Make targets

### 5. Artifact Rules
**File**: `artifact_rules.yaml`
**Purpose**: Naming, validation, schemas

## Usage Patterns

### Pattern 1: Start New Task
1. Read `system.md` for core rules
2. Check `tool_catalog.md` for available tools
3. If OCR experiment → read `ocr_experiment_agent.md`
4. Run discovery: `cd AgentQMS/interface && make discover`

### Pattern 2: OCR Experiment
1. Read `ocr_experiment_agent.md`
2. Reference `quickstart.md` for rapid workflow
3. Use VLM tools (see `AgentQMS/vlm/README.md`)
4. Collect feedback after experiment

### Pattern 3: Create Artifact
1. Check `system.md` for artifact types
2. Use `make create-{type}` from `AgentQMS/interface/`
3. Validate: `make validate`
4. Run compliance check: `make compliance`

### Pattern 4: Track Work
1. Read `tracking_cli.md` for commands
2. Create plan/experiment in tracking DB
3. Update status as work progresses
4. Export results: `make exp-export`

## Agent Specializations

| Agent | Primary File | Use Case |
|-------|-------------|----------|
| **General** | `system.md` | Any AgentQMS task |
| **OCR Experiment** | `ocr_experiment_agent.md` | OCR experiments, VLM |
| **Tracking** | `tracking_cli.md` | Plan/experiment tracking |

## Quick Command Reference

### AgentQMS Tools
```bash
cd AgentQMS/interface/
make help           # Show all commands
make discover       # List available tools
make validate       # Validate artifacts
make compliance     # Full compliance check
```

### Artifact Creation
```bash
cd AgentQMS/interface/
make create-plan NAME=my-plan TITLE="My Plan"
make create-assessment NAME=my-assessment TITLE="My Assessment"
make create-bug-report NAME=my-bug TITLE="Bug Title"
```

### OCR Experiment
```bash
cd experiment-tracker/
./scripts/start-experiment.py --type TYPE --intention "INTENTION"
./scripts/record-artifact.py --path PATH --type TYPE
uv run python -m AgentQMS.vlm.cli.analyze_image_defects --image PATH --mode MODE
./scripts/generate-assessment.py --template TEMPLATE
```

### Tracking
```bash
python AgentQMS/agent_tools/utilities/tracking/cli.py plan new --title "TITLE"
python AgentQMS/agent_tools/utilities/tracking/cli.py exp new --title "TITLE"
python AgentQMS/agent_tools/utilities/tracking/cli.py plan status --concise
```

## File Structure

```
AgentQMS/knowledge/agent/
├── README.md                    # This file (index)
├── system.md                    # Core rules (SST)
├── ocr_experiment_agent.md      # OCR experiment specialization
├── tracking_cli.md              # Tracking CLI quick reference
├── tool_catalog.md              # Available tools catalog
└── artifact_rules.yaml          # Artifact validation rules
```

## Escalation

Unclear/incomplete → check `system.md` "When Stuck" section, collect feedback

---

**Version**: 1.1
**Last Updated**: 2024-12-04 12:00 (KST)
