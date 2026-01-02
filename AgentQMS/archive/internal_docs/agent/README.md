# Agent Knowledge Index

Entry point to agent instructions and tool discovery.

## Primary Instructions

### 1. System (Core Rules) ⭐
**File**: `system.md`
**Read**: ALWAYS read first
**Contains**: Core rules, artifact creation, tool discovery, when stuck

### 2. Tracking Database Integration ⭐ NEW
**File**: `../../references/tracking/` (see below)
**Contains**: Tracking DB schema, artifact registration, status sync
**Quick Start**: Create plan → auto-registers in tracking DB → status syncs bidirectionally

### 3. OCR Experiment Agent
**File**: `ocr_experiment_agent.md`
**Read**: When working with OCR experiments
**Quick Start**: `../../experiment-tracker/docs/quickstart.md`

### 4. Tracking CLI
**File**: `tracking_cli.md`
**Purpose**: Tracking database CLI commands for manual operations

### 5. Tool Catalog
**File**: `tool_catalog.md`
**Purpose**: Automation tools and Make targets (25+ now available)

### 6. Artifact Rules
**File**: `artifact_rules.yaml`
**Purpose**: Naming conventions, validation rules, artifact schemas

## New Features (This Session - Dec 6, 2025)

### Smart Artifact Features
- **Context Suggestion** (`make context-suggest`) - AI agents get relevant context
- **Plan Progress Tracking** (`make plan-progress-*`) - Sync artifact status
- **Smart Auto-Population** (`make smart-suggest-*`) - Git-based metadata suggestions

### Advanced Migration & Repair
- **Legacy Migration** (`make artifacts-migrate`) - Batch naming convention updates
- **Manual Move Detection** (`make artifacts-detect-moves`) - Content-based duplicate detection
- **Tracking DB Repair** (`make track-repair`) - Sync stale artifact paths

### Validation & Quality
- **Deprecated Code Registry** (`make deprecated-*`) - Block modifications to deprecated symbols
- **Full Compliance Checking** (`make compliance`) - Comprehensive validation

### Framework Export
- **Adoption Scaffolding** (`make agentqms-init`) - Deploy AgentQMS to new projects
- **Quickstart Guide** (`quickstart.md`) - 5-minute setup for new teams

## Usage Patterns

### Pattern 1: Start New Task
1. Read `system.md` for core rules
2. Check `tool_catalog.md` for available tools
3. If OCR experiment → read `ocr_experiment_agent.md`
4. Run discovery: `cd AgentQMS/interface && make discover`

### Pattern 2: Create Artifact (Now with Auto-Tracking!)
1. Check `system.md` for artifact types
2. Use `make create-{type}` from `AgentQMS/interface/`
3. ✨ **NEW**: Automatically registers in tracking DB (implementation_plan, assessment, etc.)
4. Validate: `make validate`
5. Run compliance check: `make compliance`

### Pattern 3: Track Plan Progress
1. Create implementation plan → auto-registers in tracking DB
2. Add tasks to checklist: `- [ ] Task 1.1`
3. Update progress: `make plan-progress-complete FILE=path TASK="Task 1.1"`
4. Status syncs to tracking DB automatically

### Pattern 4: Detect Manual Moves
1. Check for manually moved artifacts: `make artifacts-detect-moves`
2. Preview repairs: `make artifacts-repair-moves-preview`
3. Apply repairs: `make artifacts-repair-moves`
4. Tracking DB paths auto-update

### Pattern 5: Export Framework to New Project
1. `cd /target/project`
2. `make agentqms-init TARGET=.`
3. Read generated `quickstart.md` for setup
4. Ready to use in 5 minutes!

## Agent Specializations

| Agent | Primary File | Use Case |
|-------|-------------|----------|
| **General** | `system.md` | Any AgentQMS task |
| **Quality Manager** | `system.md` + tool references | Validation, compliance |
| **Framework Developer** | `system.md` + tracking docs | Artifact lifecycle, DB sync |
| **OCR Experiment** | `ocr_experiment_agent.md` | OCR experiments, VLM |
| **Migration Specialist** | Legacy migrator tools | Artifact migration, cleanup |

## Quick Command Reference

### Artifact Creation & Auto-Tracking
```bash
cd AgentQMS/interface/

# Create with auto-tracking (implementation_plan, assessment, bug_report, design, research)
make create-plan NAME=my-feature TITLE="Feature Title"

# Check tracking status
make track-status

# Repair stale paths in tracking DB
make track-repair --dry-run
```

### Smart Features
```bash
# Get context suggestions for task
make context-suggest TASK="implement authentication"

# Get metadata suggestions
make smart-suggest-metadata TYPE=implementation_plan

# Track plan progress
make plan-progress-show FILE=docs/artifacts/plans/my-plan.md
make plan-progress-complete FILE=path TASK="Task 1.1"
```

### Migration & Repair
```bash
# Find legacy artifacts
make artifacts-find LIMIT=10

# Migrate with preview
make artifacts-migrate-dry FILE=docs/artifacts/old.md

# Detect manual moves and repair
make artifacts-detect-moves
make artifacts-repair-moves-preview
make artifacts-repair-moves
```

### Validation & Deprecated Code
```bash
# Validate and check compliance
make validate
make compliance

# Check for deprecated symbols
make deprecated-list
make deprecated-validate --all
```

### Framework Export
```bash
# Initialize AgentQMS in new project
make agentqms-init TARGET=/path/to/new/project

# Or in current directory
make agentqms-init-here
```

### Tracking
```bash
# Manual tracking operations (rarely needed - auto done via artifact creation)
python AgentQMS/agent_tools/utilities/tracking/cli.py plan new --title "TITLE"
python AgentQMS/agent_tools/utilities/tracking/cli.py exp new --title "TITLE"
make track-status
```

## File Structure

```
AgentQMS/knowledge/agent/
├── README.md                           # This file (index) ⭐ UPDATED
├── system.md                           # Core rules (SST)
├── quickstart.md                       # NEW: 5-minute framework setup
├── ocr_experiment_agent.md             # OCR experiment specialization
├── tracking_cli.md                     # Tracking CLI reference
├── tool_catalog.md                     # Available tools (25+)
└── artifact_rules.yaml                 # Artifact validation rules

AgentQMS/agent_tools/utilities/
├── suggest_context.py                  # Context suggestion engine
├── plan_progress.py                    # Plan progress tracking
├── legacy_migrator.py                  # Artifact migration (extended)
├── deprecated_registry.py              # Deprecated symbol registry
├── smart_populate.py                   # Smart metadata suggestions
├── tracking_repair.py                  # Tracking DB repair
└── tracking_integration.py             # NEW: Artifact→DB integration

AgentQMS/interface/workflows/
└── init_framework.sh                   # NEW: Framework export script
```

## Key Improvements

| Feature | Benefit | Command |
|---------|---------|---------|
| **Auto-Tracking** | Artifacts registered in DB on creation | `make create-plan` |
| **Context Suggestion** | Relevant bundles for task | `make context-suggest` |
| **Plan Progress Sync** | Checklist ↔ Tracking DB | `make plan-progress-*` |
| **Manual Move Detection** | Find & repair duplicates | `make artifacts-detect-moves` |
| **Framework Export** | Deploy to new projects | `make agentqms-init` |
| **Deprecated Registry** | Prevent modifications | `make deprecated-*` |

## Escalation

- **Unclear/incomplete** → check `system.md` "When Stuck" section
- **Need help setting up** → read `quickstart.md`
- **Tracking issues** → see tracking DB integration docs
- **Artifact problems** → run `make validate` and `make compliance`
- **Export issues** → check `init_framework.sh` output messages

---

**Version**: 2.0 - Complete Framework Enhancement
**Last Updated**: 2025-12-06 02:30 (KST)
**Status**: ✅ Production Ready
