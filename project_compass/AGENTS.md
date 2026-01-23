# Project Compass V2: Vessel AI Protocols

## 1. Core Concepts

| Term                | Concept           | Explanation                                                               | Usage Example                             |
| :------------------ | :---------------- | :------------------------------------------------------------------------ | :---------------------------------------- |
| **Project Compass** | **The Product**   | The overall name of the AI Interface / Strategy system.                   | "Using Project Compass to manage work."   |
| **Vessel**          | **The Engine**    | The V2.0 architecture (Code). Powers the system with strict state rules.  | "State is stored in `.vessel/`."          |
| **Pulse**           | **The Work Unit** | A single session or cycle of work. **Replaces "Session"**.                | "Starting a pulse to refactor OCR."       |
| **Compass**         | **The Tool**      | The CLI command you type.                                                 | `uv run compass pulse-init ...`           |
| **Star-Chart**      | **The Map**       | Roadmap milestones defining long-term goals.                              | "Linking pulse to `ocr-domain-refactor`." |
| **Staging**         | **The Workspace** | `pulse_staging/artifacts/`. **The ONLY writable location** for new files. | "Drafting `design.md` in staging."        |
| **Vault**           | **The Library**   | `vault/`. Read-only source for injected rules and directives.             | "Pulse loaded rules from vault."          |

## 2. Directory Structure

```text
project_compass/
├── .vessel/                  # State storage (vessel_state.json)
├── vault/                    # Read-only rule library
│   ├── directives/           # Core protocols
│   └── milestones/           # Star-chart definitions
├── pulse_staging/
│   └── artifacts/            # ACTIVE WORKSPACE (Write here)
└── history/                  # Archived pulses
    ├── {milestone_id}/       # Grouped by goal
    │   └── {timestamp}_{id}/ # Pulse snapshot
    └── legacy/               # Old session archives
```

## 3. Interaction Protocol

Every conversation **MUST** start with:

1. Read `vessel://state` (via MCP) or `.vessel/vessel_state.json`
2. Check if active pulse exists
3. If no pulse: init one with `pulse-init`

## 3. CLI Commands

```bash
# Check environment
uv run compass check-env

# Start pulse
uv run compass pulse-init \
  --id "domain-action-target" \
  --obj "Objective (20-500 chars)" \
  --milestone "milestone-id"

# Check status
uv run compass pulse-status

# Register artifact
uv run compass pulse-sync \
  --path "filename.md" \
  --type "design|research|walkthrough|implementation_plan|bug_report|audit"

# Export pulse (blocks if unregistered files exist)
uv run compass pulse-export

# Update token burden
uv run compass pulse-checkpoint --burden high
```

## 4. Hard Constraints

- **ALL artifacts** must be in `pulse_staging/artifacts/`
- **ALL state changes** must go through CLI/MCP tools
- **NO manual YAML/JSON editing**
- **NO generic pulse IDs** (banned: "new", "session", "test", "tmp")
- **MAX 20 words** for any status note

## 5. Pulse Lifecycle

```
┌─────────────────────────────────────────────────┐
│                   PULSE INIT                    │
│  pulse-init --id X --obj Y --milestone Z        │
│  → Creates vessel_state.json                    │
│  → Injects rules from vault/                    │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│                     WORK                        │
│  - Create files in pulse_staging/artifacts/     │
│  - Register with pulse-sync                     │
│  - Check maturity with pulse-checkpoint         │
└─────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│                  PULSE EXPORT                   │
│  pulse-export                                   │
│  → Audits staging vs manifest                   │
│  → BLOCKS if unregistered files exist           │
│  → Moves artifacts to history/{timestamp}/      │
└─────────────────────────────────────────────────┘
```

## 6. MCP Resources

| URI                | Description                           |
| ------------------ | ------------------------------------- |
| `vessel://state`   | Current VesselState JSON              |
| `vessel://rules`   | Injected vault rules for active pulse |
| `vessel://staging` | List of staging artifacts             |

## 7. Vault Directives

Rules in `vault/directives/` are auto-injected on pulse-init:

- `00_meta_cognition.md` - [Compass:Reflection] protocol
- `01_naming_standards.md` - Pulse ID and artifact naming
- `02_artifact_purity.md` - Staging constraints, zero-narrative policy

Milestone-specific rules: `vault/milestones/{milestone_id}.md`

## 8. Spec Kit Integration

Project Compass integrates with GitHub Spec Kit for spec-driven development:

### MCP Tools (Meta-Tool Pattern)
- `compass_meta_pulse` - Pulse management operations
  - `kind: "init"` - Initialize new pulse
  - `kind: "sync"` - Register staging artifact
  - `kind: "export"` - Archive pulse to history
  - `kind: "status"` - Get pulse status
  - `kind: "checkpoint"` - Update token burden
- `compass_meta_spec` - Spec-driven development operations
  - `kind: "constitution"` - Establish project principles
  - `kind: "specify"` - Create baseline specifications
  - `kind: "plan"` - Generate implementation plans
  - `kind: "tasks"` - Create actionable tasks

### CLI Commands
```bash
uv run compass pulse-init \
  --id "domain-action-target" \
  --obj "Objective (20-500 chars)" \
  --milestone "milestone-id"

uv run compass pulse-status
uv run compass pulse-sync --path "file.md" --type "design"
uv run compass pulse-export
uv run compass pulse-checkpoint --burden high

uv run compass spec-constitution "principles..."
uv run compass spec-specify --scope "context" --requirements "reqs"
uv run compass spec-plan --approach "strategy"
uv run compass spec-tasks --focus "area"
```

### Artifact Types
- `specification` - Spec documents
- `requirements` - Requirements docs
- `architecture` - Architecture specs

See: `design/spec-kit-integration.md`
