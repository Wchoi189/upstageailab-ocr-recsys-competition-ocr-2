# AgentQMS Context

Last Updated: 2025-12-06 22:00 (KST)

**AgentQMS** is a quality management framework for AI agents working with documentation artifacts.

## Structure

```
AgentQMS/
├── interface/          # AI agent interface (Makefile + workflows)
├── agent_tools/        # Core implementation (Python packages)
│   ├── audit/         # Artifact audit and frontmatter tools
│   ├── compliance/    # Validation and compliance checking
│   ├── core/          # Core utilities (config, context, plugins)
│   ├── generators/    # Artifact generators and templates
│   └── utilities/     # Helper tools (migration, repair, etc.)
├── knowledge/agent/    # Agent documentation and rules
├── conventions/        # Coding conventions and standards
└── vlm/               # Vision Language Model integration
```

## Quick Start (AI Agents)

### Essential Commands
```bash
cd AgentQMS/interface/    # Navigate to agent interface
make help                 # Show all available commands
make discover             # List all tools and workflows
make status               # Check framework status
make validate             # Validate all artifacts
make compliance           # Check compliance with standards
```

### Artifact Management

**Create New Artifacts:**
```bash
make create-plan NAME=my-plan TITLE="Implementation Plan"
make create-assessment NAME=my-assessment TITLE="Assessment Title"
make create-design NAME=my-design TITLE="Design Document"
make create-research NAME=my-research TITLE="Research Document"
make create-bug-report NAME=my-bug TITLE="Bug Report"
```

**Audit & Fix Artifacts:**
```bash
make audit-fix-batch BATCH=1         # Preview batch 1 changes
make audit-fix-batch-apply BATCH=1   # Apply batch 1 fixes
make audit-fix-all                   # Fix all (safe, excludes archive/)
make audit-report                    # Report violations without fixing
```

**VLM Reports:**
```bash
make audit-vlm-migrate               # Preview VLM report migration
make audit-vlm-migrate-apply         # Apply VLM migration
```

### Context & Discovery

**Load Context for Tasks:**
```bash
make context TASK="implement feature X"    # Auto-detect context bundle
make context-development                   # Development context
make context-docs                          # Documentation context
make context-debug                         # Debugging context
make context-plan                          # Planning context
make context-list                          # List all bundles
```

**Tool Discovery:**
```bash
make discover                        # List all tools
make tool-catalog                    # Generate tool catalog
make info-tools                      # Show detailed tool info
```

### Validation & Quality

**Validation:**
```bash
make validate                        # Validate all artifacts
make validate-file FILE=path.md      # Validate single file
make validate-naming                 # Check naming only
make boundary                        # Check framework boundaries
```

**Compliance & Quality:**
```bash
make compliance                      # Check compliance
make quality-check                   # Check documentation quality
make all-check                       # Run all checks
```

**Code Analysis:**
```bash
make ast-analyze TARGET=file.py      # Analyze code structure
make ast-check-quality TARGET=dir/   # Check code quality
make ast-generate-tests TARGET=file  # Generate test scaffolds
make ast-extract-docs TARGET=module  # Extract documentation
```

### Advanced Features

**Artifact Migration & Repair:**
```bash
make artifacts-migrate FILE=path --autofix     # Migrate artifact
make artifacts-repair-moves                    # Repair moved artifacts
make artifacts-status                          # Show artifact health
make artifacts-status-aging                    # Show aging artifacts
```

**Tracking & Changelog:**
```bash
make track-status                    # Tracking DB status
make track-repair                    # Repair tracking DB
make changelog-preview               # Preview changelog
make changelog-draft                 # Generate changelog
```

**Framework Management:**
```bash
make audit-framework                 # Audit framework compliance
make deprecated-list                 # List deprecated symbols
make deprecated-validate             # Validate for deprecated usage
make version                         # Show version info
```

## Configuration

**Main Config:** `.agentqms/settings.yaml`
- Framework settings
- Validation rules
- Excluded directories (archive/, deprecated/)
- Path configurations

**Artifact Rules:** `AgentQMS/knowledge/agent/artifact_rules.yaml`
- Artifact type definitions
- Naming conventions
- Frontmatter requirements

**Agent Documentation:** `AgentQMS/knowledge/agent/`
- `system.md` - Core agent rules and workflows
- `artifact_rules.yaml` - Artifact standards
- `artifact-audit-improvements.md` - Audit tool guide
- `tool_catalog.md` - Tool reference

## Safety Features

**Directory Exclusion:**
- `archive/` and `deprecated/` excluded by default
- Configurable via `settings.yaml`
- Override with `--include-excluded` flag

**Audit Safety:**
- Preview changes with `--dry-run`
- Confirmation prompts (bypass with `--no-confirm`)
- Automatic git stash backup (bypass with `--no-stash`)
- Smart date inference (git → filesystem → present)

## Common Workflows

**Complete Artifact Workflow:**
```bash
make workflow-create NAME=my-artifact TYPE=plan TITLE="My Plan"
```

**Documentation Workflow:**
```bash
make workflow-docs        # Generate, validate, update
```

**Validation Workflow:**
```bash
make workflow-validate    # Validate, fix, monitor
```

## For Humans

⚠️ **Warning:** The `AgentQMS/interface/` directory is for AI agents only.

Humans should use:
- Main project Makefile: `/workspaces/.../Makefile`
- Documentation: `docs/` directory
- Agent docs: `AgentQMS/knowledge/agent/` for reference
