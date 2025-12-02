# AgentQMS Framework Overview

## What is AgentQMS?

AgentQMS (Quality Management Framework) is a reusable framework that standardizes artifact creation, documentation workflows, and automation for collaborative AI coding. The framework is **containerized** so it can travel between projects as a pair of directories: `.agentqms/` + `AgentQMS/`.

## Key Features

### 1. Artifact Management
- Standardized artifact creation (implementation plans, assessments, designs, bug reports)
- Automatic validation and compliance checking
- Consistent naming and frontmatter requirements

### 2. Context Loading
- Task-specific context bundles
- Automatic task type detection
- Smart file selection based on task keywords

### 3. Tool Discovery
- Comprehensive tool registry
- Workflow suggestions based on task analysis
- Auto-discovery of available tools and commands

### 4. Auto-Execution
- Automatic validation after artifact creation
- Index updates after changes
- Workflow suggestions for next steps

## Framework Structure

```
project_root/
├── AgentQMS/                  # Framework container
│   ├── interface/             # Agent commands (Makefile)
│   ├── agent_tools/            # Implementation layer
│   ├── conventions/            # Schemas, templates, audit framework
│   └── knowledge/             # Documentation surface
│       ├── agent/              # AI agent instructions (SST)
│       ├── protocols/          # Governance, development protocols
│       └── references/         # Technical references
├── .agentqms/                  # Framework state
│   ├── settings.yaml           # Project configuration
│   ├── state/architecture.yaml # Component map
│   └── plugins/                # Project extensions
├── artifacts/                  # QMS artifacts
└── .copilot/context/           # Auto-discovery context files
```

## Entry Points

1. **System Instructions**: `AgentQMS/knowledge/agent/system.md` (REQUIRED)
2. **Architecture Map**: `.agentqms/state/architecture.yaml`
3. **Tool Catalog**: `.copilot/context/tool-catalog.md`
4. **Workflow Triggers**: `.copilot/context/workflow-triggers.yaml`

## Quick Start

### For AI Agents

1. Read `AgentQMS/knowledge/agent/system.md` first
2. Check `.agentqms/state/architecture.yaml` for component map
3. Use `cd AgentQMS/interface && make help` to see available commands
4. Create artifacts using `make create-plan NAME=... TITLE=...`
5. Validate with `make validate && make compliance`

### Common Workflows

**Creating Artifacts**:
```bash
cd AgentQMS/interface
make create-plan NAME=feature-name TITLE="Feature Title"
make create-assessment NAME=assessment-name TITLE="Assessment Title"
make create-bug-report NAME=bug-name TITLE="Bug Description"
```

**Validation**:
```bash
cd AgentQMS/interface
make validate          # Validate all artifacts
make compliance        # Full compliance check
make boundary          # Boundary validation
```

**Context Loading**:
```bash
cd AgentQMS/interface
make context-development    # Development tasks
make context-docs          # Documentation tasks
make context-debug         # Debugging tasks
make context-plan          # Planning tasks
```

## Auto-Discovery

The framework supports automatic discovery:

- **Tool Registry**: Automatically generated from `AgentQMS/agent_tools/`
- **Workflow Suggestions**: Automatically triggered based on task keywords
- **Context Bundles**: Automatically suggested based on task type
- **Auto-Validation**: Runs automatically after artifact creation

## Capabilities

The framework provides these capabilities:

- **plan_generation**: Create implementation plans using Blueprint Protocol Template
- **quality_assessment**: Validate artifacts and boundaries
- **documentation_validation**: Validate documentation links and manifests
- **context_loading**: Generate task-specific context bundles
- **audit_execution**: Run framework audits using the audit framework

See `.agentqms/state/architecture.yaml` for detailed capability mappings.

## Plugin System

AgentQMS supports project-level extensions via plugins:

- **Artifact Types**: New document types with custom templates
- **Validators**: Additional validation rules and prefixes
- **Context Bundles**: Task-specific context file collections

Plugins are defined in `.agentqms/plugins/` and automatically discovered.

## Reference

- Framework README: `README.md`
- System SST: `AgentQMS/knowledge/agent/system.md`
- Architecture: `.agentqms/state/architecture.yaml`
- Tool Catalog: `AgentQMS/knowledge/agent/tool_catalog.md`
- Copilot Instructions: `.github/copilot-instructions.md`

