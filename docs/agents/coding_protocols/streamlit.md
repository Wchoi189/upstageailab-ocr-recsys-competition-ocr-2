---
title: "Streamlit Coding Protocol"
date: "2025-11-01"
type: "guide"
category: "ai_agent"
status: "active"
version: "1.0"
tags: ["ai_agent", "coding", "protocol", "reference"]
---

# Streamlit Coding Protocol

**Quick reference for AI agents working on this codebase**

## 🎯 Core Principles

1. **Schema-Driven Architecture**: Pages defined in YAML (`streamlit_app/page_schemas/pages/`)
2. **Component Renderer Pattern**: All UI through modular renderers (not direct Streamlit calls)
3. **YAML Configuration**: All config in YAML files, never hardcoded defaults
4. **Absolute Imports**: Always use `from streamlit_app.module import Class`
5. **Browser Testing**: Streamlit errors only visible in browser, use Puppeteer scripts

## 📁 File Organization

### Safe to Modify
- `streamlit_app/page_schemas/pages/*.yaml` - Page definitions
- `streamlit_app/pages/*.py` - Page entry points
- `streamlit_app/schema_engine/renderers/*/` - New renderer files

### Protected (Modify with caution)
- `streamlit_app/schema_engine/core/` - Core engine files
- `streamlit_app/schema_engine/renderers/base.py` - Base renderer class
- `streamlit_app/schema_engine/component_renderer_factory.py` - Renderer registry

## 🛠️ Development Workflow

### Starting the App
```bash
make run          # Start Streamlit app
make stop         # Stop app
make status       # Check status
```

### Artifact Creation (CRITICAL)
**NEVER create artifacts manually** - Always use automated workflow tools:

```bash
# METHOD 1: Direct Python (always works)
python scripts/agent_tools/core/artifact_workflow.py create --type [TYPE] --name [NAME] --title "[TITLE]"

# METHOD 2: Agent-Only Directory (RECOMMENDED)
cd agent/
make create-plan NAME=my-plan TITLE="My Plan"
make create-assessment NAME=my-assessment TITLE="My Assessment"
make create-design NAME=my-design TITLE="My Design"
```

**Available Types**: `implementation_plan`, `assessment`, `design`, `research`, `template`, `bug_report`

**Requirements**:
- Naming: `YYYY-MM-DD_HHMM_[TYPE]_descriptive-name.md`
- Must include frontmatter (title, timestamp, branch, status, and type-specific fields as required)
- Files go to `artifacts/[type]s/` directory (project root, NOT docs/artifacts/)
- Never create files in project root

**Tool Discovery**:
```bash
cd agent/
make help          # Show all agent commands
make discover      # List all tools
make status        # Check system status
```

### Testing
```bash
# Browser testing (REQUIRED for Streamlit)
node scripts/browser-automation/verify_fixes.js

# Logs
python3 scripts/process_manager.py logs --port 8501 --lines 100
```

### Code Quality
- Keep functions <50 lines
- One responsibility per function
- Use absolute imports only
- Follow existing patterns

## 🔧 Common Patterns

### Adding a New Component Renderer
1. Create file in appropriate subdirectory: `renderers/selection/`, `renderers/input/`, etc.
2. Inherit from `BaseComponentRenderer`
3. Register in `component_renderer_factory.py._initialize_default_renderers()`
4. Export in `renderers/__init__.py`

### Data Binding
- Use `value_source`, `options_source` in YAML schemas
- Sync session state in page wrapper files
- Access via `DataBindingService` from `schema_engine.core.data_binding`

### Configuration
- Load from YAML: `from streamlit_app.utils.pydantic_config import load_app_config`
- Never use `AppConfig()` directly (uses hardcoded defaults)
- Update YAML files, not Python models

## 📚 Key Documentation

- **Architecture**: `.ai-context.md` (read before changes)
- **Agent System**: `docs/ai_agent/system.md` (single source of truth)
- **Coding Rules**: `docs/guidelines/2025-11-01_1330_guidelines_ai-coding-rules.md`
- **Architecture Overview**: `docs/architecture/ARCHITECTURE_OVERVIEW.md`

## ⚠️ Critical Rules

1. **Use artifact tools** - NEVER create artifacts manually; always use `artifact_workflow.py` or `make` commands
2. **No monolithic scripts** - Split into small functions
3. **No relative imports** - Always absolute: `from streamlit_app.module import Class`
4. **Browser testing only** - Streamlit errors invisible to unit tests
5. **YAML configuration** - Never hardcode defaults in Pydantic models
6. **Follow patterns** - Check existing code before creating new
7. **No files in project root** - Artifacts go to `artifacts/[type]s/` (project root) with proper naming and frontmatter. Use `artifact_workflow.py` script for all artifacts including summaries.

## 🚨 When to Ask

- Multiple valid approaches exist
- Breaking changes needed
- Unclear requirements
- Before creating new architectural patterns
