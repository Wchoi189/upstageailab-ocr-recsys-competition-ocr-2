#!/usr/bin/env python3
"""
IDE Config Generator

Generates IDE-specific configuration files (Antigravity workflows, Cursor rules, etc.)
from the central AgentQMS source of truth.
"""

import os
import sys
from pathlib import Path
import shutil

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from AgentQMS.agent_tools.utils.paths import get_project_root

PROJECT_ROOT = get_project_root()
AGENT_WORKFLOWS_DIR = PROJECT_ROOT / ".agent" / "workflows"
CURSOR_DIR = PROJECT_ROOT / ".cursor"
COPILOT_DIR = PROJECT_ROOT / ".copilot" / "context"

def generate_antigravity_workflows():
    """Generates Antigravity .md workflows from AgentQMS templates."""
    print("Generating Antigravity workflows...")
    AGENT_WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

    # Define workflows (Source of Truth)
    workflows = {
        "create_plan.md": """---
description: Create a new implementation plan using AgentQMS
---
1. Determine the appropriate NAME (slug) and TITLE for the plan.
2. Run the creation command:
   ```bash
   cd AgentQMS/interface && make create-plan NAME=plan-slug TITLE="Plan Title"
   ```
   *Example: `cd AgentQMS/interface && make create-plan NAME=auth-refactor TITLE="Authentication System Refactor"`*
3. The plan will be created in `docs/artifacts/`. Use `view_file` to read it and fill in the details.
""",
        "validate_artifacts.md": """---
description: Validate all artifacts using AgentQMS compliance tools
---
1. Run the validation suite:
   ```bash
   cd AgentQMS/interface && make validate
   ```
2. If errors are reported, fix them or run `make compliance` for more details.
""",
        "agent_qms_status.md": """---
description: Check the status of the AgentQMS framework and available tools
---
// turbo
1. Check framework status:
   ```bash
   cd AgentQMS/interface && make status
   ```
2. Discover available tools:
   ```bash
   cd AgentQMS/interface && make discover
   ```
"""
    }

    for filename, content in workflows.items():
        path = AGENT_WORKFLOWS_DIR / filename
        path.write_text(content)
        print(f"  - Created {path}")

def generate_cursor_config():
    """Generates Cursor instructions."""
    print("Generating Cursor config...")
    CURSOR_DIR.mkdir(parents=True, exist_ok=True)

    # Simple consolidation of the SST
    content = """---
title: AgentQMS – Cursor Instructions
generated_by: AgentQMS/agent_tools/utilities/generate_ide_configs.py
---

## Core Rules (AgentQMS)

1. **Source of Truth**: Read `AgentQMS/knowledge/agent/system.md` first.
2. **No Manual Artifacts**: Always use `cd AgentQMS/interface && make create-*`.
3. **Validation**: Run `make validate` after edits.
4. **Context**: Use `make context` to load relevant docs.

## Workflows
- Create Plan: `make create-plan NAME=foo TITLE="Bar"`
- Validate: `make validate`
- Status: `make status`

For full details, run `make help`.
"""

    path = CURSOR_DIR / "instructions.md"
    path.write_text(content)
    print(f"  - Created {path}")

def generate_claude_config():
    """Generates Claude instructions."""
    print("Generating Claude config...")
    CLAUDE_DIR = PROJECT_ROOT / ".claude"
    CLAUDE_DIR.mkdir(parents=True, exist_ok=True)

    # Context for Claude
    content = """# AgentQMS Project Instructions

## Core Rules
1. **Source of Truth**: Read `AgentQMS/knowledge/agent/system.md` first.
2. **Tools First**: Do not create artifacts manually. Use the tools.
3. **Validation**: Always run validation commands after making changes.

## Primary Commands
- **Create Plan**: `cd AgentQMS/interface && make create-plan NAME=slug TITLE="Title"`
- **Validate**: `cd AgentQMS/interface && make validate`
- **Status**: `cd AgentQMS/interface && make status`

## Architecture
See `.agentqms/state/architecture.yaml` for component maps.
"""
    path = CLAUDE_DIR / "project_instructions.md"
    path.write_text(content)
    print(f"  - Created {path}")

def generate_copilot_config():
    """Generates GitHub Copilot instructions."""
    print("Generating Copilot config...")
    GITHUB_DIR = PROJECT_ROOT / ".github"
    GITHUB_DIR.mkdir(parents=True, exist_ok=True)

    content = """# AgentQMS Copilot Instructions

You are working in an AgentQMS-enabled project.

## Critical Rules
1. **Discovery**: Read `.copilot/context/agentqms-overview.md` and `.copilot/context/tool-catalog.md`.
2. **Artifacts**: NEVER create `docs/artifacts/*` files manually.
   - Use: `cd AgentQMS/interface && make create-plan` (or similar)
3. **Safety**: Run `make validate` before asking the user to review.

## Context
- Tool Registry: `.copilot/context/tool-registry.json`
- Workflows: `.copilot/context/workflow-triggers.yaml`
"""
    path = GITHUB_DIR / "copilot-instructions.md"
    path.write_text(content)
    print(f"  - Created {path}")

def main():
    try:
        generate_antigravity_workflows()
        generate_cursor_config()
        generate_claude_config()
        generate_copilot_config()

        print("\n✅ IDE Configs Synced (Antigravity, Cursor, Claude, Copilot)")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
