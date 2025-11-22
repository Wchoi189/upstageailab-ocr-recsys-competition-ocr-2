#!/usr/bin/env python3
"""
Export AI Collaboration Framework

This script exports the project-agnostic AI collaboration framework components
for reuse in other projects. It creates a clean, sanitized export package
with all necessary files and documentation.

**For AI Agents:**
- See `docs/agents/references/framework-export.md` for maintenance guidelines
- This script should be updated when new artifact types, templates, or framework
  components are added
- The export process filters out project-specific content automatically

**Key Components Exported:**
1. `.cursor/rules/` - Cursor IDE rules (sanitized)
2. `agent_qms/` - Complete framework package (templates, schemas, toolbelt)
3. `docs/agents/` - AI agent documentation (sanitized)
4. `scripts/agent_tools/` - Project-agnostic automation scripts only
5. `artifacts/` - Empty directory structure with README
6. `examples/` - Example artifacts demonstrating usage

**Maintenance Notes:**
- When adding new artifact types: Update `_copy_agent_qms()` to ensure new
  templates/schemas are included
- When adding new docs: Update `_copy_docs_agents()` file list
- When adding new scripts: Update `_copy_scripts_agent_tools()` to include
  project-agnostic directories
- Sanitization: Update `_sanitize_content()` to remove new project-specific
  references

Usage:
    python scripts/agent_tools/core/export_framework.py [--output-dir OUTPUT_DIR]
"""

import argparse
import re
from pathlib import Path


class FrameworkExporter:
    """Exports the AI collaboration framework as a reusable package."""

    def __init__(self, project_root: Path, output_dir: Path):
        self.project_root = project_root
        self.output_dir = output_dir
        self.export_root = output_dir / "ai-collaboration-framework"

    def export(self):
        """Perform the complete export process."""
        print(f"Exporting AI Collaboration Framework to: {self.export_root}")

        # Create export directory structure
        self._create_directory_structure()

        # Copy project-agnostic files
        self._copy_cursor_rules()
        self._copy_agent_qms()
        self._copy_docs_agents()
        self._copy_scripts_agent_tools()
        self._create_artifacts_structure()
        self._create_examples()
        self._create_readme()
        self._create_license()

        print(f"\n✓ Export complete! Framework available at: {self.export_root}")
        print("\nNext steps:")
        print("1. Review the exported framework")
        print("2. Customize for your project if needed")
        print("3. Copy to your new project root")
        print("4. Update .cursor/rules if using Cursor IDE")

    def _create_directory_structure(self):
        """Create the export directory structure."""
        directories = [
            self.export_root / ".cursor" / "rules",
            self.export_root / "agent_qms" / "templates",
            self.export_root / "agent_qms" / "schemas",
            self.export_root / "agent_qms" / "toolbelt",
            self.export_root / "docs" / "agents" / "protocols",
            self.export_root / "docs" / "agents" / "references",
            self.export_root / "scripts" / "agent_tools" / "core",
            self.export_root / "scripts" / "agent_tools" / "compliance",
            self.export_root / "scripts" / "agent_tools" / "documentation",
            self.export_root / "scripts" / "agent_tools" / "utilities",
            self.export_root / "artifacts" / "assessments",
            self.export_root / "artifacts" / "implementation_plans",
            self.export_root / "artifacts" / "data_contracts",
            self.export_root / "examples",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            # Create .gitkeep files
            (directory / ".gitkeep").touch()

    def _copy_cursor_rules(self):
        """Copy and sanitize Cursor rules."""
        source = self.project_root / ".cursor" / "rules" / "prompts-artifacts-guidelines.mdc"
        dest = self.export_root / ".cursor" / "rules" / "prompts-artifacts-guidelines.mdc"

        if source.exists():
            content = source.read_text()
            # Sanitize: Remove project-specific references
            content = self._sanitize_content(content)
            dest.write_text(content)
            print("✓ Copied .cursor/rules/")

    def _copy_agent_qms(self):
        """Copy the complete AgentQMS framework."""
        source_dir = self.project_root / "agent_qms"
        dest_dir = self.export_root / "agent_qms"

        if source_dir.exists():
            # Copy all files recursively
            for item in source_dir.rglob("*"):
                if item.is_file() and not item.name.startswith("."):
                    rel_path = item.relative_to(source_dir)
                    dest_file = dest_dir / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)

                    # Try to read as text, skip binary files
                    try:
                        content = item.read_text(encoding="utf-8")
                        content = self._sanitize_content(content)
                        dest_file.write_text(content, encoding="utf-8")
                    except (UnicodeDecodeError, UnicodeError):
                        # Skip binary files (e.g., .pyc, images, etc.)
                        continue

            print("✓ Copied agent_qms/")

    def _copy_docs_agents(self):
        """Copy AI agent documentation (sanitized)."""
        source_dir = self.project_root / "docs" / "agents"
        dest_dir = self.export_root / "docs" / "agents"

        # Files to copy
        files_to_copy = [
            "system.md",
            "index.md",
            "protocols/governance.md",
            "protocols/development.md",
            "protocols/components.md",
            "protocols/configuration.md",
        ]

        for file_path in files_to_copy:
            source = source_dir / file_path
            if source.exists():
                dest = dest_dir / file_path
                dest.parent.mkdir(parents=True, exist_ok=True)

                content = source.read_text()
                content = self._sanitize_content(content)
                dest.write_text(content)

        print("✓ Copied docs/agents/")

    def _copy_scripts_agent_tools(self):
        """Copy project-agnostic agent tools."""
        source_dir = self.project_root / "scripts" / "agent_tools"
        dest_dir = self.export_root / "scripts" / "agent_tools"

        # Directories to copy (project-agnostic only)
        dirs_to_copy = ["core", "compliance", "documentation", "utilities"]

        for dir_name in dirs_to_copy:
            source_subdir = source_dir / dir_name
            if source_subdir.exists():
                dest_subdir = dest_dir / dir_name
                self._copy_directory_recursive(source_subdir, dest_subdir)

        print("✓ Copied scripts/agent_tools/ (project-agnostic only)")

    def _copy_directory_recursive(self, source: Path, dest: Path):
        """Recursively copy directory, sanitizing files."""
        dest.mkdir(parents=True, exist_ok=True)

        for item in source.rglob("*"):
            if item.is_file() and not item.name.startswith("."):
                rel_path = item.relative_to(source)
                dest_file = dest / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                # Try to read as text, skip binary files
                try:
                    content = item.read_text(encoding="utf-8")
                    content = self._sanitize_content(content)
                    dest_file.write_text(content, encoding="utf-8")
                except (UnicodeDecodeError, UnicodeError):
                    # Skip binary files
                    continue

    def _create_artifacts_structure(self):
        """Create empty artifacts directory structure."""
        # Already created in _create_directory_structure
        # Add README to explain structure
        readme_content = """# Artifacts Directory

This directory contains AI-generated artifacts managed by AgentQMS.

## Structure

- `assessments/` - Assessment documents
- `implementation_plans/` - Implementation plan documents
- `data_contracts/` - Data contract documents

## Usage

Artifacts are created using the AgentQMS toolbelt. See `docs/agents/system.md` for usage instructions.
"""
        (self.export_root / "artifacts" / "README.md").write_text(readme_content)
        print("✓ Created artifacts/ structure")

    def _create_examples(self):
        """Create example artifacts."""
        examples_dir = self.export_root / "examples"

        # Example assessment
        example_assessment = """---
title: "Example Assessment"
author: "ai-agent"
timestamp: "2025-01-01 12:00 KST"
branch: "main"
status: "draft"
tags: ["example"]
---

## 1. Summary

This is an example assessment artifact demonstrating the standardized frontmatter format.

## 2. Assessment

[Assessment content goes here]

## 3. Recommendations

[Recommendations go here]
"""
        (examples_dir / "example-assessment.md").write_text(example_assessment)

        # Example data contract
        example_data_contract = """---
title: "Example Data Contract"
author: "ai-agent"
timestamp: "2025-01-01 12:00 KST"
branch: "main"
status: "draft"
tags: ["data-contract", "example"]
---

## 1. Overview

**Purpose:** Define the data contract for [area name].

**Scope:** [Scope description]

## 2. Input Contract

[Input contract definition]

## 3. Output Contract

[Output contract definition]

## 4. Validation Rules

[Validation rules]

## 5. Examples

[Examples]
"""
        (examples_dir / "example-data-contract.md").write_text(example_data_contract)

        print("✓ Created example artifacts")

    def _create_readme(self):
        """Create comprehensive README for the framework."""
        readme_content = """# AI Collaboration Framework

A reusable framework for facilitating AI-to-human collaboration through standardized artifact generation, directory scaffolding, and project conventions.

## Overview

This framework provides:

- **AgentQMS (Quality Management System)**: Python toolbelt for programmatic artifact creation
- **Standardized Artifacts**: Templates and schemas for assessments, implementation plans, bug reports, and data contracts
- **Documentation Structure**: AI agent instructions and protocols
- **Automation Scripts**: Core tools for artifact management and validation
- **Cursor IDE Integration**: Rules and guidelines for AI agents

## Quick Start

### 1. Installation

Copy this framework to your project root:

```bash
cp -r ai-collaboration-framework/* /path/to/your/project/
```

### 2. Setup

1. **Install Python dependencies:**
   ```bash
   pip install pyyaml jinja2 jsonschema
   ```

2. **Configure Cursor IDE (optional):**
   - The `.cursor/rules/` directory contains rules for Cursor IDE
   - Cursor will automatically pick up these rules

3. **Verify installation:**
   ```python
   from agent_qms.toolbelt import AgentQMSToolbelt

   toolbelt = AgentQMSToolbelt()
   print(toolbelt.list_artifact_types())
   # Should output: ['implementation_plan', 'assessment', 'bug_report', 'data_contract']
   ```

### 3. Create Your First Artifact

```python
from agent_qms.toolbelt import AgentQMSToolbelt

toolbelt = AgentQMSToolbelt()
artifact_path = toolbelt.create_artifact(
    artifact_type="assessment",
    title="My First Assessment",
    content="## Summary\\nThis is my assessment.",
    author="ai-agent",
    tags=["example"]
)

print(f"Created: {artifact_path}")
```

## Framework Components

### AgentQMS

The core framework package located in `agent_qms/`:

- `q-manifest.yaml` - Central configuration defining artifact types, templates, and schemas
- `templates/` - Markdown templates for each artifact type
- `schemas/` - JSON schemas for frontmatter validation
- `toolbelt/` - Python toolbelt for programmatic artifact creation

### Artifact Types

1. **Assessment** - Evaluation of a specific aspect of the system
2. **Implementation Plan** - Detailed plan for implementing features or changes
3. **Bug Report** - Bug documentation with root cause analysis
4. **Data Contract** - Data structure definitions with validation rules

### Standardized Frontmatter

All artifacts use a standardized frontmatter format:

```yaml
---
title: "Artifact Title"
author: "ai-agent"
timestamp: "2025-01-01 12:00 KST"  # KST format with hour and minute
branch: "main"  # Git branch name
status: "draft"  # draft | in-progress | completed
tags: []
---
```

### Documentation

- `docs/agents/` - AI agent instructions and protocols
  - `system.md` - Single source of truth for AI agent behavior
  - `protocols/` - Development, governance, and component protocols

### Automation Scripts

- `scripts/agent_tools/core/` - Core automation (artifact creation, discovery)
- `scripts/agent_tools/compliance/` - Validation and monitoring
- `scripts/agent_tools/documentation/` - Documentation management
- `scripts/agent_tools/utilities/` - Helper functions

## Usage Examples

### Creating an Assessment

```python
from agent_qms.toolbelt import AgentQMSToolbelt

toolbelt = AgentQMSToolbelt()
artifact_path = toolbelt.create_artifact(
    artifact_type="assessment",
    title="Performance Analysis",
    content="## Summary\\n...",
    author="ai-agent",
    tags=["performance", "analysis"]
)
```

### Creating a Data Contract

```python
toolbelt = AgentQMSToolbelt()
artifact_path = toolbelt.create_artifact(
    artifact_type="data_contract",
    title="API Data Contract",
    content="## Overview\\n...",
    author="ai-agent",
    tags=["api", "contract"]
)
```

### Validating an Artifact

```python
toolbelt = AgentQMSToolbelt()
is_valid = toolbelt.validate_artifact("artifacts/assessments/my-assessment.md")
```

## Customization

### Adding New Artifact Types

1. Add entry to `agent_qms/q-manifest.yaml`
2. Create template in `agent_qms/templates/`
3. Create schema in `agent_qms/schemas/`
4. Update toolbelt if needed (usually not required)

### Modifying Templates

Templates use Jinja2 syntax. Modify templates in `agent_qms/templates/` as needed.

## Migration Guide

If you have existing artifacts with the old frontmatter format:

1. **Old format** (deprecated):
   ```yaml
   ---
   title: "..."
   date: "2025-01-01"
   timestamp: "2025-01-01 12:00 KST"
   ---
   ```

2. **New format** (current):
   ```yaml
   ---
   title: "..."
   timestamp: "2025-01-01 12:00 KST"
   branch: "main"
   ---
   ```

**Changes:**
- Removed `date` field (redundant, date is in timestamp)
- Added `branch` field (required, git branch name)
- `timestamp` format unchanged (YYYY-MM-DD HH:MM KST)

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]

## Support

For issues or questions, see `docs/agents/system.md` for detailed documentation.
"""
        (self.export_root / "README.md").write_text(readme_content)
        print("✓ Created README.md")

    def _create_license(self):
        """Create a placeholder LICENSE file."""
        license_content = """MIT License

[Add your license text here]

Copyright (c) 2025
"""
        (self.export_root / "LICENSE").write_text(license_content)
        print("✓ Created LICENSE")

    def _sanitize_content(self, content: str) -> str:
        """Sanitize content by removing project-specific references."""
        # Remove OCR-specific references
        content = re.sub(r"OCR[-\s]?specific", "project-specific", content, flags=re.IGNORECASE)
        content = re.sub(r"OCR Receipt Text Detection", "Your Project", content, flags=re.IGNORECASE)

        # Remove specific file paths that are project-specific
        # Keep generic patterns

        return content


def main():
    parser = argparse.ArgumentParser(description="Export AI Collaboration Framework")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "framework_export",
        help="Output directory for the exported framework (default: ./framework_export)",
    )

    args = parser.parse_args()

    # Determine project root (assume script is in scripts/agent_tools/core/)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent

    exporter = FrameworkExporter(project_root, args.output_dir)
    exporter.export()


if __name__ == "__main__":
    main()
