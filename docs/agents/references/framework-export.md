---
title: "Framework Export Reference"
author: "ai-agent"
timestamp: "2025-11-16 17:15 KST"
branch: "main"
status: "active"
tags: ["reference", "framework", "export"]
---

# Framework Export Reference

**Purpose:** Reference guide for AI agents maintaining and updating the framework export script.

**Location:** `scripts/agent_tools/core/export_framework.py`

## Overview

The `export_framework.py` script creates a reusable, project-agnostic package of the AI collaboration framework. It filters out project-specific content and creates a clean export suitable for use in other projects.

## Export Structure

The exported framework follows this structure:

```
ai-collaboration-framework/
├── README.md                    # Comprehensive setup guide
├── LICENSE                      # License file
├── .cursor/rules/               # Cursor IDE rules (sanitized)
├── agent_qms/                   # Complete framework package
│   ├── q-manifest.yaml
│   ├── templates/
│   ├── schemas/
│   └── toolbelt/
├── docs/agents/                 # AI agent documentation (sanitized)
├── scripts/agent_tools/         # Project-agnostic scripts only
├── artifacts/                   # Empty directory structure
└── examples/                    # Example artifacts
```

## Maintenance Guidelines

### When to Update the Export Script

Update `export_framework.py` when:

1. **New Artifact Types Added**
   - New templates in `agent_qms/templates/`
   - New schemas in `agent_qms/schemas/`
   - New entries in `agent_qms/q-manifest.yaml`
   - **Action:** No changes needed - `_copy_agent_qms()` copies everything automatically

2. **New Documentation Added**
   - New files in `docs/agents/`
   - **Action:** Update `_copy_docs_agents()` file list if the file should be exported

3. **New Scripts Added**
   - New directories in `scripts/agent_tools/`
   - **Action:** Update `_copy_scripts_agent_tools()` `dirs_to_copy` list if project-agnostic

4. **New Project-Specific Content**
   - New project-specific references that need sanitization
   - **Action:** Update `_sanitize_content()` method

5. **Export Structure Changes**
   - New directories needed in export
   - **Action:** Update `_create_directory_structure()`

### Key Methods

#### `_copy_agent_qms()`
- **Purpose:** Copies the complete AgentQMS framework package
- **What it does:** Recursively copies all files from `agent_qms/`, sanitizing content
- **When to modify:** Rarely - automatically includes all templates, schemas, and toolbelt code
- **Note:** Skips binary files automatically

#### `_copy_docs_agents()`
- **Purpose:** Copies AI agent documentation files
- **What it does:** Copies specific files listed in `files_to_copy`
- **When to modify:** When adding new documentation files that should be exported
- **Files currently exported:**
  - `system.md`
  - `index.md`
  - `protocols/governance.md`
  - `protocols/development.md`
  - `protocols/components.md`
  - `protocols/configuration.md`

#### `_copy_scripts_agent_tools()`
- **Purpose:** Copies project-agnostic automation scripts
- **What it does:** Copies directories listed in `dirs_to_copy`
- **When to modify:** When adding new project-agnostic script directories
- **Directories currently exported:**
  - `core/` - Core automation
  - `compliance/` - Validation tools
  - `documentation/` - Documentation management
  - `utilities/` - Helper functions
- **Excluded (project-specific):**
  - `ocr/` - OCR-specific tools
  - `maintenance/` - Project-specific migrations

#### `_sanitize_content()`
- **Purpose:** Removes project-specific references from exported content
- **What it does:** Uses regex to replace project-specific terms with generic ones
- **When to modify:** When new project-specific terms need to be sanitized
- **Current sanitizations:**
  - "OCR-specific" → "project-specific"
  - "OCR Receipt Text Detection" → "Your Project"

#### `_create_directory_structure()`
- **Purpose:** Creates the export directory structure
- **What it does:** Creates all necessary directories and `.gitkeep` files
- **When to modify:** When export structure needs new directories

#### `_create_examples()`
- **Purpose:** Creates example artifacts demonstrating framework usage
- **What it does:** Generates example assessment and data contract files
- **When to modify:** When example format changes or new examples needed

#### `_create_readme()`
- **Purpose:** Generates comprehensive README for exported framework
- **What it does:** Creates setup instructions, usage examples, and documentation
- **When to modify:** When framework usage changes significantly

## Common Update Scenarios

### Scenario 1: Adding a New Artifact Type

**Steps:**
1. Add artifact type to `agent_qms/q-manifest.yaml`
2. Create template in `agent_qms/templates/`
3. Create schema in `agent_qms/schemas/`
4. **No export script changes needed** - automatically included

### Scenario 2: Adding New Documentation

**Steps:**
1. Add file to `docs/agents/`
2. If file should be exported, add to `files_to_copy` in `_copy_docs_agents()`
3. Test export to verify file is included

### Scenario 3: Adding New Project-Agnostic Scripts

**Steps:**
1. Create new directory in `scripts/agent_tools/`
2. Add directory name to `dirs_to_copy` in `_copy_scripts_agent_tools()`
3. Test export to verify scripts are included

### Scenario 4: Project-Specific Content Leaking

**Steps:**
1. Identify project-specific terms/references
2. Add sanitization rule to `_sanitize_content()`
3. Test export to verify sanitization works

## Testing the Export

After making changes:

```bash
# Test export
python scripts/agent_tools/core/export_framework.py --output-dir /tmp/test_export

# Verify structure
ls -la /tmp/test_export/ai-collaboration-framework/

# Check key files exist
test -f /tmp/test_export/ai-collaboration-framework/README.md && echo "✓ README exists"
test -f /tmp/test_export/ai-collaboration-framework/agent_qms/q-manifest.yaml && echo "✓ Manifest exists"
test -d /tmp/test_export/ai-collaboration-framework/artifacts/data_contracts && echo "✓ Data contracts dir exists"

# Clean up
rm -rf /tmp/test_export
```

## Validation Checklist

Before committing export script changes:

- [ ] Export completes without errors
- [ ] All expected directories are created
- [ ] All expected files are included
- [ ] Project-specific content is sanitized
- [ ] README is generated correctly
- [ ] Examples are created correctly
- [ ] No binary files are included (intentionally)
- [ ] No project-specific scripts are included

## File Filtering Rules

**Included:**
- All text files (`.md`, `.yaml`, `.json`, `.py`, etc.)
- Files in project-agnostic directories
- Files explicitly listed in copy methods

**Excluded:**
- Binary files (`.pyc`, images, etc.) - automatically skipped
- Files starting with `.` (hidden files)
- Project-specific directories (`ocr/`, `maintenance/`, etc.)
- Actual artifacts (only structure is exported)

## Sanitization Guidelines

**What to sanitize:**
- Project names (e.g., "OCR Receipt Text Detection")
- Project-specific terminology
- Hard-coded paths specific to this project
- Project-specific examples

**What NOT to sanitize:**
- Generic framework terminology
- Standard directory structures
- Framework conventions
- Generic examples

## Related Documentation

- `docs/agents/system.md` - Main AI agent system documentation
- `agent_qms/q-manifest.yaml` - Framework manifest
- `artifacts/assessments/2025-11-16_1654_ai-collaboration-framework-extraction-and-standardization-assessment.md` - Original assessment

## Quick Reference

**Export command:**
```bash
python scripts/agent_tools/core/export_framework.py [--output-dir OUTPUT_DIR]
```

**Default output:** `./framework_export/ai-collaboration-framework/`

**Key files to check when updating:**
1. `_copy_docs_agents()` - Documentation file list
2. `_copy_scripts_agent_tools()` - Script directory list
3. `_sanitize_content()` - Sanitization rules
4. `_create_directory_structure()` - Directory structure

