# Audit Framework Tools Architecture

**Version**: 1.0  
**Date**: 2025-11-09

## Overview

This document defines the architecture for audit framework tools that support the audit methodology.

---

## Tool Requirements

### Functional Requirements

1. **Audit Generator Tool**
   - Load templates from `AgentQMS/conventions/audit_framework/templates/`
   - Replace placeholders with user-provided values
   - Generate audit documents in `docs/audit/`
   - Validate generated documents

2. **Audit Validator Tool**
   - Validate audit document completeness
   - Check required sections
   - Verify document structure
   - Validate against protocol requirements

3. **Checklist Tools**
   - Generate checklists from protocol documents
   - Track checklist progress
   - Generate progress reports
   - Validate checklist completion

### Non-Functional Requirements

- **Language**: Python 3.8+
- **Dependencies**: Minimal (pathlib, yaml if needed)
- **Location**: `AgentQMS/agent_tools/audit/`
- **CLI Interface**: Command-line interface with argparse
- **Error Handling**: Clear error messages
- **Documentation**: Help text and usage examples

---

## Tool Interface Design

### Audit Generator Tool

**Command**: `python AgentQMS/agent_tools/audit/audit_generator.py`

**Usage**:
```bash
# Generate all audit documents
python AgentQMS/agent_tools/audit/audit_generator.py init \
    --framework-name "Framework Name" \
    --audit-date "2025-11-09" \
    --audit-scope "Scope description" \
    --output-dir "docs/audit"

# Generate specific document
python AgentQMS/agent_tools/audit/audit_generator.py generate \
    --template "00_audit_summary_template.md" \
    --framework-name "Framework Name" \
    --output "docs/audit/00_audit_summary.md"
```

**Functions**:
- `load_template(template_path: Path) -> str`
- `replace_placeholders(content: str, values: dict) -> str`
- `generate_document(template_name: str, values: dict, output_path: Path) -> Path`
- `init_audit(framework_name: str, audit_date: str, audit_scope: str, output_dir: Path) -> None`

---

### Audit Validator Tool

**Command**: `python AgentQMS/agent_tools/audit/audit_validator.py`

**Usage**:
```bash
# Validate all audit documents
python AgentQMS/agent_tools/audit/audit_validator.py validate \
    --audit-dir "docs/audit"

# Validate specific document
python AgentQMS/agent_tools/audit/audit_validator.py validate \
    --document "docs/audit/00_audit_summary.md"
```

**Functions**:
- `validate_document(document_path: Path) -> ValidationResult`
- `check_required_sections(content: str, document_type: str) -> list[str]`
- `validate_structure(document_path: Path) -> bool`
- `validate_completeness(audit_dir: Path) -> ValidationReport`

---

### Checklist Tool

**Command**: `python AgentQMS/agent_tools/audit/checklist_tool.py`

**Usage**:
```bash
# Generate checklist for phase
python AgentQMS/agent_tools/audit/checklist_tool.py generate \
    --phase "discovery" \
    --output "docs/audit/checklist_discovery.md"

# Track progress
python AgentQMS/agent_tools/audit/checklist_tool.py track \
    --checklist "docs/audit/checklist_discovery.md" \
    --item "Task 1.1" \
    --status "completed"

# Generate report
python AgentQMS/agent_tools/audit/checklist_tool.py report \
    --audit-dir "docs/audit"
```

**Functions**:
- `generate_checklist(phase: str, output_path: Path) -> Path`
- `update_checklist_item(checklist_path: Path, item_id: str, status: str) -> None`
- `generate_progress_report(audit_dir: Path) -> str`
- `validate_checklist_completion(checklist_path: Path) -> bool`

---

## Tool Integration

### Integration Points

1. **Makefile Integration**
   - Add targets to `AgentQMS/agent_interface_interface/Makefile`
   - Commands: `make audit-init`, `make audit-validate`, `make audit-checklist`

2. **CLI Integration**
   - Unified CLI interface
   - Help documentation
   - Error handling

3. **Validation Integration**
   - Integrate with existing validation tools
   - Use framework validation infrastructure

---

## File Structure

```
AgentQMS/
├── agent_tools/
│   └── audit/                          # Audit tools (canonical)
│       ├── __init__.py
│       ├── audit_generator.py          # Template-based document generation
│       ├── audit_validator.py          # Document validation
│       ├── checklist_tool.py           # Checklist management
│       └── utils.py                    # Shared utilities
│
└── conventions/
    └── audit_framework/
        ├── protocol/                   # Protocol documents
        ├── templates/                  # Document templates
        └── tools/                      # Tool documentation
            └── tool_architecture.md    # This file
```

> **Legacy Note**: Earlier versions of this framework used `project_conventions/audit_framework/` as the conventions path. The current containerized layout uses `AgentQMS/conventions/audit_framework/`.

---

## Implementation Plan

### Step 1: Create Directory Structure
- Create `AgentQMS/agent_tools/audit/`
- Create `__init__.py`

### Step 2: Implement Audit Generator
- Template loading
- Placeholder replacement
- Document generation
- Basic validation

### Step 3: Implement Audit Validator
- Document structure validation
- Required sections checking
- Completeness validation

### Step 4: Implement Checklist Tool
- Checklist generation
- Progress tracking
- Report generation

### Step 5: Integration
- Makefile targets
- CLI interface
- Documentation

---

## Error Handling

### Error Types

1. **Template Not Found**
   - Error: Template file missing
   - Solution: Check template path, verify template exists

2. **Placeholder Missing**
   - Error: Required placeholder not provided
   - Solution: Provide all required values

3. **Invalid Document Structure**
   - Error: Document doesn't match expected structure
   - Solution: Check document format, regenerate if needed

4. **Validation Failure**
   - Error: Document fails validation
   - Solution: Review errors, fix issues

---

## Testing Strategy

### Unit Tests
- Template loading
- Placeholder replacement
- Validation logic

### Integration Tests
- End-to-end document generation
- Validation workflow
- Checklist tracking

### Manual Testing
- Generate sample audit
- Validate generated documents
- Test checklist functionality

---

**Last Updated**: 2025-11-09

