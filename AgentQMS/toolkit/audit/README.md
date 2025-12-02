# Audit Framework Tools

**Version**: 1.0  
**Date**: 2025-11-09

## Overview

Tools for generating, validating, and managing framework audit documents following the audit framework methodology.

---

## Tools

### 1. Audit Generator (`audit_generator.py`)

Generates audit documents from templates by replacing placeholders.

**Usage**:
```bash
# Initialize complete audit
python AgentQMS/agent_tools/audit/audit_generator.py init \
    --framework-name "Framework Name" \
    --audit-date "2025-11-09" \
    --audit-scope "Complete Framework Audit" \
    --output-dir "docs/audit"

# Generate single document
python AgentQMS/agent_tools/audit/audit_generator.py generate \
    --template "00_audit_summary_template.md" \
    --framework-name "Framework Name" \
    --audit-date "2025-11-09" \
    --output "docs/audit/00_audit_summary.md"
```

**Makefile**:
```bash
make audit-init FRAMEWORK="Framework Name" DATE="2025-11-09" SCOPE="Scope description"
```

---

### 2. Audit Validator (`audit_validator.py`)

Validates audit documents for completeness, structure, and required sections.

**Usage**:
```bash
# Validate all documents in audit directory
python AgentQMS/agent_tools/audit/audit_validator.py validate \
    --audit-dir "docs/audit"

# Validate single document
python AgentQMS/agent_tools/audit/audit_validator.py validate \
    --document "docs/audit/00_audit_summary.md"
```

**Makefile**:
```bash
make audit-validate
```

---

### 3. Checklist Tool (`checklist_tool.py`)

Generates, tracks, and reports on audit framework checklists.

**Usage**:
```bash
# Generate checklist for phase
python AgentQMS/agent_tools/audit/checklist_tool.py generate \
    --phase "discovery" \
    --output "docs/audit/checklist_discovery.md"

# Update checklist item
python AgentQMS/agent_tools/audit/checklist_tool.py track \
    --checklist "docs/audit/checklist_discovery.md" \
    --item "Scan for broken dependencies" \
    --status "completed"

# Generate progress report
python AgentQMS/agent_tools/audit/checklist_tool.py report \
    --audit-dir "docs/audit"
```

**Makefile**:
```bash
make audit-checklist-generate PHASE="discovery"
make audit-checklist-report
```

---

## Quick Start

### 1. Initialize Audit

```bash
cd AgentQMS/agent_interface
make audit-init FRAMEWORK="My Framework" DATE="2025-11-09" SCOPE="Complete Audit"
```

This generates all audit document templates in `docs/audit/`.

### 2. Generate Checklists

```bash
make audit-checklist-generate PHASE="discovery"
make audit-checklist-generate PHASE="analysis"
make audit-checklist-generate PHASE="design"
make audit-checklist-generate PHASE="implementation"
make audit-checklist-generate PHASE="automation"
```

### 3. Validate Documents

```bash
make audit-validate
```

### 4. Track Progress

```bash
python AgentQMS/agent_tools/audit/checklist_tool.py track \
    --checklist "docs/audit/checklist_discovery.md" \
    --item "Scan for broken dependencies" \
    --status "completed"
```

### 5. Generate Report

```bash
make audit-checklist-report
```

---

## Workflow

1. **Initialize**: Generate all audit documents from templates
2. **Fill Content**: Complete each document following the methodology
3. **Generate Checklists**: Create checklists for each phase
4. **Track Progress**: Update checklist items as you complete tasks
5. **Validate**: Validate documents for completeness
6. **Report**: Generate progress reports

---

## Integration

### Makefile Targets

- `make audit-init` - Initialize complete audit
- `make audit-validate` - Validate audit documents
- `make audit-checklist-generate` - Generate checklist for phase
- `make audit-checklist-report` - Generate progress report

### Python API

All tools can be imported and used programmatically:

```python
from AgentQMS.agent_tools.audit.audit_generator import init_audit, generate_document
from AgentQMS.agent_tools.audit.audit_validator import validate_document, validate_audit
from AgentQMS.agent_tools.audit.checklist_tool import generate_checklist, update_checklist_item
```

---

## Error Handling

All tools provide clear error messages:

- **Template Not Found**: Lists available templates
- **Placeholder Missing**: Lists required placeholders
- **Invalid Document**: Describes validation errors
- **File Not Found**: Provides helpful suggestions

---

## Requirements

- Python 3.8+
- Pathlib (standard library)
- AgentQMS framework structure

---

**Last Updated**: 2025-11-09

