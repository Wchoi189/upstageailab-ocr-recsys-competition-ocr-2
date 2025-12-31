# Audit Framework

**Version**: 1.0  
**Date**: 2025-11-09  
**Status**: Active

## Overview

The Audit Framework provides a systematic methodology for conducting framework audits. It extracts reusable patterns from audit documents to create a standardized process for maintaining framework quality over time.

---

## Purpose

This framework enables:
- **Systematic Auditing**: Follow a proven methodology for framework audits
- **Reusable Patterns**: Use extracted patterns for future audits
- **Quality Maintenance**: Maintain framework quality over time
- **Documentation**: Create consistent audit documentation

---

## Structure

```
audit_framework/
├── README.md                       # This file
├── usage_guide.md                  # Practical execution guide
├── protocol/                       # Protocol documents
│   ├── 00_audit_protocol.md        # Main protocol overview
│   ├── 01_discovery_protocol.md    # Discovery phase methodology
│   ├── 02_analysis_protocol.md     # Analysis phase methodology
│   ├── 03_design_protocol.md       # Design phase methodology
│   ├── 04_implementation_protocol.md # Implementation phase methodology
│   └── 05_automation_protocol.md   # Automation phase methodology
├── templates/                      # Audit document templates
└── tools/                          # Tool documentation
    ├── README.md                   # Tool usage reference
    └── tool_architecture.md        # Tool architecture details
```

---

## Quick Start

1. **Read the Main Protocol**: Start with `protocol/00_audit_protocol.md`
2. **Follow Phase Protocols**: Use phase-specific protocols in order
3. **Use Templates**: Copy templates from `templates/` into `docs/audit/`
4. **Generate Documents**: Run `make audit-init FRAMEWORK="Name" DATE="YYYY-MM-DD" SCOPE="Scope"`
5. **Validate Output**: Run `make audit-validate` to ensure structure is correct
6. **Track Progress**: Generate checklists with `make audit-checklist-generate PHASE="discovery"`

---

## Methodology

The audit framework follows a five-phase methodology:

### Phase 1: Discovery
**Protocol**: `protocol/01_discovery_protocol.md`

**Purpose**: Identify all issues and removal candidates

**Deliverable**: `01_removal_candidates.md`

**Key Activities**:
- Issue identification
- Priority categorization
- Impact analysis
- Removal candidate documentation

---

### Phase 2: Analysis
**Protocol**: `protocol/02_analysis_protocol.md`

**Purpose**: Map workflows and identify pain points

**Deliverable**: `02_workflow_analysis.md`

**Key Activities**:
- Workflow mapping
- Pain point analysis
- Bottleneck identification
- Goal vs. implementation assessment

---

### Phase 3: Design
**Protocol**: `protocol/03_design_protocol.md`

**Purpose**: Propose solutions and define standards

**Deliverables**: 
- `03_restructure_proposal.md`
- `04_standards_specification.md`

**Key Activities**:
- Solution design
- Standards definition
- Design decision documentation
- Restructure planning

---

### Phase 4: Implementation
**Protocol**: `protocol/04_implementation_protocol.md`

**Purpose**: Create phased implementation plan

**Deliverable**: `[date]_IMPLEMENTATION_PLAN_[name].md`

**Key Activities**:
- Phase planning
- Success criteria definition
- Risk mitigation planning
- Timeline creation

---

### Phase 5: Automation
**Protocol**: `protocol/05_automation_protocol.md`

**Purpose**: Design self-maintaining mechanisms

**Deliverable**: `05_automation_recommendations.md`

**Key Activities**:
- Self-enforcing compliance design
- Validation automation design
- Proactive maintenance design
- Monitoring and alerts design

---

## Priority System

All issues are categorized using a four-tier priority system:

- **🔴 Critical (Blocking)**: Framework non-functional, must fix immediately
- **🟡 High Priority (Reusability)**: Prevents framework reuse, fix in Phase 2
- **🟠 Medium Priority (Maintainability)**: Technical debt, fix in Phase 3
- **🟢 Low Priority (Optimization)**: Nice-to-have improvements, fix in Phase 4

---

## Usage

### Starting a New Audit

1. **Create Audit Directory**: Create `docs/audit/` directory
2. **Read Protocols**: Review all phase protocols
3. **Generate Templates**: Run `make audit-init` to scaffold documents
4. **Start Discovery**: Follow `01_discovery_protocol.md`
5. **Continue Through Phases**: Follow protocols sequentially
6. **Track Progress**: Generate and update checklists per phase
7. **Validate Work**: Run `make audit-validate` routinely

### Using Templates

Templates are available in `templates/`:
- `00_audit_summary_template.md`
- `01_removal_candidates_template.md`
- `02_workflow_analysis_template.md`
- `03_restructure_proposal_template.md`
- `04_standards_specification_template.md`
- `05_automation_recommendations_template.md`

### Tooling

Phase 3 introduces supporting tools (see `tools/README.md` for details):

| Tool | Purpose | Make Target |
|------|---------|-------------|
| `audit_generator.py` | Generate audit documents from templates with placeholder replacement | `make audit-init` |
| `audit_validator.py` | Validate document completeness, sections, and frontmatter | `make audit-validate` |
| `checklist_tool.py` | Generate and track phase checklists, produce progress reports | `make audit-checklist-*` |

Refer to `tools/tool_architecture.md` for architectural details.

---

## Example Audit

The `docs/audit/` directory should contain the working set of audit documents generated from templates. Run `make audit-init` to scaffold the initial set, then update each file per the protocols. An example sequence:

1. `00_audit_summary.md` – summarize findings as you progress
2. `01_removal_candidates.md` – log discoveries from Phase 1
3. `02_workflow_analysis.md` – capture workflow maps and pain points
4. `03_restructure_proposal.md` / `04_standards_specification.md` – document design outcomes
5. `05_automation_recommendations.md` – finalize automation strategy

---

## Success Criteria

### Framework Success
- ✅ All phases documented
- ✅ Clear methodology flow
- ✅ Reusable templates available
- ✅ Checklists for each phase

### Audit Success
- ✅ All issues identified and categorized
- ✅ Workflows mapped and analyzed
- ✅ Solutions proposed with priorities
- ✅ Implementation plan created
- ✅ Automation strategy defined

---

## Related Documents

- **Usage Guide**: `AgentQMS/conventions/audit_framework/usage_guide.md`
- **Tool Reference**: `AgentQMS/conventions/audit_framework/tools/README.md`
- **Tool Architecture**: `AgentQMS/conventions/audit_framework/tools/tool_architecture.md`
- **Audit Tools (Implementation)**: `AgentQMS/agent_tools/audit/`

---

## Legacy Layout Note

Earlier versions of this framework used `project_conventions/audit_framework/` as the conventions path. The current containerized layout uses `AgentQMS/conventions/audit_framework/`. If you encounter references to the legacy path in older documentation, treat them as historical.

---

**Last Updated**: 2025-11-25  
**Next Review**: After next audit using this framework

