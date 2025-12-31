# Audit Framework Usage Guide

**Version**: 1.0  
**Date**: 2025-11-09  
**Status**: Draft

---

## 1. Overview

This guide explains how to run the audit framework end-to-end, including prerequisites, required commands, template usage, and troubleshooting tips. It complements the protocol documents by focusing on practical execution.

---

## 2. Prerequisites

- AgentQMS framework installed (`AgentQMS/` directory available)
- Python 3.8+ environment
- Writable `docs/audit/` directory for outputs
- Familiarity with protocol documents in `AgentQMS/conventions/audit_framework/protocol/`

---

## 3. Setup Checklist

1. `cd AgentQMS/agent_interface`
2. Run `make discover` to verify tooling
3. Ensure `docs/audit/` exists (create if missing)
4. Confirm templates and protocols are up to date (`git pull` / sync)

---

## 4. End-to-End Workflow

| Phase | Primary Actions | Outputs |
|-------|-----------------|---------|
| Discovery | Follow `01_discovery_protocol.md`, use checklist, gather evidence | `01_removal_candidates.md` |
| Analysis | Map workflows, document pain points, use diagrams | `02_workflow_analysis.md` |
| Design | Propose solutions, define standards, capture decisions | `03_restructure_proposal.md`, `04_standards_specification.md` |
| Implementation | Build phased plan, define success criteria, document risks | Implementation plan artifact |
| Automation | Design validation & monitoring, document automation strategy | `05_automation_recommendations.md` |

---

## 5. Template Usage

1. **Generate documents automatically**  
   ```bash
   make audit-init FRAMEWORK="Framework Name" DATE="2025-11-09" SCOPE="Full Framework Audit"
   ```
   This copies all templates into `docs/audit/` and fills basic metadata.

2. **Manual copy (optional)**  
- Copy files from `AgentQMS/conventions/audit_framework/templates/`
   - Replace placeholders (`{{FRAMEWORK_NAME}}`, `{{AUDIT_DATE}}`, etc.)

3. **Validation**  
   ```bash
   make audit-validate
   ```
   Ensures placeholders are resolved and sections are complete.

---

## 6. Tooling Reference

| Command | Description | Notes |
|---------|-------------|-------|
| `make audit-init` | Generate all audit documents | Overwrite existing files with `--overwrite` (see CLI help) |
| `make audit-validate` | Validate documents for completeness | Provides detailed error output |
| `make audit-checklist-generate PHASE="discovery"` | Create checklist for a phase | Files saved to `docs/audit/checklist_<phase>.md` |
| `make audit-checklist-report` | Summarize checklist progress | Outputs report to stdout |

Refer to `AgentQMS/agent_tools/audit/README.md` for additional CLI examples.

---

## 7. Checklist Workflow

1. Generate all phase checklists.
2. Update items as they are completed:  
   ```bash
   python AgentQMS/agent_tools/audit/checklist_tool.py track \
       --checklist docs/audit/checklist_discovery.md \
       --item "Scan for broken dependencies" \
       --status completed
   ```
3. Run `make audit-checklist-report` before each stand-up or review.

---

## 8. Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| `Template not found` | Wrong template name or outdated repo | Check `AgentQMS/conventions/audit_framework/templates/` |
| `Missing placeholder` | Required value not provided | Re-run command with `--framework-name`, `--audit-date`, etc. |
| `Missing section` | Template not fully filled | Compare document with protocol requirements |
| `Checklist item missing` | Text mismatch | Copy text exactly as it appears in checklist |

---

## 9. References

- Protocols: `AgentQMS/conventions/audit_framework/protocol/`
- Templates: `AgentQMS/conventions/audit_framework/templates/`
- Tools: `AgentQMS/agent_tools/audit/`
- Tool architecture: `AgentQMS/conventions/audit_framework/tools/tool_architecture.md`

---

**Last Updated**: 2025-11-09  
**Next Review**: After first production audit using automated tooling

