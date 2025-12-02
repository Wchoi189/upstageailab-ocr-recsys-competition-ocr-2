# Audit Framework Templates

**Version**: 1.0  
**Date**: 2025-11-09

## Overview

This directory contains reusable templates for creating audit documents following the audit framework methodology.

---

## Available Templates

### 1. Audit Summary Template
**File**: `00_audit_summary_template.md`

**Purpose**: Executive summary of audit findings

**Usage**: 
1. Copy template to `docs/audit/00_audit_summary.md`
2. Replace placeholders:
   - `{{FRAMEWORK_NAME}}` - Name of framework being audited
   - `{{AUDIT_DATE}}` - Date of audit (YYYY-MM-DD)
   - `{{AUDIT_SCOPE}}` - Scope of audit
   - `{{STATUS}}` - Current status
3. Fill in content following guidance in template

---

### 2. Removal Candidates Template
**File**: `01_removal_candidates_template.md`

**Purpose**: List of items to remove or refactor

**Usage**:
1. Copy template to `docs/audit/01_removal_candidates.md`
2. Replace placeholders
3. Document all issues found during discovery phase
4. Categorize by priority (🔴🟡🟠🟢)

---

### 3. Workflow Analysis Template
**File**: `02_workflow_analysis_template.md`

**Purpose**: Workflow maps and pain point analysis

**Usage**:
1. Copy template to `docs/audit/02_workflow_analysis.md`
2. Replace placeholders
3. Map all major workflows
4. Document pain points and bottlenecks
5. Assess goal vs. implementation alignment

---

### 4. Restructure Proposal Template
**File**: `03_restructure_proposal_template.md`

**Purpose**: Proposed solutions and implementation plan

**Usage**:
1. Copy template to `docs/audit/03_restructure_proposal.md`
2. Replace placeholders
3. Design solutions for all identified issues
4. Organize by implementation phases
5. Define success criteria

---

### 5. Standards Specification Template
**File**: `04_standards_specification_template.md`

**Purpose**: Mandatory standards definition

**Usage**:
1. Copy template to `docs/audit/04_standards_specification.md`
2. Replace placeholders
3. Define all mandatory standards
4. Include validation rules
5. Create compliance checklist

---

### 6. Automation Recommendations Template
**File**: `05_automation_recommendations_template.md`

**Purpose**: Self-maintaining framework design

**Usage**:
1. Copy template to `docs/audit/05_automation_recommendations.md`
2. Replace placeholders
3. Design automation mechanisms
4. Create implementation plan
5. Define success criteria

---

## Placeholder Reference

### Common Placeholders

- `{{FRAMEWORK_NAME}}` - Name of the framework being audited
- `{{AUDIT_DATE}}` - Date of audit (format: YYYY-MM-DD)
- `{{AUDIT_SCOPE}}` - Scope description of the audit
- `{{STATUS}}` - Current status (e.g., "Draft", "In Progress", "Complete")

### Template-Specific Placeholders

Some templates may have additional placeholders. Check the template file for specific guidance.

---

## Usage Workflow

1. **Start Audit**: Create `docs/audit/` directory
2. **Use Templates**: Copy templates in order (00-05)
3. **Follow Protocols**: Use phase protocols from `protocol/` directory
4. **Fill Content**: Replace placeholders and fill in content
5. **Validate**: Ensure all required sections are complete

---

## Template Structure

All templates follow this structure:
- **Frontmatter**: YAML metadata
- **Executive Summary**: High-level overview
- **Main Sections**: Detailed content
- **Success Criteria**: Measurable outcomes
- **Next Steps**: Action items

---

## Best Practices

1. **Follow Order**: Use templates in numerical order (00-05)
2. **Complete Sections**: Don't skip sections, mark as N/A if not applicable
3. **Be Specific**: Provide concrete examples and details
4. **Use Checklists**: Refer to protocol checklists for completeness
5. **Validate**: Review against protocol requirements

---

## Example

See `docs/audit/` for a complete example audit using these templates.

---

**Last Updated**: 2025-11-09

