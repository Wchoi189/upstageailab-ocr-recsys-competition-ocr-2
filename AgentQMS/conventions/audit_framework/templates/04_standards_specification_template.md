---
type: "template"
category: "audit-framework"
artifact_type: "standards_specification"
version: "1.0"
tags: ["audit", "template", "standards"]
title: "Standards Specification Template"
date: "2025-11-09 00:00 (KST)"
---

# Standards Specification

**Date**: {{AUDIT_DATE}}  
**Audit Scope**: Mandatory Standards Definition  
**Status**: {{STATUS}}

## Executive Summary

[Guidance: Provide high-level overview of the standards. Describe what standards are defined and how they are enforced.]

This document defines mandatory standards for the {{FRAMEWORK_NAME}}, including [standard categories]. These standards are designed to be self-enforcing through automated validation.

---

## 1. Output Directory Structure

### 1.1 Standard Directory Layout

```
project_root/
├── [directory1]/                    # [Purpose]
│   ├── [subdirectory]/              # [Purpose]
│   └── [subdirectory]/              # [Purpose]
├── [directory2]/                    # [Purpose]
└── [directory3]/                    # [Purpose]
```

### 1.2 Directory Naming Rules

**Mandatory**:
- [Rule 1]
- [Rule 2]
- [Rule 3]

**Forbidden**:
- ❌ [Forbidden pattern 1] ([Reason])
- ❌ [Forbidden pattern 2] ([Reason])

**Validation**: [How it's validated]

---

## 2. File Naming Conventions

### 2.1 Artifact File Naming

**Format**: `[format_pattern]`

**Components**:
- `[component1]`: [Description]
- `[component2]`: [Description]
- `[component3]`: [Description]

**Examples**:
- ✅ `[valid_example_1]`
- ✅ `[valid_example_2]`
- ❌ `[invalid_example_1]` ([Reason])
- ❌ `[invalid_example_2]` ([Reason])

### 2.2 Artifact Type Prefixes

| Type | Prefix | Directory | Example |
|------|--------|-----------|---------|
| [Type 1] | `[prefix]` | `[directory]/` | `[example]` |
| [Type 2] | `[prefix]` | `[directory]/` | `[example]` |

**Validation**: [How it's validated]

---

## 3. Frontmatter Schemas

### 3.1 Standard Frontmatter Structure

All artifacts must include YAML frontmatter with required fields:

```yaml
---
title: "[Title]"
date: "YYYY-MM-DD"
status: "[status]"
version: "[version]"
category: "[category]"
tags: ["tag1", "tag2"]
---
```

### 3.2 Required Fields by Type

#### [Artifact Type 1]
```yaml
---
title: "[Title Pattern]"
date: "YYYY-MM-DD"
status: "[status_options]"
version: "[version]"
category: "[category]"
tags: ["tag1", "tag2"]
[optional_field]: "[value]"  # Optional
---
```

#### [Artifact Type 2]
[Repeat structure above]

### 3.3 Field Validation Rules

**[Field Name]**:
- Required: [Yes/No]
- Type: [Type]
- [Additional rules]

**[Field Name]**:
- Required: [Yes/No]
- Type: [Type]
- [Additional rules]

**Validation**: [How it's validated]

---

## 4. Artifact Templates

### 4.1 Template Location

All templates in: `[template_directory]/`

**Naming**: `[naming_pattern]`

**Examples**:
- `[example_1]`
- `[example_2]`

### 4.2 Template Structure

Templates must include:
1. [Requirement 1]
2. [Requirement 2]
3. [Requirement 3]

### 4.3 Placeholder Format

**Format**: `{{PLACEHOLDER_NAME}}`

**Examples**:
- `{{PROJECT_NAME}}`
- `{{DATE}}`
- `{{AUTHOR}}`

---

## 5. Code Standards

### 5.1 Code Style

**Language**: [Language]

**Style Guide**: [Style guide reference]

**Key Rules**:
- [Rule 1]
- [Rule 2]
- [Rule 3]

### 5.2 Import Conventions

**Format**: [Format description]

**Examples**:
- ✅ [Valid example]
- ❌ [Invalid example]

---

## 6. Configuration Formats

### 6.1 Configuration File Format

**Format**: [Format] (YAML/JSON/etc.)

**Structure**:
```[format]
[example_structure]
```

### 6.2 Configuration Hierarchy

1. [Level 1]: [Description]
2. [Level 2]: [Description]
3. [Level 3]: [Description]

---

## 7. Validation Rules

### 7.1 Automated Validation

**Tools**: [Tool names]

**Checks**:
- [Check 1]
- [Check 2]
- [Check 3]

### 7.2 Manual Validation

**Process**: [Process description]

**Checklist**:
- [ ] [Check 1]
- [ ] [Check 2]

---

## 8. Extension Standards

### 8.1 Adding New Artifact Types

**Process**:
1. [Step 1]
2. [Step 2]
3. [Step 3]

### 8.2 Custom Validators

**Requirements**:
- [Requirement 1]
- [Requirement 2]

---

## 9. Compliance Checklist

### Directory Structure
- [ ] All directories follow naming rules
- [ ] Directory structure matches standard layout
- [ ] No forbidden patterns

### File Naming
- [ ] All files follow naming conventions
- [ ] Artifact files use correct prefixes
- [ ] No invalid characters or patterns

### Frontmatter
- [ ] All artifacts have frontmatter
- [ ] Required fields present
- [ ] Field values valid

### Templates
- [ ] Templates in correct location
- [ ] Templates follow structure requirements
- [ ] Placeholders properly formatted

### Code Standards
- [ ] Code follows style guide
- [ ] Imports follow conventions
- [ ] No style violations

---

**Last Updated**: {{AUDIT_DATE}}

