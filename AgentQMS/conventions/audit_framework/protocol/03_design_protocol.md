---
type: "protocol"
category: "audit-framework"
phase: "design"
version: "1.0"
tags: ["audit", "design", "methodology"]
title: "Design Phase Protocol"
date: "2025-11-09 00:00 (KST)"
---

# Design Phase Protocol

**Phase**: Design  
**Version**: 1.0  
**Date**: 2025-11-09

## Purpose

The Design Phase proposes solutions for identified issues, defines standards for framework usage, and creates design documents that guide implementation.

---

## Objectives

1. **Propose Solutions**: Design fixes for all identified issues
2. **Define Standards**: Create mandatory standards for framework usage
3. **Create Design Documents**: Document design decisions and rationale
4. **Plan Restructure**: Design new structure if needed

---

## Process

### Step 1: Solution Design

**Design Principles**:
1. **Zero External Dependencies**: Framework must work standalone
2. **Self-Enforcing Standards**: Validation prevents violations
3. **Minimal Configuration**: Sensible defaults, easy customization
4. **Clear Error Messages**: Fail fast with actionable guidance
5. **Extensible Design**: Easy to add new artifact types, validators, tools

**Solution Categories**:

#### Critical Fixes
- Fix broken dependencies
- Resolve path issues
- Eliminate missing modules

#### High Priority Solutions
- Remove project-specific content
- Create templates with placeholders
- Add configuration system

#### Medium Priority Solutions
- Refactor monolithic components
- Consolidate duplicate code
- Extract configuration

#### Low Priority Solutions
- Optimize performance
- Improve documentation
- Enhance user experience

**Solution Documentation Format**:

```markdown
### [Priority] [Issue Name]

**Problem**: [Description]

**Solution**: [Proposed solution]

**Implementation**:
```[code example]```

**Actions**:
- [ ] Action item 1
- [ ] Action item 2

**Success Criteria**:
- [ ] Criterion 1
- [ ] Criterion 2
```

**Output**: Restructure proposal document

---

### Step 2: Standards Definition

**Standards Categories**:

#### Directory Structure Standards
- Standard directory layout
- Directory naming rules
- Validation rules

#### File Naming Conventions
- Artifact file naming format
- Artifact type prefixes
- Validation rules

#### Frontmatter Schemas
- Required fields by type
- Field validation rules
- Schema definitions

#### Artifact Templates
- Template location
- Template structure
- Placeholder format

#### Code Standards
- Code style guidelines
- Import conventions
- Documentation requirements

#### Configuration Formats
- Configuration file format
- Configuration hierarchy
- Validation rules

**Standards Documentation Format**:

```markdown
## [Standard Category]

### [Subcategory]

**Rule**: [Description]

**Format**: [Format specification]

**Examples**:
- ✅ Valid example 1
- ✅ Valid example 2
- ❌ Invalid example

**Validation**: [How it's validated]
```

**Output**: Standards specification document

---

### Step 3: Design Decision Documentation

**Design Decisions to Document**:

1. **Structure Decisions**
   - Why this structure?
   - What alternatives were considered?
   - What are the trade-offs?

2. **Naming Decisions**
   - Why these names?
   - What conventions are followed?
   - How are conflicts resolved?

3. **Technology Decisions**
   - Why these technologies?
   - What are the dependencies?
   - How are they maintained?

4. **Process Decisions**
   - Why this process?
   - What are the alternatives?
   - How is it enforced?

**Decision Documentation Format**:

```markdown
### Decision: [Decision Name]

**Context**: [Why this decision is needed]

**Options Considered**:
1. **Option A**: [Description]
   - Pros: [List]
   - Cons: [List]

2. **Option B**: [Description]
   - Pros: [List]
   - Cons: [List]

**Decision**: [Selected option]

**Rationale**: [Why this option was chosen]

**Trade-offs**: [What we're giving up]

**Alternatives Rejected**: [Why other options were rejected]
```

**Output**: Design documents with rationale

---

### Step 4: Restructure Planning

**If Restructure Needed**:

1. **Current Structure Analysis**
   - Document current structure
   - Identify problems
   - List pain points

2. **Target Structure Design**
   - Design new structure
   - Document rationale
   - Define boundaries

3. **Migration Planning**
   - Plan migration steps
   - Identify risks
   - Create rollback strategy

**Restructure Documentation**:

```markdown
## Restructure Overview

### Current Structure
```
[Current structure diagram]
```

### Target Structure
```
[Target structure diagram]
```

### Migration Strategy
1. [Migration step 1]
2. [Migration step 2]

### Benefits
- [Benefit 1]
- [Benefit 2]
```

**Output**: Restructure proposal or design document

---

## Deliverables

### Deliverable 1: Restructure Proposal
**File**: `03_restructure_proposal.md`

**Required Sections**:
1. Executive Summary
2. Restructure Principles
3. Phase 1: Critical Fixes
4. Phase 2: High Priority Solutions
5. Phase 3: Medium Priority Solutions
6. Phase 4: Low Priority Solutions
7. Implementation Priority Matrix
8. Success Criteria
9. Risk Mitigation

### Deliverable 2: Standards Specification
**File**: `04_standards_specification.md`

**Required Sections**:
1. Executive Summary
2. Output Directory Structure
3. File Naming Conventions
4. Frontmatter Schemas
5. Artifact Templates
6. Code Standards
7. Configuration Formats
8. Validation Rules
9. Extension Standards
10. Compliance Checklist

---

## Design Checklist

### Solution Design
- [ ] Design solutions for all critical issues
- [ ] Design solutions for all high priority issues
- [ ] Design solutions for medium/low priority issues
- [ ] Document solution rationale
- [ ] Create implementation actions
- [ ] Define success criteria

### Standards Definition
- [ ] Define directory structure standards
- [ ] Define file naming conventions
- [ ] Define frontmatter schemas
- [ ] Define artifact template standards
- [ ] Define code standards
- [ ] Define configuration formats
- [ ] Create validation rules

### Design Decisions
- [ ] Document structure decisions
- [ ] Document naming decisions
- [ ] Document technology decisions
- [ ] Document process decisions
- [ ] Document alternatives considered
- [ ] Document trade-offs

### Restructure Planning
- [ ] Analyze current structure
- [ ] Design target structure
- [ ] Plan migration strategy
- [ ] Identify risks
- [ ] Create rollback plan

---

## Success Criteria

### Design Phase Success
- ✅ Solutions proposed for all issues
- ✅ Standards defined and documented
- ✅ Design decisions documented
- ✅ Restructure plan created (if needed)
- ✅ Implementation actions defined

### Quality Checks
- ✅ Solutions address root causes
- ✅ Standards are clear and enforceable
- ✅ Design decisions have rationale
- ✅ Implementation actions are specific

---

## Common Patterns

### Pattern 1: Path Resolution Solution
**Problem**: Multiple path patterns, no single source of truth

**Solution**: Centralized path resolution utility

**Example**:
```python
# agent_tools/utils/paths.py
def get_project_root() -> Path:
    """Find project root by looking for marker files."""
    # Implementation
```

### Pattern 2: Template-Based Solution
**Problem**: Project-specific content in templates

**Solution**: Templates with placeholders

**Example**:
```markdown
# Template
**Project**: {{PROJECT_NAME}}
**Date**: {{DATE}}
```

### Pattern 3: Validation-Based Standards
**Problem**: Standards not enforced

**Solution**: Automated validation

**Example**:
```python
# Validation rule
def validate_naming(filename: str) -> bool:
    pattern = r'^\d{4}-\d{2}-\d{2}_\d{4}_\w+_.+\.md$'
    return bool(re.match(pattern, filename))
```

---

## Next Phase

After completing Design, proceed to **Implementation Phase** (`04_implementation_protocol.md`) to:
- Create phased implementation plan
- Define success criteria
- Plan risk mitigation

---

**Last Updated**: 2025-11-09

---

