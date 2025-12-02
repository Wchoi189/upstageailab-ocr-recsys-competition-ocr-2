---
type: "protocol"
category: "audit-framework"
phase: "discovery"
version: "1.0"
tags: ["audit", "discovery", "methodology"]
title: "Discovery Phase Protocol"
date: "2025-11-09 00:00 (KST)"
---

# Discovery Phase Protocol

**Phase**: Discovery  
**Version**: 1.0  
**Date**: 2025-11-09

## Purpose

The Discovery Phase systematically identifies all issues, broken dependencies, project-specific content, and structural problems that prevent the framework from functioning or being reusable.

---

## Objectives

1. **Identify Critical Issues**: Find all blocking problems
2. **Categorize by Priority**: Classify issues using priority system
3. **Document Removal Candidates**: List items to remove or refactor
4. **Quantify Impact**: Measure scope of each issue

---

## Process

### Step 1: Issue Identification

**Activities**:
- Scan codebase for broken dependencies
- Search for project-specific references
- Identify hardcoded paths and values
- Find missing modules and files
- Check for structural problems

**Tools**:
- Code search (grep, semantic search)
- Dependency analysis
- Path resolution testing
- Import validation

**Output**: Raw list of issues

---

### Step 2: Issue Categorization

**Priority Levels**:

#### 🔴 Critical (Blocking)
**Examples**:
- Missing bootstrap module
- Missing external dependencies
- Path mismatches causing failures
- Security vulnerabilities

**Criteria**:
- Framework non-functional
- Breaking changes
- Data loss risks

#### 🟡 High Priority (Reusability)
**Examples**:
- Project-specific content (125+ references)
- Hardcoded paths
- Missing configuration
- Project names in templates

**Criteria**:
- Prevents framework reuse
- Project-specific content
- Hardcoded values

#### 🟠 Medium Priority (Maintainability)
**Examples**:
- Monolithic validators (560+ lines)
- Duplicate Makefile targets
- Missing dependency declarations
- Structural complexity

**Criteria**:
- Technical debt
- Code complexity
- Documentation gaps

#### 🟢 Low Priority (Optimization)
**Examples**:
- Code style improvements
- Documentation enhancements
- Nice-to-have features
- Performance optimizations

**Criteria**:
- Code style
- Documentation
- Nice-to-have features

**Output**: Categorized issue list

---

### Step 3: Impact Analysis

**For Each Issue**:
- **Files Affected**: Count and list files
- **Impact Description**: What breaks or is affected
- **User Impact**: How users are affected
- **Workaround**: Temporary solutions (if any)

**Output**: Impact assessment for each issue

---

### Step 4: Removal Candidate Documentation

**Document Structure**:

```markdown
## [Priority] [Category]: [Issue Name]

**Location**: [File paths or patterns]

**Issue**: [Description]

**Impact**: 
- [Impact point 1]
- [Impact point 2]

**Action**: 
- [Recommended action]

**Files Affected**:
- [List of files]

**Files Requiring Updates**:
- [List of files to modify]
```

**Output**: Removal Candidate List document

---

## Deliverable: Removal Candidate List

**File**: `01_removal_candidates.md`

**Required Sections**:
1. Executive Summary
2. Critical Issues (Blocking)
3. High Priority Issues (Reusability)
4. Medium Priority Issues (Maintainability)
5. Low Priority Issues (Optimization)
6. Implementation Priority Matrix
7. Cleanup Plan

---

## Discovery Checklist

### Issue Identification
- [ ] Scan for broken dependencies
- [ ] Search for project-specific references
- [ ] Identify hardcoded paths
- [ ] Find missing modules
- [ ] Check structural problems
- [ ] Validate imports
- [ ] Test path resolution

### Categorization
- [ ] Classify all issues by priority
- [ ] Verify priority assignments
- [ ] Group related issues
- [ ] Identify dependencies between issues

### Impact Analysis
- [ ] Count files affected per issue
- [ ] Document user impact
- [ ] Identify workarounds
- [ ] Assess severity

### Documentation
- [ ] Create removal candidate list
- [ ] Document all critical issues
- [ ] Document all high priority issues
- [ ] Document medium/low priority issues
- [ ] Create priority matrix
- [ ] Create cleanup plan

---

## Success Criteria

### Discovery Phase Success
- ✅ All critical issues identified
- ✅ All issues categorized by priority
- ✅ Impact assessed for each issue
- ✅ Removal candidate list complete
- ✅ Priority matrix created

### Quality Checks
- ✅ No critical issues missed
- ✅ Priority assignments accurate
- ✅ Impact descriptions clear
- ✅ Action items specific

---

## Common Patterns

### Pattern 1: Missing Dependencies
**Symptoms**:
- Import errors
- Module not found errors
- Missing file errors

**Discovery**:
- Search for import statements
- Test imports
- Check file existence

**Example**:
```python
# Found in code
from scripts._bootstrap import _load_bootstrap

# Discovery result
❌ scripts/_bootstrap.py does not exist
```

### Pattern 2: Project-Specific Content
**Symptoms**:
- Hardcoded project names
- Project-specific paths
- Project-specific examples

**Discovery**:
- Search for project names
- Search for absolute paths
- Review templates

**Example**:
```python
# Found in code
PROJECT_NAME = "Korean Grammar Correction"

# Discovery result
🟡 High Priority: Project-specific content
```

### Pattern 3: Path Mismatches
**Symptoms**:
- File not found errors
- Path resolution failures
- Inconsistent path patterns

**Discovery**:
- Search for path references
- Test path resolution
- Compare path patterns

**Example**:
```bash
# Found in Makefile
python ../scripts/agent_tools/core/discover.py

# Actual location
agent_tools/core/discover.py

# Discovery result
🔴 Critical: Path mismatch
```

---

## Next Phase

After completing Discovery, proceed to **Analysis Phase** (`02_analysis_protocol.md`) to:
- Map current workflows
- Identify pain points
- Analyze bottlenecks

---

**Last Updated**: 2025-11-09

