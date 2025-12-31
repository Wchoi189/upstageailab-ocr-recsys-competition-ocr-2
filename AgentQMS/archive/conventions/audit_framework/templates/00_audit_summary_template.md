---
type: "template"
category: "audit-framework"
artifact_type: "audit_summary"
version: "1.0"
tags: ["audit", "template", "summary"]
title: "Audit Summary Template"
date: "2025-11-09 00:00 (KST)"
---
 
# {{FRAMEWORK_NAME}} Audit Summary

**Date**: {{AUDIT_DATE}}  
**Audit Scope**: {{AUDIT_SCOPE}}  
**Status**: {{STATUS}}
 s
## Executive Summary

[Guidance: Provide high-level overview of audit findings. Describe the current state of the framework, key issues identified, and overall assessment. Keep it concise but comprehensive.]

This audit identified [summary of main issues] preventing the {{FRAMEWORK_NAME}} from [functioning/being reusable/maintaining quality]. The framework is currently **[functional/non-functional/partially functional]** due to [main causes], but [positive assessment if applicable].

---

## Key Findings

### 🔴 Critical Issues (Blocking)
[Guidance: List issues that prevent framework from functioning. These must be fixed immediately.]

1. **[Issue Name]**: [Brief description]
   - [Specific problem 1]
   - [Specific problem 2]
   - [Specific problem 3]

2. **Impact**: [What breaks or is affected]

[Repeat for each critical issue]

---

### 🟡 High Priority Issues (Reusability)
[Guidance: List issues that prevent framework reuse across projects. Fix in Phase 2.]

1. **[Issue Name]**: [Brief description]
   - [Specific problem 1]
   - [Specific problem 2]

2. **Impact**: [What is affected]

[Repeat for each high priority issue]

---

### 🟠 Medium Priority Issues (Maintainability)
[Guidance: List technical debt and complexity issues. Fix in Phase 3.]

1. **[Issue Name]**: [Brief description]
   - [Specific problem 1]
   - [Specific problem 2]

2. **Impact**: [What is affected]

[Repeat for each medium priority issue]

---

### 🟢 Low Priority Issues (Optimization)
[Guidance: List nice-to-have improvements. Fix in Phase 4.]

1. **[Issue Name]**: [Brief description]
   - [Specific problem 1]

2. **Impact**: [What is affected]

[Repeat for each low priority issue]

---

## Deliverables

### 1. Removal Candidate List
**File**: `01_removal_candidates.md`

**Contents**:
- [Number] removal candidates categorized by priority
- Detailed rationale for each
- Implementation priority matrix
- [Number]-phase cleanup plan

**Key Items**:
- [Key item 1] ([Priority] - [count] files)
- [Key item 2] ([Priority] - [count] files)
- [Key item 3] ([Priority] - [count] files)

---

### 2. Workflow Analysis
**File**: `02_workflow_analysis.md`

**Contents**:
- End-to-end workflow maps for all major operations
- Pain point analysis ([number] critical/high/medium issues)
- Goal vs implementation alignment assessment
- Structural bottleneck identification
- Efficiency metrics and recommendations

**Key Findings**:
- [Finding 1]
- [Finding 2]
- [Finding 3]

---

### 3. Restructure Proposal
**File**: `03_restructure_proposal.md`

**Contents**:
- [Number]-phase implementation plan
- Detailed solutions for each issue
- Implementation priority matrix
- Success criteria and risk mitigation
- Timeline: [duration] to [goal]

**Phases**:
1. **Phase 1 (Critical)**: [Goal] - [duration]
2. **Phase 2 (High)**: [Goal] - [duration]
3. **Phase 3 (Medium)**: [Goal] - [duration]
4. **Phase 4 (Low)**: [Goal] - [duration]

---

### 4. Standards Specification
**File**: `04_standards_specification.md`

**Contents**:
- Mandatory standards for:
  - Directory structure
  - File naming conventions
  - Frontmatter schemas
  - Artifact templates
  - Code standards
  - Configuration formats
- Validation rules
- Extension standards
- Compliance checklist

**Key Standards**:
- [Standard 1]: [Description]
- [Standard 2]: [Description]
- [Standard 3]: [Description]
- Validation: [How it's enforced]

---

### 5. Automation Recommendations
**File**: `05_automation_recommendations.md`

**Contents**:
- Self-enforcing compliance mechanisms
- Automated validation strategies
- Proactive maintenance automation
- Self-documenting systems
- Monitoring and alerts
- [Number]-phase implementation plan

**Key Recommendations**:
- [Recommendation 1]
- [Recommendation 2]
- [Recommendation 3]
- [Recommendation 4]
- [Recommendation 5]

---

## Framework Assessment

### Systematic Organization
**Status**: [✅ Aligned / ⚠️ Partially Aligned / ❌ Misaligned]

[Assessment details]
- [Positive point 1]
- [Positive point 2]
- **Gap**: [What's missing or broken]

---

### Scalability
**Status**: [✅ Aligned / ⚠️ Partially Aligned / ❌ Misaligned]

[Assessment details]
- [Positive point 1]
- [Positive point 2]
- **Gap**: [What's missing or broken]

---

### Automation
**Status**: [✅ Aligned / ⚠️ Partially Aligned / ❌ Misaligned]

[Assessment details]
- [Positive point 1]
- [Positive point 2]
- **Gap**: [What's missing or broken]

---

### Robustness
**Status**: [✅ Aligned / ⚠️ Partially Aligned / ❌ Misaligned]

[Assessment details]
- [Positive point 1]
- [Positive point 2]
- **Gap**: [What's missing or broken]

---

## Implementation Roadmap

### Week 1: Critical Fixes
**Goal**: [Goal statement]

**Tasks**:
1. [Task 1]
2. [Task 2]
3. [Task 3]
4. [Task 4]

**Success Criteria**:
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

---

### Week 2: [Phase Name]
**Goal**: [Goal statement]

**Tasks**:
1. [Task 1]
2. [Task 2]
3. [Task 3]

**Success Criteria**:
- [ ] [Criterion 1]
- [ ] [Criterion 2]

---

### Week 3: [Phase Name]
**Goal**: [Goal statement]

**Tasks**:
1. [Task 1]
2. [Task 2]

**Success Criteria**:
- [ ] [Criterion 1]
- [ ] [Criterion 2]

---

### Week 4: [Phase Name]
**Goal**: [Goal statement]

**Tasks**:
1. [Task 1]
2. [Task 2]

**Success Criteria**:
- [ ] [Criterion 1]
- [ ] [Criterion 2]

---

## Risk Assessment

### High Risk
- **[Risk Name]**: [Description]
  - **Mitigation**: [How it's mitigated]

### Medium Risk
- **[Risk Name]**: [Description]
  - **Mitigation**: [How it's mitigated]

### Low Risk
- **[Risk Name]**: [Description]
  - **Mitigation**: [How it's mitigated]

---

## Success Metrics

### Current State
- **Functionality**: [X]% ([Description])
- **Reusability**: [X]% ([Description])
- **Maintainability**: [X]% ([Description])
- **Automation**: [X]% ([Description])

### Target State (After [duration])
- **Functionality**: [X]% ([Description])
- **Reusability**: [X]% ([Description])
- **Maintainability**: [X]% ([Description])
- **Automation**: [X]% ([Description])

---

## Next Steps

### Immediate Actions
1. **Review audit deliverables** with stakeholders
2. **Approve restructure proposal** and timeline
3. **Assign resources** for Phase 1 implementation
4. **Set up tracking** for implementation progress

### Phase 1 Kickoff
1. [Action 1]
2. [Action 2]
3. [Action 3]
4. [Action 4]
5. [Action 5]

---

## Conclusion

[Guidance: Summarize the audit findings, provide overall assessment, and make a recommendation. Keep it concise but comprehensive.]

The {{FRAMEWORK_NAME}} [assessment]. With focused effort over [duration], the framework can be [goal].

The audit provides:
- ✅ Clear identification of all issues
- ✅ Prioritized implementation plan
- ✅ Detailed standards specification
- ✅ Automation strategy for self-maintenance
- ✅ Success criteria and metrics

**Recommendation**: [Recommendation statement]

---

## Document Index

1. **00_audit_summary.md** (this document) - Executive summary
2. **01_removal_candidates.md** - Items to remove/refactor
3. **02_workflow_analysis.md** - Workflow maps and pain points
4. **03_restructure_proposal.md** - Implementation plan
5. **04_standards_specification.md** - Mandatory standards
6. **05_automation_recommendations.md** - Self-maintenance strategy

---

**Audit Completed**: {{AUDIT_DATE}}  
**Next Review**: [Date or condition]

