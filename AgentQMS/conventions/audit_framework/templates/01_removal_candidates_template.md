---
type: "template"
category: "audit-framework"
artifact_type: "removal_candidates"
version: "1.0"
tags: ["audit", "template", "removal_candidates"]
title: "Removal Candidate List Template"
date: "2025-11-09 00:00 (KST)"
---

# Removal Candidate List

**Date**: {{AUDIT_DATE}}  
**Audit Scope**: {{FRAMEWORK_NAME}} Cleanup  
**Status**: {{STATUS}}

## Executive Summary

[Guidance: Provide high-level overview of what this document identifies. Describe the types of items to be removed or refactored and why.]

This document identifies project-specific artifacts, broken dependencies, and unnecessary components that must be removed or refactored to make the framework reusable across projects.

---

## 🔴 CRITICAL: Broken Dependencies (Must Fix)

### 1. [Issue Name]
**Location**: [File paths or patterns where issue occurs]

**Issue**: [Clear description of the problem]

**Impact**: 
- [Impact point 1]
- [Impact point 2]
- [Impact point 3]
- **[Severity statement]**

**Action**: 
- **Option A**: [First solution option]
- **Option B**: [Second solution option]
- **Recommendation**: [Recommended option and why]

**Files Affected**:
- `[file_path]` ([description])
- `[file_path]` ([description])

**Files Requiring Updates**:
- `[file_path]` ([line numbers or description])
- `[file_path]` ([line numbers or description])

---

### 2. [Issue Name]
[Repeat structure above]

---

## 🟡 HIGH PRIORITY: Project-Specific Content

### [Number]. [Issue Name]
**Location**: [File paths or patterns]

**Issue**: [Description of project-specific content issue]

**Impact**: 
- [Impact point 1]
- [Impact point 2]
- [Impact point 3]

**Action**: [Recommended action]

**Files to Clean**:
- `[file_path]` ([description])
- `[file_path]` ([description])

**Strategy**: 
- [Strategy point 1]
- [Strategy point 2]
- [Strategy point 3]

---

## 🟠 MEDIUM PRIORITY: Structural Issues

### [Number]. [Issue Name]
**Location**: [File paths or patterns]

**Issue**: [Description of structural issue]

**Impact**: 
- [Impact point 1]
- [Impact point 2]

**Action**: [Recommended action]

**Files Affected**:
- `[file_path]` ([description])

**Recommendation**: [Specific recommendation]

---

## 🟢 LOW PRIORITY: Cleanup Opportunities

### [Number]. [Issue Name]
**Location**: [File paths or patterns]

**Issue**: [Description of cleanup opportunity]

**Impact**: 
- [Impact point 1]

**Action**: [Recommended action]

**Files Affected**:
- `[file_path]` ([description])

---

## Implementation Priority Matrix

| Priority | Issue Count | Estimated Effort | Phase |
|----------|-------------|------------------|-------|
| 🔴 Critical | [Number] | [Duration] | Phase 1 |
| 🟡 High | [Number] | [Duration] | Phase 2 |
| 🟠 Medium | [Number] | [Duration] | Phase 3 |
| 🟢 Low | [Number] | [Duration] | Phase 4 |

---

## Cleanup Plan

### Phase 1: Critical Fixes
**Goal**: [Goal statement]

**Issues**:
1. [Issue 1]
2. [Issue 2]
3. [Issue 3]

**Timeline**: [Duration]

---

### Phase 2: High Priority Cleanup
**Goal**: [Goal statement]

**Issues**:
1. [Issue 1]
2. [Issue 2]

**Timeline**: [Duration]

---

### Phase 3: Medium Priority Cleanup
**Goal**: [Goal statement]

**Issues**:
1. [Issue 1]
2. [Issue 2]

**Timeline**: [Duration]

---

### Phase 4: Low Priority Cleanup
**Goal**: [Goal statement]

**Issues**:
1. [Issue 1]
2. [Issue 2]

**Timeline**: [Duration]

---

## Summary

**Total Issues Identified**: [Number]
- 🔴 Critical: [Number]
- 🟡 High: [Number]
- 🟠 Medium: [Number]
- 🟢 Low: [Number]

**Estimated Total Effort**: [Duration]

**Recommendation**: [Overall recommendation]

---

**Last Updated**: {{AUDIT_DATE}}

