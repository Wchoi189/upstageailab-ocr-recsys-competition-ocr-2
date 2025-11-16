---
title: "{{ title }}"
author: "{{ author }}"
timestamp: "{{ timestamp }}"
branch: "{{ branch }}"
type: "bug_report"
category: "troubleshooting"
status: "{{ status }}"
version: "1.0"
tags: {{ tags }}
bug_id: "{{ bug_id }}"
severity: "{{ severity }}"
---

# Bug Report: {{ title }}

## Bug ID
{{ bug_id }}

## Summary
{{ summary | default('Brief description of the bug.') }}

## Environment
- **OS**: {{ environment.get('os', 'Not specified') if environment else 'Not specified' }}
- **Python Version**: {{ environment.get('python_version', 'Not specified') if environment else 'Not specified' }}
- **Dependencies**: {{ environment.get('dependencies', 'Not specified') if environment else 'Not specified' }}
- **Browser**: {{ environment.get('browser', 'Not specified') if environment else 'Not specified' }}

## Steps to Reproduce
{{ steps_to_reproduce | default('1. Step 1\n2. Step 2\n3. Step 3') }}

## Expected Behavior
{{ expected_behavior | default('What should happen.') }}

## Actual Behavior
{{ actual_behavior | default('What actually happens.') }}

## Error Messages
```
{{ error_messages | default('Error message here') }}
```

## Screenshots/Logs
{{ screenshots_logs | default('If applicable, include screenshots or relevant log entries.') }}

## Impact
- **Severity**: {{ severity }}
- **Affected Users**: {{ affected_users | default('Who is affected') }}
- **Workaround**: {{ workaround | default('Any temporary workarounds') }}

## Investigation

### Root Cause Analysis
- **Cause**: {{ root_cause | default('What is causing the issue') }}
- **Location**: {{ root_cause_location | default('Where in the code') }}
- **Trigger**: {{ root_cause_trigger | default('What triggers the issue') }}

### Related Issues
{{ related_issues | default('Related issue 1\nRelated issue 2') }}

## Proposed Solution

### Fix Strategy
{{ fix_strategy | default('How to fix the issue.') }}

### Implementation Plan
{{ implementation_plan | default('1. Step 1\n2. Step 2') }}

### Testing Plan
{{ testing_plan | default('How to test the fix.') }}

## Status
- [ ] Confirmed
- [ ] Investigating
- [ ] Fix in progress
- [ ] Fixed
- [ ] Verified

## Assignee
{{ assignee | default('Who is working on this bug.') }}

## Priority
{{ priority | default('High/Medium/Low') }}

---

*This bug report follows the project's standardized format for issue tracking.*

