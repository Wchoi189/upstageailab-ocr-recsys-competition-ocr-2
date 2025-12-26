# Preprocessing Diagnosis Prompt

Diagnose preprocessing failure root causes and propose fixes.

## Analysis Context

Image shows preprocessing failure:
- Technique applied: [Name + parameters]
- Expected outcome: [Description]
- Actual outcome: [What happened]
- Failure type: [No effect / Partial / Opposite / Artifacts / Coordinate misalignment]

## Analysis Structure

### 1. Failure Characterization
- Expected vs. actual outcome
- Failure type classification
- Severity (1-10)
- Visual evidence: [Specific regions, characters, features]
- Quantitative evidence:
  - Metric 1: Before [X] → After [X] (Expected: [X])
  - Metric 2: Before [X] → After [X] (Expected: [X])

### 2. Root Cause Hypotheses

For each hypothesis:
- **Cause**: [Concise statement]
- **Mechanism**: [How it causes failure]
- **Evidence**: [Visual/quantitative support]
- **Likelihood**: High/Medium/Low

Potential causes:
- Geometric assumptions violated
- Parameter sensitivity
- Algorithm limitations
- Input characteristics
- Coordinate frame errors

### 3. Remediation Strategies

Prioritized fixes:
1. **Primary Fix**: [Most likely solution]
   - Code change: [Specific modification]
   - Parameter adjustment: [Exact values]
   - Alternative: [Different approach]

2. **Secondary Fix**: [Backup solution]
   - [Details]

3. **Conditional Application**: [Skip conditions]
   - If [condition], skip this preprocessing

## Output Format

```markdown
# Preprocessing Diagnosis

**Image**: [Filename]
**Technique**: [Name + parameters]
**Failure Type**: [Classification]
**Severity**: [1-10]

## Summary
- Root Cause: [Primary hypothesis]
- Recommended Fix: [Action]
- Fallback: [Alternative if primary fails]

## Failure Evidence

| Metric | Before | After | Expected | Δ |
|--------|--------|-------|----------|---|
| Metric 1 | X | X | X | ΔX |
| Metric 2 | X | X | X | ΔX |

**Visual**: [Describe specific failure indicators in image]

## Root Cause Analysis

### Hypothesis 1: [Primary] (Likelihood: High/Medium/Low)
**Mechanism**: [Explanation]
**Evidence**: [Support]

### Hypothesis 2: [Secondary] (Likelihood: High/Medium/Low)
**Mechanism**: [Explanation]
**Evidence**: [Support]

## Remediation

### Fix 1: [Primary recommendation]
```python
# Code modification
[Specific code change]
```
**Parameters**: [Exact adjustments]

### Fix 2: [Alternative approach]
[Different technique or algorithm]

### Conditional Skip
**Skip if**: [Conditions when preprocessing should be bypassed]
**Detection**: [How to detect these conditions programmatically]
```
