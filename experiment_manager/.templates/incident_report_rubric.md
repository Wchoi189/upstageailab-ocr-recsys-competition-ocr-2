# Incident Report Quality Rubric

This rubric defines the standards for a complete, actionable incident report. A report must pass all four criteria to be considered complete.

## Assessment Criteria

### 1. Root Cause Depth ✅

**Passing Standard:** Identifies *why* the logic failed at a fundamental level.

**Examples:**
- ✅ **Pass**: "Coordinate system mismatch - algorithm assumes Y increases upward but image coordinates use Y increasing downward"
- ✅ **Pass**: "Short segment instability - small angular errors on short segments are magnified over distance (Lever Arm Effect)"
- ❌ **Fail**: "The box was in the wrong place" (symptom, not root cause)
- ❌ **Fail**: "Bad thresholds" (symptom, not root cause)

**Question to Ask:** Does this explain *why* the failure occurred, or just *what* failed?

---

### 2. Evidence Quality ✅

**Passing Standard:** Links to specific artifacts that demonstrate the issue.

**Examples:**
- ✅ **Pass**: "See `debug_viz_01.jpg` showing collinear corner points"
- ✅ **Pass**: "Logs show RMSE > 50px in `logs/edge_detection_20251129.log`"
- ✅ **Pass**: "Comparison visualization: `artifacts/before_after_comparison.png`"
- ❌ **Fail**: "It looked weird" (vague description)
- ❌ **Fail**: "The output was wrong" (no specific evidence)

**Question to Ask:** Can someone verify this issue exists by looking at the referenced artifacts?

---

### 3. Remediation Logic ✅

**Passing Standard:** Proposes a fix that addresses the *Root Cause*, not just the symptom.

**Examples:**
- ✅ **Pass**: "Fix coordinate system by inverting Y-axis in angle calculations" (addresses root cause)
- ✅ **Pass**: "Use weighted regression by segment length to reduce short-segment instability" (addresses root cause)
- ❌ **Fail**: "Just tweak the threshold" (patches symptom)
- ❌ **Fail**: "Add a check to skip bad cases" (avoids problem, doesn't fix it)

**Question to Ask:** If I implement this fix, will it prevent the root cause from occurring again?

---

### 4. Metric Impact ✅

**Passing Standard:** Predicts quantifiable improvement with specific metrics.

**Examples:**
- ✅ **Pass**: "Should raise success rate from 0% to >80% on worst performers"
- ✅ **Pass**: "Expected to reduce RMSE from 50px to <10px for 90% of cases"
- ✅ **Pass**: "Will eliminate all 'invalid_edge_angles' failures (currently 15/25)"
- ❌ **Fail**: "Should make it better" (no quantifiable prediction)
- ❌ **Fail**: "Will improve performance" (vague, no metrics)

**Question to Ask:** Can I measure whether this fix actually worked using the predicted metrics?

---

## Assessment Workflow

1. **Review the incident report** against each criterion
2. **Mark each criterion** as Pass ✅ or Fail ❌
3. **If any criterion fails:**
   - List exactly what details are missing
   - Request revision with specific guidance
4. **If all criteria pass:**
   - Approve for committal
   - Proceed to create task tickets
   - Track metrics impact

## Example Assessment

**Report Title:** "Perspective Correction Fails on Low Contrast Images"

| Criterion | Status | Notes |
|-----------|--------|-------|
| Root Cause Depth | ✅ Pass | Identifies contrast threshold issue causing edge detection failure |
| Evidence Quality | ✅ Pass | Links to `artifacts/low_contrast_failures/` with 5 example images |
| Remediation Logic | ✅ Pass | Proposes adaptive threshold based on local contrast statistics |
| Metric Impact | ✅ Pass | Predicts improvement from 40% to 85% success rate on low-contrast subset |

**Result:** ✅ **APPROVED** - All criteria pass. Ready for committal.

---

## Integration with Experiment Tracker

- Incident reports are linked to experiments via `experiment_id` in frontmatter
- Failed assessments trigger revision workflow
- Passed assessments automatically create task tickets
- Metric predictions are tracked in experiment state for validation
