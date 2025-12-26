---
title: "Interim Assessment: Run {{run_number}} - {{experiment_name}}"
author: "{{author}}"
date: "{{date_kst}}"
status: "draft"
kind: "run_log"
template_id: "run-log-negative-result"
---

# Interim Assessment: Run {{run_number}} - {{experiment_name}}

**Date:** {{date_iso}}
**Run ID:** {{run_id}}
**Hypothesis:** {{hypothesis}}

## 1. Executive Summary

**Result:** {{result_summary}}
**Verdict:** {{verdict}}

## 2. Failure Mode Shift

*Did the errors change, even if the result didn't?*

| Metric | Previous Run | Current Run | Delta |
| :--- | :--- | :--- | :--- |
| **Success Rate** | {{prev_success_rate}} | {{curr_success_rate}} | {{delta_success_rate}} |
| **Primary Error** | {{prev_primary_error}} | {{curr_primary_error}} | {{delta_primary_error}} |
| **Secondary Error** | {{prev_secondary_error}} | {{curr_secondary_error}} | {{delta_secondary_error}} |

**Observation:**
{{failure_migration_observation}}

## 3. Key Samples for Inspection

### A. Status Quo Failure – No Material Change

- **ID:** `{{sample_status_quo_id}}`
- **Behavior:** {{sample_status_quo_behavior}}
- **Implication:** {{sample_status_quo_implication}}

### B. New Failure – Changed or Novel Mode

- **ID:** `{{sample_new_failure_id}}`
- **Behavior:** {{sample_new_failure_behavior}}
- **Implication:** {{sample_new_failure_implication}}

### C. Partial Improvement – Not Yet Passing

- **ID:** `{{sample_partial_id}}`
- **Behavior:** {{sample_partial_behavior}}
- **Implication:** {{sample_partial_implication}}

## 4. Next Steps

1. {{next_step_1}}
2. {{next_step_2}}
3. {{next_step_3}}


