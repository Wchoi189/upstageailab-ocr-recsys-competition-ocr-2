# Experiment Tracker Workflow Improvements - Implementation Summary

## Overview

This document summarizes the workflow improvements implemented to address workflow fatigue and reduce errors in experiment tracking.

## Key Improvements

### 1. Context-Aware Operations ✅

All operations now show context before execution:
- Current experiment information (type, status, intention)
- Files that will be updated
- Warnings if experiment is in unexpected state (COMPLETED/INCOMPLETE)

**Usage:**
```bash
# Shows context and asks for confirmation
./scripts/add-task.py "Fix edge detection bug"

# Skip confirmation for automation
./scripts/add-task.py "Fix edge detection bug" --no-confirm

# Skip context display
./scripts/add-task.py "Fix edge detection bug" --no-context
```

### 2. Smart Path Resolution ✅

Artifact recording now supports:
- `{timestamp}` placeholder resolution (finds latest matching artifact)
- Automatic path resolution (relative to experiment, absolute, or artifacts directory)
- Helpful error messages with suggestions

**Usage:**
```bash
# Old way (would fail with literal {timestamp})
./scripts/record-artifact.py --path "artifacts/{timestamp}_worst_performers_test/results.json"

# New way (automatically resolves {timestamp})
./scripts/record-artifact.py --path "artifacts/{timestamp}_worst_performers_test/results.json"

# Or use smart detection
./scripts/workflow.py record-test-results
```

### 3. Metadata Synchronization ✅

Automatic synchronization between `state.json` and `.metadata/` files:
- Auto-syncs after every write operation
- Manual sync available via `workflow.py sync`
- Consistency validation

**Usage:**
```bash
# Manual sync
./scripts/workflow.py sync

# Auto-sync happens automatically on:
# - add-task
# - record-artifact
# - record-decision
# - etc.
```

### 4. Incident Report Workflow ✅

Standardized 4-phase workflow with quality rubric:

1. **Drafting**: Capture raw observations
2. **Synthesis**: Create structured report
3. **Assessment**: Evaluate against quality rubric
4. **Committal**: Save, create tasks, track metrics

**Usage:**
```bash
# 1. Draft
./scripts/workflow.py incident-draft \
  --observations "Edge detection failed on 5 images" \
  --context "Testing worst performers"

# 2. Generate structured report
./scripts/generate-incident-report.py \
  --title "Edge Detection Failure" \
  --severity high

# 3. Assess against rubric
./scripts/workflow.py incident-assess \
  --report incident_reports/20251129_1200-edge-detection-failure.md

# 4. Commit (if assessment passes)
./scripts/workflow.py incident-commit \
  --report incident_reports/20251129_1200-edge-detection-failure.md
```

### 5. Quality Rubric ✅

Incident reports are assessed against 4 criteria:
- **Root Cause Depth**: Identifies *why*, not just *what*
- **Evidence Quality**: Links to specific artifacts
- **Remediation Logic**: Addresses root cause, not symptoms
- **Metric Impact**: Quantifiable prediction

See `.templates/incident_report_rubric.md` for details.

### 6. Workflow Documentation ✅

Complete documentation created:
- `docs/WORKFLOW_LIFECYCLE.md` - Complete workflow guide
- `docs/workflow_diagram.mmd` - Experiment lifecycle diagram
- `docs/incident_report_workflow.mmd` - Incident report workflow diagram

## Files Created/Modified

### New Files
- `src/experiment_tracker/utils/sync.py` - Metadata synchronization
- `scripts/workflow.py` - Workflow automation script
- `.templates/incident_report_rubric.md` - Quality rubric template
- `docs/WORKFLOW_LIFECYCLE.md` - Workflow documentation
- `docs/workflow_diagram.mmd` - Lifecycle diagram
- `docs/incident_report_workflow.mmd` - Incident workflow diagram

### Modified Files
- `src/experiment_tracker/core.py` - Added validation, context display, auto-sync
- `src/experiment_tracker/utils/path_utils.py` - Added path resolution methods
- `scripts/add-task.py` - Added context awareness and confirmation
- `scripts/record-artifact.py` - Added path resolution and confirmation

## Migration Guide

### For Existing Users

1. **Update your commands** to use new features:
   ```bash
   # Old
   ./scripts/add-task.py "My task"

   # New (shows context, asks confirmation)
   ./scripts/add-task.py "My task"

   # Or skip confirmation
   ./scripts/add-task.py "My task" --no-confirm
   ```

2. **Use path resolution** for artifacts:
   ```bash
   # Old (would fail)
   ./scripts/record-artifact.py --path "artifacts/{timestamp}_test/results.json"

   # New (works!)
   ./scripts/record-artifact.py --path "artifacts/{timestamp}_test/results.json"

   # Or use smart detection
   ./scripts/workflow.py record-test-results
   ```

3. **Sync metadata** if needed:
   ```bash
   ./scripts/workflow.py sync
   ```

### For New Workflows

1. **Always check current experiment**:
   ```bash
   ./scripts/resume-experiment.py --current
   ```

2. **Use workflow automation** for common patterns:
   ```bash
   # Smart artifact recording
   ./scripts/workflow.py record-test-results

   # Incident report workflow
   ./scripts/workflow.py incident-draft --observations "..." --context "..."
   ```

3. **Follow incident report workflow** for failures:
   - Draft → Generate → Assess → Commit

## Benefits

1. **Reduced Errors**: Context awareness prevents operations on wrong experiments
2. **Less Fatigue**: Automation reduces manual steps
3. **Better Quality**: Quality rubric ensures actionable incident reports
4. **Clear Workflows**: Documentation and diagrams clarify when to do what
5. **Path Resolution**: No more literal `{timestamp}` failures
6. **Synchronization**: Metadata stays consistent automatically

## Next Steps

1. Review the workflow documentation: `docs/WORKFLOW_LIFECYCLE.md`
2. Try the new workflow commands
3. Use incident report workflow for next failure
4. Provide feedback on what works/doesn't work

## Troubleshooting

### "No active experiment found"
```bash
# Check current experiment
./scripts/resume-experiment.py --current

# Switch to correct experiment
./scripts/resume-experiment.py --id <experiment_id>
```

### "Artifact not found" with {timestamp}
```bash
# Use smart detection instead
./scripts/workflow.py record-test-results
```

### Metadata out of sync
```bash
# Manual sync
./scripts/workflow.py sync
```

## Examples

### Complete Test Workflow
```bash
# 1. Run test
python scripts/test_worst_performers.py

# 2. Record results (auto-detects)
./scripts/workflow.py record-test-results

# 3. Add task
./scripts/add-task.py "Review test results - 100% success rate"

# 4. Generate assessment if needed
./scripts/generate-assessment.py --template ab-regression
```

### Complete Incident Report Workflow
```bash
# 1. Draft
./scripts/workflow.py incident-draft \
  --observations "Edge detection failed" \
  --context "Testing worst performers"

# 2. Generate report
./scripts/generate-incident-report.py \
  --title "Edge Detection Failure" \
  --severity high

# 3. Assess
./scripts/workflow.py incident-assess \
  --report incident_reports/20251129_1200-edge-detection-failure.md

# 4. If passes, commit
./scripts/workflow.py incident-commit \
  --report incident_reports/20251129_1200-edge-detection-failure.md

# 5. Create fix task
./scripts/add-task.py "Fix edge detection bug per incident report"
```
