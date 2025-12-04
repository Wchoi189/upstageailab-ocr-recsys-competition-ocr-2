# Experiment Tracker Workflow Lifecycle

## Overview

This document defines the complete workflow lifecycle for experiments, including when to read/write files, when to generate assessments and incident reports, and how to maintain synchronization between different metadata files.

## Experiment States

```
START → ACTIVE → [INCOMPLETE | COMPLETED]
```

### State Transitions

- **START**: Experiment created via `start-experiment.py`
- **ACTIVE**: Main working state, experiment is in progress
- **INCOMPLETE**: Experiment was interrupted/stashed (can be resumed)
- **COMPLETED**: Experiment finished successfully

## File Structure

Each experiment has the following files:

```
experiment_id/
├── state.json                    # Core experiment state
├── .metadata/
│   ├── state.yml                 # Human-readable state metadata
│   ├── tasks.yml                 # Task tracking
│   ├── decisions.yml             # Decision log
│   └── components.yml            # Component tracking
├── artifacts/                    # Generated outputs
├── assessments/                  # Assessment documents
├── incident_reports/            # Incident reports
├── scripts/                      # Experiment scripts
└── logs/                         # Execution logs
```

## When to Read Files

### state.json
- **Read on:** Experiment start, status checks, context retrieval, any operation that needs current state
- **Contains:** Core experiment metadata, artifact list, assessment list, incident report list, status

### .metadata/state.yml
- **Read on:** Displaying experiment context, status checks, phase tracking
- **Contains:** Human-readable state, current phase, success metrics, key achievements, next steps

### .metadata/tasks.yml
- **Read on:** Task operations (add, complete, list), context display
- **Contains:** All tasks with status, timestamps, notes, cross-references

### .metadata/decisions.yml
- **Read on:** Decision operations, context display, assessment generation
- **Contains:** All decisions with rationale, alternatives, impact, cross-references

### .metadata/components.yml
- **Read on:** Component tracking, context display
- **Contains:** Component definitions and relationships

## When to Write Files

### state.json
- **Write on:**
  - Status changes (active → incomplete/completed)
  - Artifact additions (via `record-artifact`)
  - Assessment additions (via `generate-assessment`)
  - Incident report additions (via `generate-incident-report`)
  - Experiment completion
- **Auto-sync:** Automatically synced with .metadata/ files after write operations

### .metadata/state.yml
- **Write on:**
  - Status updates
  - Phase changes (milestones, achievements)
  - Success metrics updates
  - Key achievements logging
- **Auto-sync:** Automatically synced with state.json

### .metadata/tasks.yml
- **Write on:**
  - `add-task` operations
  - `complete-task` operations
  - Task status updates
- **Auto-sync:** Automatically synced with state.json

### .metadata/decisions.yml
- **Write on:**
  - `record-decision` operations
- **Auto-sync:** Automatically synced with state.json

## Cross-Referencing Strategy

### Tasks Reference Decisions
```yaml
tasks:
  - id: task_003
    description: "Implement geometric synthesis"
    related_decisions: [dec_003]  # References decision that led to this task
```

### Decisions Reference Tasks
```yaml
decisions:
  - id: dec_003
    decision: "Use geometric synthesis"
    related_tasks: [task_003, task_004]  # Tasks implementing this decision
```

### Assessments Reference Artifacts
```markdown
## Related Artifacts
- artifacts/test_results_20251129.json
- artifacts/comparison_visualization.png
```

### Incident Reports Reference Tasks/Decisions
```markdown
## Related Tasks
- task_005: Fix coordinate system bug

## Related Decisions
- dec_002: Use position-based classification
```

## Assessment Triggers

Generate assessments (`generate-assessment`) when:

1. **After Test Runs**: Significant test results (success/failure milestones)
2. **Milestones**: Major progress points, phase completions
3. **Significant Findings**: Important discoveries or insights
4. **Experiment Completion**: Final assessment summarizing results
5. **Negative Results**: Important failures that need documentation (use `run-log-negative-result` template)

### Assessment Templates
- `visual-evidence-cluster`: Failure mode clustering board
- `triad-deep-dive`: Input/Output discrepancy analysis
- `ab-regression`: Baseline vs experiment comparison
- `run-log-negative-result`: Interim run log for low-success or negative-result runs

## Incident Report Triggers

Generate incident reports (`generate-incident-report`) when:

1. **Bugs Discovered**: Code bugs, logic errors, implementation issues
2. **Unexpected Failures**: Failures that shouldn't have occurred
3. **Critical Issues**: Issues that block progress or cause significant problems
4. **Validation Failures**: Quality checks that fail unexpectedly

## Incident Report Workflow

### 4-Phase Workflow

1. **Drafting Phase** (`workflow.py incident-draft`)
   - Capture raw observations immediately after failure
   - Logs, screenshots, context
   - Quick capture without structure

2. **Synthesis Phase** (`generate-incident-report`)
   - Transform raw observations into structured Technical Incident Report
   - Use incident_report.md template
   - Include: Visual artifacts, Input characteristics, Geometric analysis, Hypothesis

3. **Assessment Phase** (`workflow.py incident-assess`)
   - Evaluate report against Quality Rubric
   - Check 4 criteria:
     - Root Cause Depth (not just symptoms)
     - Evidence Quality (artifacts linked)
     - Remediation Logic (addresses root cause)
     - Metric Impact (quantifiable prediction)
   - If fail: Request revision with specific guidance
   - If pass: Proceed to committal

4. **Committal Phase** (`workflow.py incident-commit`)
   - Save to `incident_reports/` directory
   - Create task tickets for fixes
   - Link to experiment in state.json
   - Track metrics impact predictions

## Synchronization

### Auto-Sync on Write Operations

When any write operation occurs:
1. Write to primary file (e.g., tasks.yml)
2. Auto-sync to state.json
3. Update cross-references
4. Validate consistency

### Manual Sync

Use `workflow.py sync` to manually synchronize all metadata files if needed.

## Common Workflows

### Starting an Experiment
```bash
./scripts/start-experiment.py --type perspective_correction --intention "Fix edge detection"
```
- Creates experiment directory structure
- Initializes state.json and .metadata/ files
- Sets as current experiment

### Recording Progress
```bash
# Add a task
./scripts/add-task.py "Implement geometric synthesis"

# Record a decision
./scripts/record-decision.py --decision "Use geometric synthesis" --rationale "Eliminates validation paradox"

# Record an artifact
./scripts/record-artifact.py --path artifacts/results.json --type test_results
```

### Handling Failures
```bash
# 1. Draft incident report
./scripts/workflow.py incident-draft --observations "Edge detection failed on 5 images" --context "Testing worst performers"

# 2. Generate structured report
./scripts/generate-incident-report.py --title "Edge Detection Failure" --severity high

# 3. Assess against rubric
./scripts/workflow.py incident-assess --report incident_reports/20251129_1200-edge-detection-failure.md

# 4. If passes, commit
./scripts/workflow.py incident-commit --report incident_reports/20251129_1200-edge-detection-failure.md
```

### Completing an Experiment
```bash
# Generate final assessment
./scripts/generate-assessment.py --template ab-regression

# Mark as completed (updates state.json and .metadata/state.yml)
# (Currently manual - may add script for this)
```

## Best Practices

1. **Always check current experiment** before operations:
   ```bash
   ./scripts/resume-experiment.py --current
   ```

2. **Use workflow automation** for common patterns:
   ```bash
   ./scripts/workflow.py test-complete  # Test → record → task → suggest assessment
   ```

3. **Cross-reference related items** when creating tasks/decisions

4. **Assess incident reports** before committing to ensure quality

5. **Track metrics** in incident reports to validate fixes later

6. **Sync metadata** regularly or use auto-sync feature

## Troubleshooting

### Metadata Out of Sync
```bash
# Manually sync all files
./scripts/workflow.py sync
```

### Wrong Experiment Context
```bash
# Check current
./scripts/resume-experiment.py --current

# Switch to correct experiment
./scripts/resume-experiment.py --id 20251129_173500_perspective_correction_implementation
```

### Missing Artifacts
- Use path resolution with `{timestamp}` placeholder support
- Use `workflow.py record-test-results` for auto-detection
