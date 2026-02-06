---
ads_version: "1.0"
type: guide
experiment_id: "experiment_manager_v1"
title: "experiment_workflow"
created: "2025-12-27 03:45 (KST)"
updated: "2025-12-27 03:45 (KST)"
tags: ["workflow", "cli", "eds-v1.0"]
status: active
---

# Guide: Experiment Workflow

## Prerequisites
- "Python 3.10+"
- "etk v1.0.0+"
- "AgentQMS configured"

## Command Sequences

### 1. Initialize Experiment
```bash
etk init "{experiment_name}" --type "{type}" --intention "{intention}"
```

### 2. Execution Loop
```bash
# Record Artifacts
etk record "{path_to_artifact}" --type other

# Record Task
etk task "{task_description}"

# VLM Analysis (Optional)
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image "{image_path}" \
  --mode defect \
  --output "{output_path}"
```

### 3. Assessment & Reporting
```bash
# Record Decision
etk create assessment "Decision: {decision_title}" --type assessment --phase execution

# Log Insight
etk create assessment "Insight: {insight_title}" --type assessment

# Generate Report
etk create report "{report_title}" --metrics "{metric_list}"
```

### 4. Finalization
```bash
# Sync Metadata
etk sync --all

# Commit
git commit -m "feat(exp): [{experiment_id}]"
```

## Expected Outputs
- `etk init`: "✅ Initialized experiment: {experiment_id}"
- `etk record`: "✅ Recorded artifact: {path}"
- `etk create`: "✅ Created {type}: {filename}"
