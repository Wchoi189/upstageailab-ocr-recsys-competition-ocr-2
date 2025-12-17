---
ads_version: "1.0"
type: "guide"
experiment_id: "statistical_data_collection_framework"
status: "complete"
created: "2025-12-18T00:00:00Z"
updated: "2025-12-18T00:00:00Z"
tags: ["data-collection", "metrics", "framework", "infrastructure"]
commands: []
prerequisites: []
---

# Statistical Data Collection Framework for Experiment Tracker

## Problem Statement

Experiments require structured numerical data collection for:
- Sequential run tracking
- Progress visualization
- Quick reference without reading full reports
- Parallel comparison of multiple runs
- Statistical analysis

## Solution: Run Metrics Table (RMT) Format

**Principle**: Append-only table in dedicated artifact for each experiment phase.

### File Naming Convention

```
YYYYMMDD_HHMM_report_run-metrics-[phase].md
```

Examples:
- `20251217_1800_report_run-metrics-baseline.md`
- `20251217_2000_report_run-metrics-phase1-white-balance.md`
- `20251218_1200_report_run-metrics-phase2-deskewing.md`

### Template Structure

```markdown
---
ads_version: "1.0"
type: "report"
experiment_id: "[experiment_id]"
status: "active"
created: "[timestamp]"
updated: "[timestamp]"
tags: ["metrics", "[phase]"]
metrics: ["[metric1]", "[metric2]", ...]
baseline: "[baseline_run_id]"
comparison: "baseline"
---

# Run Metrics: [Phase Name]

**Phase**: [phase_name]
**Metric Focus**: [Primary metrics being tracked]
**Baseline**: [Reference run for comparison]

## Run History

| Run | Date | Parameters | Metric1 | Metric2 | Metric3 | Status | Notes |
|-----|------|------------|---------|---------|---------|--------|-------|
| 001 | 2025-12-17 | [params] | 0.85 | 45ms | 98% | ✅ | Baseline |
| 002 | 2025-12-17 | [params] | 0.87 | 52ms | 97% | ⚠️ | Slower |
| 003 | 2025-12-18 | [params] | 0.90 | 43ms | 99% | ✅ | Best |

## Best Performance

| Metric | Best Value | Run | Date | Parameters |
|--------|------------|-----|------|------------|
| Metric1 | 0.90 | 003 | 2025-12-18 | [params] |
| Metric2 | 43ms | 003 | 2025-12-18 | [params] |
| Metric3 | 99% | 003 | 2025-12-18 | [params] |

## Trend Analysis

- **Metric1**: Improving (+5.9% from baseline)
- **Metric2**: Stable (±2ms variance)
- **Metric3**: Improving (+1% from baseline)

## Current Recommendation

**Deploy**: Run 003 (best overall performance)
```

## Usage Workflows

### Workflow 1: Initialize New Metrics Artifact

```bash
# Use ETK to create metrics artifact
cd experiment-tracker
etk create report "Run Metrics: Phase 1 White Balance" \
  --experiment 20251217_024343_image_enhancements_implementation \
  --metrics "cer,wer,latency,success_rate" \
  --baseline "run_001" \
  --tags "metrics,phase1,white-balance"
```

This creates skeleton with frontmatter. Then manually add table structure.

### Workflow 2: Append New Run

**Manual Method** (Simple):

```bash
# Open file in editor
vim experiment-tracker/experiments/[exp_id]/[metrics_file].md

# Add new row to Run History table
| 004 | 2025-12-18 | kernel=5 | 0.89 | 44ms | 98% | ✅ | Kernel tuning |
```

**Script Method** (Automated):

```bash
# Append run using helper script
./experiment-tracker/scripts/append-run.py \
  --experiment 20251217_024343_image_enhancements_implementation \
  --metrics-file 20251217_1800_report_run-metrics-phase1.md \
  --run-id 004 \
  --params "kernel=5,threshold=0.8" \
  --metrics "0.89,44,98" \
  --status "✅" \
  --notes "Kernel tuning experiment"
```

### Workflow 3: Cross-Phase Comparison

Create summary artifact comparing best runs from each phase:

```markdown
# Cross-Phase Performance Summary

| Phase | Best Run | Metric1 | Metric2 | Metric3 | Improvement |
|-------|----------|---------|---------|---------|-------------|
| Baseline | - | 0.85 | 45ms | 98% | - |
| Phase 1: White Balance | 003 | 0.90 | 43ms | 99% | +5.9% |
| Phase 2: Deskewing | 007 | 0.92 | 41ms | 99% | +8.2% |
| Phase 3: Shadow Removal | 012 | 0.95 | 40ms | 99.5% | +11.8% |
```

## Metric Categories

### OCR Accuracy Metrics
- `cer`: Character Error Rate (lower is better)
- `wer`: Word Error Rate (lower is better)
- `accuracy`: Prediction accuracy (higher is better)
- `f1`: F1 score (higher is better)

### Performance Metrics
- `latency`: Processing time per image (ms)
- `throughput`: Images per second
- `memory`: Peak memory usage (MB)

### Quality Metrics
- `success_rate`: Percentage of images processed without errors
- `failure_count`: Number of failed images
- `artifact_severity`: Average artifact score (1-10)

### Status Indicators
- ✅ **Success**: Run completed, meets criteria
- ⚠️ **Warning**: Run completed with issues
- ❌ **Failed**: Run failed or catastrophic regression
- 🔄 **Running**: Currently executing
- ⏸️ **Paused**: Execution paused

## Integration with EDS v1.0

### Placement
Run metrics artifacts go in experiment root with EDS naming:
```
experiments/[experiment_id]/YYYYMMDD_HHMM_report_run-metrics-[phase].md
```

### Frontmatter Requirements
```yaml
type: "report"
metrics: ["cer", "wer", "latency"]  # List tracked metrics
baseline: "run_001"                 # Reference run
comparison: "baseline"              # Comparison type
```

### Syncing to Database

```bash
# Sync metrics to database after updates
etk sync --all

# Query metrics across experiments
etk query "run metrics phase1"

# View analytics
etk analytics
```

## Helper Script: append-run.py

Create script to automate row appending:

```python
#!/usr/bin/env python3
"""Append run data to metrics artifact."""

import argparse
import re
from datetime import datetime
from pathlib import Path

def append_run(metrics_file: Path, run_data: dict):
    """Append new run row to metrics table."""

    # Read current file
    with open(metrics_file, 'r') as f:
        content = f.read()

    # Find Run History table
    table_pattern = r'(## Run History.*?\n\|.*?\n\|.*?\n)(.*?)(\n\n##)'
    match = re.search(table_pattern, content, re.DOTALL)

    if not match:
        raise ValueError("Run History table not found")

    # Build new row
    new_row = (
        f"| {run_data['run_id']} "
        f"| {run_data['date']} "
        f"| {run_data['params']} "
        f"| {run_data['metrics']} "
        f"| {run_data['status']} "
        f"| {run_data['notes']} |\n"
    )

    # Insert new row
    new_content = content[:match.end(2)] + new_row + content[match.end(2):]

    # Update frontmatter timestamp
    new_content = re.sub(
        r'updated: ".*?"',
        f'updated: "{datetime.utcnow().isoformat()}Z"',
        new_content
    )

    # Write back
    with open(metrics_file, 'w') as f:
        f.write(new_content)

    print(f"✅ Appended run {run_data['run_id']} to {metrics_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append run to metrics artifact")
    parser.add_argument("--experiment", required=True, help="Experiment ID")
    parser.add_argument("--metrics-file", required=True, help="Metrics filename")
    parser.add_argument("--run-id", required=True, help="Run ID (e.g., 004)")
    parser.add_argument("--params", required=True, help="Parameters (comma-separated)")
    parser.add_argument("--metrics", required=True, help="Metric values (comma-separated)")
    parser.add_argument("--status", required=True, help="Status emoji")
    parser.add_argument("--notes", default="", help="Notes")

    args = parser.parse_args()

    # Build run data
    run_data = {
        'run_id': args.run_id,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'params': args.params,
        'metrics': args.metrics.replace(',', ' | '),
        'status': args.status,
        'notes': args.notes
    }

    # Find metrics file
    metrics_path = Path(f"experiments/{args.experiment}/{args.metrics_file}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    # Append run
    append_run(metrics_path, run_data)
```

## Example: Image Enhancement Experiment

### Phase 1: White Balance Testing

**File**: `20251217_1800_report_run-metrics-phase1-white-balance.md`

```markdown
## Run History

| Run | Date | Method | Kernel | Threshold | CER | Latency | Success% | Status | Notes |
|-----|------|--------|--------|-----------|-----|---------|----------|--------|-------|
| 001 | 12-17 | gray_world | - | - | 0.0850 | 45ms | 98.0% | ✅ | Baseline |
| 002 | 12-17 | white_patch | 15 | 0.95 | 0.0845 | 47ms | 97.5% | ⚠️ | Slower, slight regression |
| 003 | 12-17 | white_patch | 11 | 0.90 | 0.0820 | 44ms | 98.5% | ✅ | Best CER |
| 004 | 12-18 | adaptive | 9 | 0.85 | 0.0815 | 46ms | 99.0% | ✅ | Best accuracy |
| 005 | 12-18 | adaptive | 7 | 0.80 | 0.0810 | 43ms | 99.2% | ✅ | **BEST OVERALL** |
```

**Quick Glance**: Run 005 is best (lowest CER, fast latency, highest success rate)

### Phase 2: Deskewing Testing

**File**: `20251218_1200_report_run-metrics-phase2-deskewing.md`

```markdown
## Run History

| Run | Date | Method | Angle Range | CER | Latency | Success% | Status | Notes |
|-----|------|--------|-------------|-----|---------|----------|--------|-------|
| 006 | 12-18 | projection | ±15° | 0.0795 | 52ms | 99.0% | ⚠️ | Slow |
| 007 | 12-18 | hough | ±10° | 0.0780 | 48ms | 99.5% | ✅ | Good balance |
| 008 | 12-18 | hough | ±20° | 0.0785 | 50ms | 99.2% | ✅ | More robust |
```

### Cross-Phase Summary

**File**: `20251218_1800_report_cross-phase-summary.md`

```markdown
## Best Runs by Phase

| Phase | Run | CER | Latency | Success% | Improvement vs Baseline |
|-------|-----|-----|---------|----------|-------------------------|
| Baseline | 001 | 0.0850 | 45ms | 98.0% | - |
| Phase 1 (WB) | 005 | 0.0810 | 43ms | 99.2% | -4.7% CER |
| Phase 2 (Deskew) | 007 | 0.0780 | 48ms | 99.5% | -8.2% CER |
| **Cumulative** | **P1+P2** | **0.0780** | **91ms** | **99.5%** | **-8.2% CER** |
```

## Advanced: Automated Metrics Collection

### Integration with Test Scripts

```python
# In your test script
import json
from pathlib import Path

def record_run_metrics(experiment_id: str, phase: str, run_id: str, metrics: dict):
    """Record metrics to run metrics artifact."""

    # Build command
    cmd = [
        "./experiment-tracker/scripts/append-run.py",
        f"--experiment={experiment_id}",
        f"--metrics-file=20251217_1800_report_run-metrics-{phase}.md",
        f"--run-id={run_id}",
        f"--params={metrics['params']}",
        f"--metrics={metrics['cer']},{metrics['latency']},{metrics['success_rate']}",
        f"--status={'✅' if metrics['success_rate'] > 98 else '⚠️'}",
        f"--notes={metrics['notes']}"
    ]

    subprocess.run(cmd, check=True)

# Usage in test script
metrics = {
    'params': 'kernel=7,threshold=0.85',
    'cer': 0.0810,
    'latency': 43,
    'success_rate': 99.2,
    'notes': 'Adaptive white balance'
}

record_run_metrics(
    experiment_id="20251217_024343_image_enhancements_implementation",
    phase="phase1-white-balance",
    run_id="005",
    metrics=metrics
)
```

## Benefits Summary

| Feature | Benefit |
|---------|---------|
| **Append-only format** | Historical tracking, no data loss |
| **Table structure** | Ultra-quick visual scanning |
| **Parallel comparison** | See all runs at once |
| **Status indicators** | Immediate visual feedback |
| **Best performance tracking** | Quick reference for optimal parameters |
| **Trend analysis** | See progress over time |
| **Complements reports** | Numerical data + detailed assessments |
| **Database sync** | Query metrics across experiments |

## Recommendations

1. **Create metrics artifact at phase start** - Initialize table structure
2. **Append after each run** - Keep table current
3. **Update best performance** - Track optimal configurations
4. **Add trend analysis** - Summarize progress periodically
5. **Cross-reference detailed reports** - Link to full assessments
6. **Sync to database** - Enable cross-experiment queries
7. **Use status emojis** - Quick visual indicators

## Conclusion

The Run Metrics Table (RMT) format provides structured numerical tracking without replacing detailed assessment documents. Append-only tables enable quick progress visualization and parallel run comparison while maintaining full EDS v1.0 compliance.

**Implementation**: Create helper script `append-run.py` and start using RMT format in ongoing experiments.
