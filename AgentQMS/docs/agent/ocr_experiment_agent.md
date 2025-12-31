---
title: "OCR Experiment Agent – Specialized Instructions"
date: "2025-12-04 00:00 (KST)"
type: "guide"
category: "ai_agent"
status: "active"
version: "1.1"
tags: ["ocr", "experiment", "agent", "vlm", "tracking"]
---

OCR Experiment Agent – Specialized Instructions
===============================================

**Role**: OCR experiment workflows with `experiment-tracker/` + VLM tools.
**Mode**: Manual with standardized metadata and traceability.
**Priority**: Standardization, traceability, feedback.

Prerequisites
-------------
- `experiment-tracker/README.md` – CLI tools
- `AgentQMS/vlm/README.md` – VLM capabilities
- `AgentQMS/knowledge/agent/system.md` – Core rules

Core Workflow
-------------

### 1. Start Experiment

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment-tracker
./scripts/start-experiment.py --type [TYPE] --intention "[CLEAR_INTENTION]"
```

**Types**: `perspective_correction`, `ocr_training`, `synthetic_data`, `preprocessing`, `evaluation`

### 2. Record Artifacts

```bash
# Record visual artifacts with metadata
./scripts/record-artifact.py \
  --path path/to/artifact.jpg \
  --type [TYPE] \
  --metadata '{"technique": "homography_basic", "failure_mode": "corner_overshoot"}'
```

**Artifact Types**: `poor_performance`, `baseline`, `improved`, `edge_case`, `regression`

### 3. VLM Image Analysis

Use VLM tools for automated defect analysis and description generation:

```bash
# Analyze defects in output image
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image path/to/output.jpg \
  --mode defect \
  --output-format markdown \
  --output path/to/analysis.md

# Auto-populate incident report
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image path/to/failure.jpg \
  --mode defect \
  --auto-populate \
  --incident-report path/to/incident_report.md

# Analyze input characteristics
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image path/to/input.jpg \
  --mode input

# Before/after comparison
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
  --image path/to/input.jpg \
  --mode comparison \
  --comparison-image path/to/output.jpg
```

**Modes**: `defect` (output), `input` (characteristics), `comparison` (before/after), `comprehensive` (all)

### 4. Generate Assessment

```bash
# Visual evidence clustering (failure mode analysis)
./scripts/generate-assessment.py \
  --template visual-evidence-cluster \
  --verbose minimal

# Triad deep-dive (input/output discrepancy)
./scripts/generate-assessment.py \
  --template triad-deep-dive \
  --verbose minimal

# Run log for negative results
./scripts/generate-assessment.py \
  --template run-log-negative-result \
  --verbose minimal
```

**Templates**: `visual-evidence-cluster`, `triad-deep-dive`, `ab-regression`, `run-log-negative-result`

### 5. Generate Incident Report

```bash
./scripts/generate-incident-report.py --title "TITLE" --severity SEVERITY --tags "tag1,tag2"
```

**Severity**: `low`, `medium`, `high`, `critical`

### 6. Context Tracking

```bash
# Add task
./scripts/add-task.py \
  --description "Implement corner validation" \
  --status in_progress

# Record decision
./scripts/record-decision.py \
  --decision "Use RANSAC for outlier rejection" \
  --rationale "Handles noisy corner detection better than least-squares"

# Log insight
./scripts/log-insight.py \
  --insight "Low-contrast scenes fail when mask topology touches all borders"
```

### 7. Resume & Export

```bash
# Resume latest experiment
./scripts/resume-experiment.py --type perspective_correction

# Resume specific experiment
./scripts/resume-experiment.py --id 20251129_173500_perspective_correction_implementation

# List all experiments
./scripts/resume-experiment.py --list

# Export experiment
./scripts/export-experiment.py --format archive --destination ./exports
```

Metadata & Results Standards
----------------------------

### Experiment State (`state.json`)

Required fields (see `.schemas/experiment_state.json`):
- `id`: `YYYYMMDD_HHMMSS_type`
- `timestamp`: ISO 8601 format
- `type`: Experiment type
- `intention`: Clear objective statement
- `status`: `active`, `incomplete`, `completed`, `archived`
- `artifacts`: Array of recorded artifacts with metadata
- `assessments`: Array of assessment file paths
- `incident_reports`: Array of incident report paths

### Artifact Metadata

Standardized metadata for traceability:
```json
{
  "technique": "homography_ransac",
  "failure_mode": "corner_overshoot",
  "scene_type": "urban_low_contrast",
  "preprocessing": "rembg_shadow_removal",
  "parameters": {
    "ransac_threshold": 5.0,
    "min_inliers": 4
  }
}
```

### VLM Analysis Output

VLM tools generate markdown with YAML frontmatter:
```yaml
---
analysis_type: "defect"
timestamp: "2024-12-04T10:30:00Z"
backend: "openrouter"
model: "qwen/qwen-2-vl-72b-instruct"
---
```

Traceability Requirements
-------------------------

### Naming Conventions

- **Experiments**: `YYYYMMDD_HHMMSS_type/`
- **Artifacts**: `{type}_{description}_{seq}.jpg`
- **Assessments**: `YYYY-MM-DD_HHMM_assessment-name.md`
- **Incident Reports**: `YYYY-MM-DD_HHMM_BUG_name.md`

### Logging

- Record decisions/insights immediately via context tracking tools
- Document failure modes with VLM before fixes
- Commit after milestones: `git commit -m "feat(exp): [YYYYMMDD_HHMMSS]"`

### Cross-Referencing

```markdown
## Related Resources
* `artifacts/poor_performance_urban_001.jpg`
* `assessments/2024-12-04_1030_visual-evidence-cluster.md`
* `incident_reports/2024-12-04_1045_BUG_perspective_overshoot.md`
```

VLM Tool Integration
-------------------

**Use Cases**: Defect analysis, input characterization, before/after comparison, batch analysis

**Config**: `AgentQMS/vlm/config.yaml`

**Setup**:
```bash
cp AgentQMS/vlm/env.example AgentQMS/vlm/.env
export OPENROUTER_API_KEY="key"
```

### VLM Error Handling

- Check VLM backend status if analysis fails
- Verify image file paths are absolute
- Ensure image resolution ≤ 2048px (see `config.yaml`)
- Check API key environment variables are set

Common Experiment Patterns
--------------------------

### Pattern 1: Failure Mode Investigation

1. Start experiment with clear failure mode hypothesis
2. Record baseline and failure artifacts
3. Run VLM defect analysis on all failures
4. Generate visual-evidence-cluster assessment
5. Create incident report with VLM-populated analysis
6. Document hypothesis and proposed fix
7. Implement fix and record new artifacts
8. Generate triad-deep-dive assessment for comparison

### Pattern 2: Parameter Tuning

1. Start experiment with parameter exploration objective
2. Record baseline performance artifacts
3. For each parameter variation:
   - Record artifacts with metadata including parameters
   - Run VLM comparison analysis
   - Log insights about parameter effects
4. Generate ab-regression assessment
5. Document optimal parameters with rationale

### Pattern 3: Negative Results Documentation

1. Start experiment with clear hypothesis
2. Record all attempted approaches with metadata
3. Use VLM analysis to document WHY approaches failed
4. Generate run-log-negative-result assessment
5. Archive experiment with clear "lessons learned"
6. Ensure future agents can avoid same approaches

Feedback Collection
------------------

After EVERY experiment: `./scripts/generate-feedback.py`

**Review**: Weekly. Update `experiment-tracker/WORKFLOW_IMPROVEMENTS_SUMMARY.md`

Do / Don't
----------

✅ Record intention, use VLM tools, document decisions, cross-reference, collect feedback, export
❌ Skip intention, manual incident reports, skip metadata, ignore negative results, create outside structure

Escalation
----------

Blocked → check `experiment-tracker/README.md`, `AgentQMS/vlm/README.md`, `.schemas/`
3. Review `.schemas/` for validation requirements
4. Check logs in `experiment-tracker/logs/`
5. Document issue in feedback log
6. Consult `AgentQMS/knowledge/agent/system.md` for core rules

Quick Reference
--------------

```bash
# Start
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2/experiment-tracker
./scripts/start-experiment.py --type TYPE --intention "INTENTION"

# Record artifact
./scripts/record-artifact.py --path PATH --type TYPE --metadata '{}'

# VLM analysis
uv run python -m AgentQMS.vlm.cli.analyze_image_defects --image PATH --mode MODE

# Assessment
./scripts/generate-assessment.py --template TEMPLATE --verbose minimal

# Incident report
./scripts/generate-incident-report.py --title "TITLE" --severity SEVERITY

# Context
./scripts/add-task.py --description "DESC" --status STATUS
./scripts/record-decision.py --decision "DEC" --rationale "RAT"
./scripts/log-insight.py --insight "INS"

# Resume
./scripts/resume-experiment.py --list
./scripts/resume-experiment.py --id ID

# Export
./scripts/export-experiment.py --format archive --destination ./exports
```

---

**Version**: 1.1
**Last Updated**: 2024-12-04 12:00 (KST)
**Feedback**: `./scripts/generate-feedback.py` after every experiment
