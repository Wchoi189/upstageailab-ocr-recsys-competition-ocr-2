# Incident Reports Integration Plan

## Overview

Add incident report functionality to the experiment-tracker system to track and analyze defects/failures during experiments. Incident reports will be stored in a dedicated `incident_reports/` directory within each experiment, with full schema validation and CLI tooling.

## Components to Update

### 1. Schema Definition

**File**: `experiment-tracker/.schemas/incident_report.json`

- Create new JSON schema for incident report validation
- Include frontmatter fields: title, date, experiment_id, tags (array), severity (enum), status (enum), author
- Note: `related_artifacts` and `related_assessments` are NOT in frontmatter - they appear at bottom of document
- Follow same pattern as `assessment.json` schema

### 2. Template File

**File**: `experiment-tracker/.templates/incident_report.md`

- Create markdown template with frontmatter section
- Include all sections from user's template:
  - Defect Analysis (Visual Artifacts, Input Characteristics, Geometric/Data Analysis, Hypothesis & Action Items)
- Include "Related Resources" section at bottom with:
  - Related Artifacts (itemized list)
  - Related Assessments (itemized list)
- Use YAML frontmatter format consistent with assessments

### 3. Directory Structure

**File**: `experiment-tracker/src/experiment_tracker/utils/path_utils.py`

- Update `ExperimentPaths.ensure_structure()` to include `incident_reports/` directory
- Add `get_incident_reports_path()` method to ExperimentPaths class

### 4. State Tracking

**File**: `experiment-tracker/.schemas/experiment_state.json`

- Add `incident_reports` array property (similar to `assessments` array)
- Array items should be strings (filenames/paths)

**File**: `experiment-tracker/src/experiment_tracker/core.py`

- Update `start_experiment()` to initialize `incident_reports: []` in state dict
- Add `record_incident_report()` method to track incident reports in state.json (similar to how assessments could be tracked)

### 5. CLI Script

**File**: `experiment-tracker/scripts/generate-incident-report.py`

- Create new script following pattern of `generate-assessment.py`
- Accept arguments: `--title` (required), `--severity` (optional, default: "medium"), `--tags` (optional, comma-separated), `--related-artifacts` (optional), `--related-assessments` (optional)
- Generate filename using format: `YYYYMMDD_HHMM-slug.md` (same as assessments)
- Load template from `.templates/incident_report.md`
- Populate frontmatter with experiment context
- Add related artifacts and assessments as itemized lists at bottom of document (not in frontmatter)
- Write to `experiments/<id>/incident_reports/` directory
- Track in state.json

### 6. Documentation

**File**: `experiment-tracker/README.md`

- Add section on incident reports in Quick Start
- Update Directory Structure to show `incident_reports/` folder
- Add example usage of `generate-incident-report.py`

## Implementation Details

### Frontmatter Structure

```yaml
---
title: "[Short Title, e.g., Perspective Overshoot]"
date: "2025-11-23 14:30 (KST)"
experiment_id: "20251122_1723_perspective_correction"  # Format: YYYYMMDD_HHMM_experiment_type (no seconds)
severity: "high"  # low, medium, high, critical
status: "open"  # open, investigating, resolved, closed
tags: ["perspective", "corner-detection", "geometric-transform"]
author: "AI Agent"
---
```

**Note**: `related_artifacts` and `related_assessments` are NOT included in frontmatter. They appear at the bottom of the document as itemized lists in the "Related Resources" section.

### Template Content

The template includes:

- Defect Analysis sections:
  - Visual Artifacts fields
  - Input Characteristics fields
  - Geometric/Data Analysis fields
  - Hypothesis & Action Items fields
- Related Resources section at bottom:
  - Related Artifacts (itemized list format for long path names)
  - Related Assessments (itemized list format for long path names)

### State Management

- Incident reports are tracked in `state.json` under `incident_reports` array
- Each entry is the relative path: `incident_reports/YYYYMMDD_HHMM-slug.md`
- State is updated when new incident reports are created via CLI script

