---
ads_version: "1.0"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation', 'plan', 'standardization', 'eds-v1.0', 'experiment-tracker']
title: "Experiment Documentation Standard (EDS v1.0) Implementation"
date: "2025-12-17 17:05 (KST)"
branch: "refactor/inference-module-consolidation"
related_artifacts:
  - "2025-12-17_1703_assessment-experiment-tracker-standardization.md"
  - "AI Documentation Standardization (ADS v1.0)"
priority: "critical"
estimated_effort: "26-38 hours (4 phases)"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Experiment Documentation Standard (EDS v1.0)**. Your primary responsibility is to execute this blueprint systematically, applying lessons from the successful AgentQMS ADS v1.0 overhaul to the experiment-tracker framework. Do not ask for clarification; execute tasks sequentially as defined below.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** Eliminate chaos in experiment artifacts through AI-optimized standardization
2. **Execute:** Complete tasks in order (Phase 1 → Phase 2 → Phase 3 → Phase 4)
3. **Handle Outcome & Update:** Report results, update this blueprint, proceed to next task

**Authorization**: Complete overhaul authorized. Aggressive pruning/modification permitted. Zero legacy support required.

---

# Living Implementation Blueprint: EDS v1.0 Implementation

## Progress Tracker
- **STATUS:** Not Started
- **CURRENT PHASE:** Phase 1 (Foundation)
- **CURRENT STEP:** Task 1.1 - Create EDS v1.0 Schema Specification
- **LAST COMPLETED TASK:** Assessment complete (2025-12-17_1703)
- **NEXT TASK:** Create `experiment-tracker/.ai-instructions/schema/eds-v1.0-spec.yaml`

**Key Metrics to Track**:
- Compliance Rate: 0% → 100% (target)
- Token Footprint: ~8,500 → ~850 tokens (90% reduction target)
- Naming Violations: 6/7 → 0/7 (target)
- Framework Stability: Regressing → Stable

---

## Implementation Outline (Checklist)

### **PHASE 1: FOUNDATION (Week 1-2) - CRITICAL**
**Goal**: Establish EDS v1.0 specification and enforcement infrastructure
**Effort**: 7-11 hours
**Blocking**: All other work depends on this

#### **Task 1.1: Create EDS v1.0 Schema Specification** ⏳ NEXT
**Objective**: Define machine-readable YAML format for experiment artifacts
**Effort**: 2-4 hours

**Deliverables**:
- [ ] Create `experiment-tracker/.ai-instructions/` directory structure
- [ ] Create `experiment-tracker/.ai-instructions/schema/eds-v1.0-spec.yaml`
- [ ] Create `experiment-tracker/.ai-instructions/schema/validation-rules.json` (JSON Schema)
- [ ] Create `experiment-tracker/.ai-instructions/schema/compliance-checker.py` (Python validator)

**EDS v1.0 Specification Requirements**:
```yaml
# Required frontmatter fields
required_fields:
  - ads_version: "1.0"  # Reuse ADS versioning for consistency
  - type: enum[assessment, report, guide, script, manifest]
  - experiment_id: YYYYMMDD_HHMMSS_name
  - status: enum[draft, active, complete, deprecated]
  - created: ISO8601 datetime
  - updated: ISO8601 datetime
  - tags: array[string]
  - phase: enum[phase_0, phase_1, phase_2, phase_3, phase_4]
  - priority: enum[critical, high, medium, low]

# Prohibited content (similar to ADS v1.0)
prohibited:
  - user_oriented_tutorials: true
  - emoji_usage: true
  - verbose_prose: true
  - conceptual_explanations: true

# Document types
document_types:
  assessment: "Structured failure analysis with evidence clusters"
  report: "Quantitative metrics and comparison tables"
  guide: "Step-by-step execution commands (no explanations)"
  script: "Executable Python/Bash with structured comments"
  manifest: "Experiment metadata (state.json format)"
```

**Success Criteria**:
- [ ] Schema passes JSON Schema validation
- [ ] compliance-checker.py validates YAML files correctly
- [ ] Spec aligns with ADS v1.0 principles (machine-readable, AI-only, structured)

**Estimated Time**: 2-4 hours

---

#### **Task 1.2: Extract Critical Rules to Tier 1**
**Objective**: Create ultra-concise YAML files with experiment management rules
**Effort**: 3-4 hours
**Dependencies**: Task 1.1 complete

**Deliverables**:
- [ ] Create `experiment-tracker/.ai-instructions/tier1-sst/` directory
- [ ] Create `artifact-naming-rules.yaml` (~40 lines)
- [ ] Create `artifact-placement-rules.yaml` (~60 lines)
- [ ] Create `artifact-workflow-rules.yaml` (~90 lines)
- [ ] Create `experiment-lifecycle-rules.yaml` (~70 lines)
- [ ] Create `validation-protocols.yaml` (~70 lines)

**Artifact Naming Rules (artifact-naming-rules.yaml)**:
```yaml
pattern: "YYYYMMDD_HHMM_{TYPE}_{slug}.md"
examples:
  - "20251217_1703_assessment_failure-cluster-analysis.md"
  - "20251217_1705_report_ocr-accuracy-comparison.md"
  - "20251217_1710_guide_enhancement-testing-protocol.md"

prohibited:
  all_caps_filenames: "MASTER_ROADMAP.md"  # critical violation
  camel_case: "masterRoadmap.md"
  pascal_case: "MasterRoadmap.md"
  spaces: "master roadmap.md"

enforcement:
  pre_commit_hook: "naming-validation.sh"
  error_message: "Filename must match YYYYMMDD_HHMM_{TYPE}_{slug}.md pattern"
```

**Artifact Placement Rules (artifact-placement-rules.yaml)**:
```yaml
structure:
  experiment_root:
    - .metadata/           # REQUIRED - experiment metadata
    - assessments/         # Failure analysis, evidence clusters
    - reports/             # Quantitative metrics, comparisons
    - guides/              # Execution protocols, command references
    - scripts/             # Python/Bash scripts
    - artifacts/           # Generated outputs (images, data)
    - state.json           # REQUIRED - experiment state

root_exceptions:
  - README.md              # Brief experiment overview only
  - state.json             # State tracking

prohibited_at_root:
  - "MASTER_ROADMAP.md"    # Move to guides/
  - "EXECUTIVE_SUMMARY.md" # Move to reports/
  - "PRIORITY_PLAN.md"     # Move to guides/

enforcement:
  pre_commit_hook: "placement-validation.sh"
```

**Artifact Workflow Rules (artifact-workflow-rules.yaml)**:
```yaml
mandatory_commands:
  start_experiment: "./scripts/start-experiment.py --type <type> --intention \"<goal>\""
  resume_experiment: "./scripts/resume-experiment.py --id <id>"
  record_artifact: "./scripts/record-artifact.py --path <path> --type <type>"
  generate_assessment: "./scripts/generate-assessment.py --template <template>"

frontmatter_requirements:
  all_artifacts: ["ads_version", "type", "experiment_id", "status", "created", "updated"]
  assessments: ["phase", "priority", "evidence_count"]
  reports: ["metrics", "baseline", "comparison"]
  guides: ["commands", "prerequisites"]

prohibited_actions:
  manual_artifact_creation: true  # CRITICAL - must use CLI tools
  direct_file_creation: true
  skip_metadata: true
```

**Experiment Lifecycle Rules (experiment-lifecycle-rules.yaml)**:
```yaml
lifecycle_stages:
  active:
    status: "active"
    retention: "indefinite"
    updates: "permitted"

  complete:
    status: "complete"
    retention: "indefinite"
    updates: "corrections_only"

  deprecated:
    status: "deprecated"
    retention: "30_days"
    migration_required: true

deprecation_policy:
  tier1_retention: 30  # days
  tier2_retention: 60
  tier3_retention: 90

  warnings:
    - at_day_15: "Experiment scheduled for archival"
    - at_day_25: "Final warning - 5 days until deletion"

supersession_tracking:
  required_fields:
    - superseded_by: "experiment_id"
    - supersedes: "experiment_id"
    - migration_notes: "summary"
```

**Validation Protocols (validation-protocols.yaml)**:
```yaml
validation_commands:
  validate_experiment:
    command: "python .ai-instructions/schema/compliance-checker.py <experiment_dir>"
    frequency: "pre_commit"
    scope: "all_yaml_files"

  validate_naming:
    command: ".ai-instructions/tier4-workflows/pre-commit-hooks/naming-validation.sh"
    frequency: "pre_commit"
    scope: "new_files"

  compliance_dashboard:
    command: "python .ai-instructions/tier4-workflows/compliance-reporting/generate-compliance-report.py"
    frequency: "on_demand"
    scope: "all_experiments"

failure_actions:
  commit_blocked: true
  error_message_required: true
  fix_suggestions_required: true
```

**Success Criteria**:
- [ ] Total lines <330 (target similar to AgentQMS tier1-sst: 348 lines)
- [ ] All files pass EDS v1.0 validation
- [ ] 100% machine-readable YAML format
- [ ] Zero user-oriented prose

**Estimated Time**: 3-4 hours

---

#### **Task 1.3: Implement Pre-Commit Hooks**
**Objective**: Create validation hooks that block violations at commit time
**Effort**: 2-3 hours
**Dependencies**: Task 1.2 complete

**Deliverables**:
- [ ] Create `experiment-tracker/.ai-instructions/tier4-workflows/` directory
- [ ] Create `pre-commit-hooks/naming-validation.sh`
- [ ] Create `pre-commit-hooks/metadata-validation.sh`
- [ ] Create `pre-commit-hooks/eds-compliance.sh`
- [ ] Create `pre-commit-hooks/install-hooks.sh`
- [ ] Install hooks in `experiment-tracker/.git/hooks/`

**Naming Validation Hook (naming-validation.sh)**:
```bash
#!/bin/bash
# Blocks ALL-CAPS filenames in experiment artifacts

PATTERN="^[0-9]{8}_[0-9]{4}_(assessment|report|guide|script)_[a-z0-9-]+\.md$"

for file in $(git diff --cached --name-only --diff-filter=ACM experiment-tracker/experiments/); do
  if [[ "$file" == *".md" ]]; then
    filename=$(basename "$file")

    # Check for ALL-CAPS violations
    if [[ "$filename" =~ ^[A-Z_]+\.md$ ]]; then
      echo "❌ NAMING VIOLATION: $filename"
      echo "   ALL-CAPS filenames prohibited"
      echo "   Expected: YYYYMMDD_HHMM_{TYPE}_{slug}.md"
      exit 1
    fi

    # Check pattern compliance
    if [[ ! "$filename" =~ $PATTERN ]] && [[ "$filename" != "README.md" ]] && [[ "$filename" != "state.json" ]]; then
      echo "⚠️  NAMING WARNING: $filename"
      echo "   Expected: YYYYMMDD_HHMM_{TYPE}_{slug}.md"
    fi
  fi
done
```

**Metadata Validation Hook (metadata-validation.sh)**:
```bash
#!/bin/bash
# Requires .metadata/ directory in all experiments

for dir in $(git diff --cached --name-only --diff-filter=ACM experiment-tracker/experiments/ | cut -d'/' -f1-3 | sort -u); do
  if [[ -d "$dir" ]]; then
    if [[ ! -d "$dir/.metadata" ]]; then
      echo "❌ METADATA VIOLATION: $dir"
      echo "   Missing required .metadata/ directory"
      echo "   Run: mkdir $dir/.metadata && touch $dir/.metadata/manifest.json"
      exit 1
    fi
  fi
done
```

**EDS Compliance Hook (eds-compliance.sh)**:
```bash
#!/bin/bash
# Validates YAML frontmatter against EDS v1.0

CHECKER="experiment-tracker/.ai-instructions/schema/compliance-checker.py"

for file in $(git diff --cached --name-only --diff-filter=ACM experiment-tracker/experiments/**/*.md); do
  if [[ "$file" == *".md" ]] && [[ "$file" != *"README.md" ]]; then
    python3 "$CHECKER" "$file" 2>&1
    if [[ $? -ne 0 ]]; then
      echo "❌ EDS COMPLIANCE VIOLATION: $file"
      exit 1
    fi
  fi
done
```

**Hook Installer (install-hooks.sh)**:
```bash
#!/bin/bash
# Master hook installer

HOOKS_DIR="experiment-tracker/.ai-instructions/tier4-workflows/pre-commit-hooks"
GIT_HOOKS_DIR="experiment-tracker/.git/hooks"

# Create pre-commit orchestrator
cat > "$GIT_HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
HOOKS_DIR="experiment-tracker/.ai-instructions/tier4-workflows/pre-commit-hooks"

echo "🔍 Running EDS v1.0 validation..."

bash "$HOOKS_DIR/naming-validation.sh" || exit 1
bash "$HOOKS_DIR/metadata-validation.sh" || exit 1
bash "$HOOKS_DIR/eds-compliance.sh" || exit 1

echo "✅ All validations passed"
EOF

chmod +x "$GIT_HOOKS_DIR/pre-commit"
echo "✅ Pre-commit hooks installed successfully"
```

**Success Criteria**:
- [ ] Hooks block ALL-CAPS filenames
- [ ] Hooks block missing .metadata/ directories
- [ ] Hooks block non-compliant YAML frontmatter
- [ ] Error messages are clear and actionable
- [ ] Hooks execute in <1 second

**Estimated Time**: 2-3 hours

---

### **PHASE 2: AUTOMATION (Week 2-3) - HIGH PRIORITY**
**Goal**: Convert templates and enable monitoring
**Effort**: 8-11 hours
**Dependencies**: Phase 1 complete

#### **Task 2.1: Convert Templates to AI-Optimized Format**
**Objective**: Rewrite templates as structured YAML with execution rules
**Effort**: 3-4 hours

**Deliverables**:
- [ ] Create `experiment-tracker/.ai-instructions/tier2-framework/` directory
- [ ] Create `artifact-catalog.yaml` (~250 lines)
- [ ] Deprecate `.templates/assessment_templates.json`
- [ ] Update CLI tools to reference new catalog

**Artifact Catalog Structure (artifact-catalog.yaml)**:
```yaml
ads_version: "1.0"
type: framework
tier: tier2
priority: high

artifact_types:
  assessment:
    type: "assessment"
    purpose: "Structured failure analysis with evidence clusters"
    required_frontmatter:
      - ads_version
      - type
      - experiment_id
      - status
      - phase
      - priority
      - evidence_count
    required_sections:
      - failure_clusters
      - evidence_samples
      - root_cause_analysis
      - next_actions
    prohibited_content:
      - user_tutorials
      - conceptual_explanations
      - emoji

  report:
    type: "report"
    purpose: "Quantitative metrics and comparison tables"
    required_frontmatter:
      - ads_version
      - type
      - experiment_id
      - metrics
      - baseline
      - comparison
    required_sections:
      - metrics_table
      - baseline_comparison
      - statistical_analysis
    format: "tables_and_numbers_only"

  guide:
    type: "guide"
    purpose: "Step-by-step execution commands (no explanations)"
    required_frontmatter:
      - ads_version
      - type
      - experiment_id
      - commands
      - prerequisites
    required_sections:
      - command_sequence
      - expected_outputs
      - validation_steps
    format: "commands_only"

workflow_commands:
  start_experiment:
    command: "./scripts/start-experiment.py"
    args:
      --type: enum[perspective_correction, data_augmentation, training_run, custom]
      --intention: string
    example: "./scripts/start-experiment.py --type perspective_correction --intention \"Analyze failure cases\""

  generate_assessment:
    command: "./scripts/generate-assessment.py"
    args:
      --template: enum[visual-evidence-cluster, triad-deep-dive, ab-regression, run-log-negative-result]
      --verbose: enum[minimal, normal, detailed]
    output: "assessments/YYYYMMDD_HHMM_assessment_{template}.md"

validation_rules:
  when_to_create_assessment:
    - "Failure rate >20%"
    - "Evidence samples collected (≥5)"
    - "Root cause hypothesis formed"

  when_to_create_report:
    - "Metrics collected"
    - "Baseline comparison available"
    - "Statistical significance tested"

  when_to_create_guide:
    - "Reusable procedure identified"
    - "Commands tested and validated"
    - "Prerequisites documented"

workflow_triggers:
  high_failure_rate:
    condition: "failure_rate > 0.2"
    suggested_action: "generate_assessment --template visual-evidence-cluster"

  baseline_comparison_ready:
    condition: "baseline_data && experiment_data"
    suggested_action: "generate_assessment --template ab-regression"

  procedure_stabilized:
    condition: "success_rate > 0.8 && iterations >= 3"
    suggested_action: "generate_guide --type reusable-procedure"
```

**Success Criteria**:
- [ ] Catalog <250 lines (similar to AgentQMS tier2: 243 lines)
- [ ] All artifact types defined with required fields
- [ ] Workflow commands structured with exact syntax
- [ ] Validation rules specify when to use each artifact type
- [ ] Zero prose descriptions, 100% structured data

**Estimated Time**: 3-4 hours

---

#### **Task 2.2: Create Compliance Dashboard**
**Objective**: Build automated compliance monitoring tool
**Effort**: 3-4 hours
**Dependencies**: Task 1.3, Task 2.1 complete

**Deliverables**:
- [ ] Create `tier4-workflows/compliance-reporting/` directory
- [ ] Create `generate-compliance-report.py` (~400 lines)
- [ ] Create report templates with color-coded output

**Dashboard Functionality**:
```python
def check_eds_compliance():
    """Validate all YAML files against EDS v1.0"""
    # Check: ads_version, type, required fields, prohibited content

def check_naming_violations():
    """Scan for ALL-CAPS filenames"""
    # Check: Pattern YYYYMMDD_HHMM_{TYPE}_{slug}.md

def check_metadata_presence():
    """Verify .metadata/ directories exist"""
    # Check: All experiments have .metadata/manifest.json

def check_artifact_placement():
    """Detect misplaced artifacts"""
    # Check: Files in correct subdirectories (assessments/, reports/, etc.)

def calculate_token_footprint():
    """Estimate token usage by tier"""
    # Check: Total lines, estimated tokens per tier

def generate_report():
    """Generate comprehensive dashboard"""
    # Output: Compliance score (%), detailed violations, fix suggestions
```

**Report Format**:
```
AI EXPERIMENT COMPLIANCE DASHBOARD
Generated: 2025-12-17 17:30:00

📊 OVERALL STATUS
├─ EDS v1.0 Compliance: 16/17 files pass (94%)
├─ Naming Violations: 0 files (100% clean)
├─ Metadata Presence: 5/5 experiments (100%)
├─ Placement Violations: 0 files (100% correct)
└─ Token Footprint: ~850 tokens (90% reduction)

🎯 COMPLIANCE SCORE: 98% (4.9/5 checks passed)
✅ EXCELLENT COMPLIANCE
```

**Success Criteria**:
- [ ] Dashboard generates report in <5 seconds
- [ ] All 5 checks operational (EDS, naming, metadata, placement, footprint)
- [ ] Color-coded output (✅ pass, ⚠️ warning, ❌ fail)
- [ ] Detailed violation reporting with fix suggestions
- [ ] Report saved to `latest-report.txt`

**Estimated Time**: 3-4 hours

---

#### **Task 2.3: Audit & Fix Existing Experiments**
**Objective**: Assess all 5 experiments, fix violations, document patterns
**Effort**: 2-3 hours
**Dependencies**: Task 2.2 complete

**Deliverables**:
- [ ] Run compliance dashboard on all experiments
- [ ] Create `tier4-workflows/audit/experiment-audit-report.yaml`
- [ ] Create batch fix scripts for common violations
- [ ] Identify deprecation candidates

**Audit Process**:
1. **Inventory**: List all 5 experiments with metadata
2. **Compliance Check**: Run dashboard per experiment
3. **Violation Analysis**: Categorize violations (naming, format, placement)
4. **Batch Fixes**: Create scripts for automated fixes
5. **Manual Review**: Flag experiments needing human review
6. **Deprecation**: Identify superseded experiments

**Audit Report Structure**:
```yaml
ads_version: "1.0"
type: audit
audit_date: "2025-12-17"
experiments_audited: 5

summary:
  total_violations: 42
  naming_violations: 18
  format_violations: 21
  placement_violations: 3

per_experiment:
  20251217_024343_image_enhancements_implementation:
    compliance_score: 14%  # 6/7 naming violations
    violations:
      - ALL-CAPS filenames: 6
      - Missing .metadata/: 1
      - Verbose prose: 7
    fix_strategy: "batch_rename + metadata_creation"

  20251128_005231_perspective_correction:
    compliance_score: 78%
    violations:
      - Missing frontmatter: 2
    fix_strategy: "add_frontmatter"

deprecation_candidates:
  - 20251122_172313_perspective_correction  # Superseded by 20251129
  - 20251128_220100_perspective_correction  # Superseded by 20251129
```

**Batch Fix Scripts**:
```bash
# fix-naming-violations.sh
for file in experiment-tracker/experiments/*/[A-Z_]*.md; do
  new_name=$(echo "$file" | sed 's/MASTER_ROADMAP/20251217_1730_guide_master-roadmap/')
  mv "$file" "$new_name"
done

# create-missing-metadata.sh
for exp in experiment-tracker/experiments/*/; do
  if [[ ! -d "$exp/.metadata" ]]; then
    mkdir "$exp/.metadata"
    echo '{"experiment_id": "'$(basename "$exp")'", "created": "'$(date -Iseconds)'"}' > "$exp/.metadata/manifest.json"
  fi
done
```

**Success Criteria**:
- [ ] All 5 experiments audited with detailed reports
- [ ] ≥3/5 experiments fixed to 80%+ compliance
- [ ] Batch fix scripts tested and validated
- [ ] Deprecation candidates documented with migration paths
- [ ] Audit report saved to `tier4-workflows/audit/`

**Estimated Time**: 2-3 hours

---

### **PHASE 3: OPTIMIZATION (Week 3-4) - MEDIUM PRIORITY**
**Goal**: Agent-specific configs and lifecycle management
**Effort**: 6-8 hours
**Dependencies**: Phase 2 complete

#### **Task 3.1: Create Agent-Specific Entry Points**
**Objective**: Write per-agent configurations (Claude, Copilot, Cursor)
**Effort**: 4-5 hours

**Deliverables**:
- [ ] Create `experiment-tracker/.ai-instructions/tier3-agents/` directory
- [ ] Create `claude/config.yaml` (~80 lines)
- [ ] Create `claude/quick-reference.yaml` (~40 lines)
- [ ] Create `claude/validation.sh` (~30 lines)
- [ ] Create `copilot/config.yaml` (~76 lines)
- [ ] Create `copilot/quick-reference.yaml` (~40 lines)
- [ ] Create `copilot/validation.sh` (~30 lines)
- [ ] Create `cursor/config.yaml` (~81 lines)
- [ ] Create `cursor/quick-reference.yaml` (~40 lines)
- [ ] Create `cursor/validation.sh` (~30 lines)

**Claude Configuration (claude/config.yaml)**:
```yaml
ads_version: "1.0"
type: agent_config
agent: claude
tier: tier3
priority: high

critical_protocols:
  - "READ tier1-sst/artifact-naming-rules.yaml BEFORE creating files"
  - "READ tier2-framework/artifact-catalog.yaml BEFORE generating artifacts"
  - "USE CLI tools ONLY - NEVER create files manually"
  - "VALIDATE with compliance-checker.py BEFORE committing"

prohibited_actions:
  critical:
    - manual_artifact_creation: "Use ./scripts/generate-assessment.py instead"
    - all_caps_filenames: "Pattern: YYYYMMDD_HHMM_{TYPE}_{slug}.md"
    - user_tutorials: "Machine-readable format ONLY"

  high:
    - emoji_usage: "Professional technical documentation"
    - verbose_prose: "Structured data preferred"

workflow_commands:
  start_experiment: "./scripts/start-experiment.py --type <type> --intention \"<goal>\""
  generate_artifact: "./scripts/generate-assessment.py --template <template>"
  validate: "python .ai-instructions/schema/compliance-checker.py <file>"
  compliance_check: "python .ai-instructions/tier4-workflows/compliance-reporting/generate-compliance-report.py"

pre_operation_checks:
  - "Confirm .metadata/ directory exists"
  - "Check artifact-catalog.yaml for artifact type requirements"
  - "Verify CLI tool availability"

post_operation_checks:
  - "Validate generated artifacts with compliance-checker.py"
  - "Confirm naming pattern compliance"
  - "Update state.json if needed"

emergency_protocols:
  violation_detected:
    action: "STOP immediately"
    fix_procedure: "Run compliance dashboard, fix violations, re-validate"

  tool_unavailable:
    action: "Alert user"
    fallback: "None - DO NOT create files manually"
```

**Quick Reference (claude/quick-reference.yaml)**:
```yaml
ads_version: "1.0"
type: quick_reference
agent: claude
tier: tier3

mandatory_rules:
  - "NEVER create files manually - use CLI tools"
  - "ALL filenames: YYYYMMDD_HHMM_{TYPE}_{slug}.md"
  - "NO ALL-CAPS filenames (CRITICAL violation)"
  - "Require .metadata/ directory in all experiments"

common_commands:
  start: "./scripts/start-experiment.py --type perspective_correction --intention \"<goal>\""
  assess: "./scripts/generate-assessment.py --template visual-evidence-cluster"
  validate: "python .ai-instructions/schema/compliance-checker.py <experiment_dir>"
  dashboard: "python .ai-instructions/tier4-workflows/compliance-reporting/generate-compliance-report.py"

validation_requirements:
  frequency: "pre_commit"
  scope: "all_yaml_files"
  enforcement: "pre_commit_hooks"
```

**Validation Script (claude/validation.sh)**:
```bash
#!/bin/bash
# Validates Claude agent configuration

AGENT_DIR="experiment-tracker/.ai-instructions/tier3-agents/claude"

echo "🔍 Validating Claude agent configuration..."

# Check config.yaml exists
if [[ ! -f "$AGENT_DIR/config.yaml" ]]; then
  echo "❌ Missing config.yaml"
  exit 1
fi

# Check dependencies
python3 experiment-tracker/.ai-instructions/schema/compliance-checker.py "$AGENT_DIR/config.yaml"

if [[ $? -eq 0 ]]; then
  echo "✅ Claude agent configuration valid"
else
  echo "❌ Claude agent configuration invalid"
  exit 1
fi
```

**Success Criteria**:
- [ ] 3/3 agents configured (Claude, Copilot, Cursor)
- [ ] All configs <100 lines each
- [ ] Validation scripts operational (pass compliance checks)
- [ ] Quick references <50 lines each
- [ ] All files pass EDS v1.0 validation

**Estimated Time**: 4-5 hours

---

#### **Task 3.2: Implement Artifact Deprecation System**
**Objective**: Create lifecycle management tooling
**Effort**: 2-3 hours
**Dependencies**: Task 3.1 complete

**Deliverables**:
- [ ] Create `experiment-tracker/.ai-instructions/DEPRECATED/` directory
- [ ] Create `DEPRECATED/README.md` with deprecation policy
- [ ] Create `tier4-workflows/lifecycle/deprecate-experiment.py` script
- [ ] Create deprecation notice templates

**Deprecation Policy (DEPRECATED/README.md)**:
```yaml
ads_version: "1.0"
type: policy
category: lifecycle

deprecation_tiers:
  tier1_active_use:
    retention: "30 days"
    applies_to: "Recently superseded experiments"
    actions:
      - at_day_0: "Move to DEPRECATED/, add notice"
      - at_day_15: "Warning - scheduled for archival"
      - at_day_30: "Delete experiment directory"

  tier2_reference:
    retention: "60 days"
    applies_to: "Historical reference experiments"
    actions:
      - at_day_0: "Move to DEPRECATED/reference/"
      - at_day_45: "Warning - scheduled for archival"
      - at_day_60: "Delete experiment directory"

  tier3_archival:
    retention: "90 days"
    applies_to: "Long-term archival experiments"
    actions:
      - at_day_0: "Export to archive.tar.gz"
      - at_day_75: "Warning - scheduled for deletion"
      - at_day_90: "Delete experiment directory"

supersession_tracking:
  required_fields:
    - superseded_by: "YYYYMMDD_HHMMSS_name"
    - supersedes: "YYYYMMDD_HHMMSS_name"
    - migration_notes: "Brief summary"

  example:
    superseded_by: "20251129_173500_perspective_correction_implementation"
    supersedes: "20251122_172313_perspective_correction"
    migration_notes: "Upgraded to Max-Edge aspect ratio preservation"
```

**Deprecation Script (deprecate-experiment.py)**:
```python
#!/usr/bin/env python3
"""Deprecate experiment with retention policy"""

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

def deprecate_experiment(experiment_id: str, tier: int, superseded_by: str = None):
    """Move experiment to DEPRECATED/ with retention notice"""
    exp_dir = Path(f"experiment-tracker/experiments/{experiment_id}")
    deprecated_dir = Path(f"experiment-tracker/.ai-instructions/DEPRECATED/tier{tier}/{experiment_id}")

    # Create deprecation notice
    retention_days = {1: 30, 2: 60, 3: 90}[tier]
    delete_date = datetime.now() + timedelta(days=retention_days)

    notice = {
        "deprecated_date": datetime.now().isoformat(),
        "retention_tier": tier,
        "delete_date": delete_date.isoformat(),
        "superseded_by": superseded_by,
        "migration_notes": "See README.md"
    }

    # Move experiment
    shutil.move(str(exp_dir), str(deprecated_dir))

    # Write notice
    with open(deprecated_dir / "DEPRECATION_NOTICE.json", "w") as f:
        json.dump(notice, f, indent=2)

    print(f"✅ Deprecated {experiment_id}")
    print(f"   Tier {tier} retention: {retention_days} days")
    print(f"   Delete after: {delete_date.strftime('%Y-%m-%d')}")
```

**Success Criteria**:
- [ ] DEPRECATED/ directory created with README
- [ ] Deprecation policy documented (3 tiers)
- [ ] deprecate-experiment.py script operational
- [ ] Retention notices automatically generated
- [ ] Migration paths documented

**Estimated Time**: 2-3 hours

---

### **PHASE 4: ENHANCEMENT (Week 4+) - OPTIONAL**
**Goal**: CLI improvements and registry
**Effort**: 5-7 hours
**Dependencies**: Phase 3 complete

#### **Task 4.1: Redesign CLI Tools for AI Agents**
**Objective**: Make CLI tools emit machine-readable output
**Effort**: 3-4 hours

**Deliverables**:
- [ ] Add `--json` flag to all CLI tools
- [ ] Structured error messages (JSON format)
- [ ] Exit codes aligned with Unix conventions
- [ ] Update CLI documentation

**JSON Output Example**:
```json
{
  "status": "success",
  "action": "generate_assessment",
  "output_file": "assessments/20251217_1703_assessment_failure-cluster.md",
  "metadata": {
    "template": "visual-evidence-cluster",
    "evidence_count": 12,
    "phase": "phase_1"
  }
}
```

**Success Criteria**:
- [ ] All CLI tools support `--json` flag
- [ ] Error messages structured (parseable by AI)
- [ ] Exit codes: 0 (success), 1 (error), 2 (validation failure)
- [ ] AI can parse outputs without regex

**Estimated Time**: 3-4 hours

---

#### **Task 4.2: Create Experiment Registry**
**Objective**: Build central registry of all experiments
**Effort**: 2-3 hours

**Deliverables**:
- [ ] Create `experiment-tracker/.registry/` directory
- [ ] Create `experiments.yaml` with all experiments
- [ ] Auto-update via CLI tools
- [ ] Query support (by status, phase, success_rate)

**Registry Structure**:
```yaml
ads_version: "1.0"
type: registry
updated: "2025-12-17T17:30:00"

experiments:
  20251217_024343_image_enhancements_implementation:
    status: "active"
    type: "custom"
    phase: "phase_1"
    success_rate: 0.0
    created: "2025-12-17T02:43:43"
    related_experiments:
      - "20251129_173500_perspective_correction_implementation"
    deprecated: false

  20251129_173500_perspective_correction_implementation:
    status: "complete"
    type: "perspective_correction"
    phase: "phase_3"
    success_rate: 1.0
    created: "2025-11-29T17:35:00"
    supersedes:
      - "20251122_172313_perspective_correction"
    deprecated: false

query_support:
  by_status: "experiments.yaml | yq '.experiments | select(.status == \"active\")'"
  by_success: "experiments.yaml | yq '.experiments | select(.success_rate > 0.8)'"
```

**Success Criteria**:
- [ ] Registry auto-generates from experiments/
- [ ] Query support via yq/jq
- [ ] CLI tools update registry automatically
- [ ] Registry validates against schema

**Estimated Time**: 2-3 hours

---

## 📋 Technical Requirements Checklist

### **Architecture & Design**
- [ ] EDS v1.0 schema aligns with ADS v1.0 principles (machine-readable, AI-only, structured)
- [ ] 4-tier hierarchy (schema → tier1-sst → tier2-framework → tier3-agents → tier4-workflows)
- [ ] YAML-driven configuration (no markdown prose in AI docs)
- [ ] Pre-commit hook enforcement (naming, metadata, compliance)

### **Integration Points**
- [ ] Reuse AgentQMS compliance-checker.py pattern
- [ ] CLI tools integrate with .ai-instructions/ structure
- [ ] Existing experiments migrate without breaking changes
- [ ] Git hooks integrate with existing workflow

### **Quality Assurance**
- [ ] All YAML files pass EDS v1.0 validation
- [ ] Pre-commit hooks block violations (<1 second execution)
- [ ] Compliance dashboard generates reports (<5 seconds)
- [ ] Batch fix scripts tested on 2+ experiments

---

## 🎯 Success Criteria Validation

### **Functional Requirements**
- [ ] EDS v1.0 schema validates all artifact types (assessment, report, guide, script, manifest)
- [ ] Pre-commit hooks block ALL-CAPS filenames, missing metadata, non-compliant YAML
- [ ] Compliance dashboard reports 5 checks (EDS, naming, metadata, placement, footprint)
- [ ] Agent configurations operational for Claude, Copilot, Cursor

### **Technical Requirements**
- [ ] Token footprint reduced by 90% (~8,500 → ~850 tokens)
- [ ] Compliance rate improves from 0% → 100% on new experiments
- [ ] Naming violations reduced from 6/7 → 0/7
- [ ] Framework stability: regression stopped, standardization enforced

### **Quality Metrics**
- [ ] Tier 1 rules <330 lines total (similar to AgentQMS: 348 lines)
- [ ] Tier 2 catalog <250 lines (similar to AgentQMS: 243 lines)
- [ ] Tier 3 agent configs <100 lines each
- [ ] Pre-commit hooks execute in <1 second
- [ ] Compliance dashboard runs in <5 seconds

---

## Risk Mitigation

### **High Risk**
1. **Existing experiments may break during migration**
   - Mitigation: Backup branch, test on 1 experiment first, incremental rollout

2. **CLI tools require breaking changes**
   - Mitigation: Maintain backward compatibility for 1 release, deprecation warnings

### **Medium Risk**
3. **AI agents may still ignore rules initially**
   - Mitigation: Pre-commit hooks enforce compliance, clear error messages

4. **Compliance overhead may slow workflow**
   - Mitigation: Fast validation (<1s), emergency skip flags

### **Low Risk**
5. **Schema evolution may require EDS v2.0**
   - Mitigation: Version field in frontmatter, migration scripts

---

## Timeline & Effort Summary

| Phase | Tasks | Effort | Dependencies |
|-------|-------|--------|--------------|
| Phase 1 | 1.1-1.3 | 7-11 hours | None (Foundation) |
| Phase 2 | 2.1-2.3 | 8-11 hours | Phase 1 complete |
| Phase 3 | 3.1-3.2 | 6-8 hours | Phase 2 complete |
| Phase 4 | 4.1-4.2 | 5-7 hours | Phase 3 complete |
| **Total** | **10 tasks** | **26-37 hours** | Sequential phases |

**Critical Path**: Phase 1 → Phase 2 (Tasks 1.1 → 1.2 → 1.3 → 2.1 → 2.2)
**Optional Work**: Phase 4 (can defer indefinitely)

---

## Next Steps

1. **Immediate**: Execute Phase 1, Task 1.1 (Create EDS v1.0 schema)
2. **Short-term**: Complete Phase 1-2 (Foundation + Automation)
3. **Long-term**: Evaluate Phase 3-4 based on Phase 2 results

**Authorization**: Proceed with autonomous execution. Update this blueprint after each task completion.

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW / MEDIUM / HIGH
### **Active Mitigation Strategies**:
1. [Mitigation Strategy 1 (e.g., Incremental Development)]
2. [Mitigation Strategy 2 (e.g., Comprehensive Testing)]
3. [Mitigation Strategy 3 (e.g., Regular Code Quality Checks)]

### **Fallback Options**:
1. [Fallback Option 1 if Risk A occurs (e.g., Simplified version of a feature)]
2. [Fallback Option 2 if Risk B occurs (e.g., CPU-only mode)]
3. [Fallback Option 3 if Risk C occurs (e.g., Phased Rollout)]

---

## 🔄 **Blueprint Update Protocol**

**Update Triggers:**
- Task completion (move to next task)
- Blocker encountered (document and propose solution)
- Technical discovery (update approach if needed)
- Quality gate failure (address issues before proceeding)

**Update Format:**
1. Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
2. Mark completed items with [x]
3. Add any new discoveries or changes to approach
4. Update risk assessment if needed

---

## 🚀 **Immediate Next Action**

**TASK:** [Description of the immediate next task]

**OBJECTIVE:** [Clear, concise goal of the task]

**APPROACH:**
1. [Step 1 to execute the task]
2. [Step 2 to execute the task]
3. [Step 3 to execute the task]

**SUCCESS CRITERIA:**
- [Measurable outcome 1 that defines task completion]
- [Measurable outcome 2 that defines task completion]

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
