# EDS v1.0 Compliance Report

**Generated**: 2025-12-17 17:59:59

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Experiments | 5 |
| Total Artifacts | 42 |
| Compliant Artifacts | 33 (78.6%) |
| Violations | 9 (21.4%) |
| Average Compliance | 57.0% |

## Critical Violations Summary

| Violation Type | Count | Percentage |
|----------------|-------|------------|
| ALL-CAPS Filenames | 10 | 23.8% |
| Missing Frontmatter | 0 | 0.0% |
| Missing .metadata/ Directory | 0 | 0.0% |

## Experiment Compliance (Worst First)

| Experiment ID | Artifacts | Compliant | Violations | Compliance | .metadata/ | ALL-CAPS | Missing FM |
|---------------|-----------|-----------|------------|------------|------------|----------|------------|
| 20251128_005231_perspective_correction | 1 | 0 | 1 | 0.0% | ✅ | 0 | 0 |
| 20251128_220100_perspective_correction | 4 | 0 | 4 | 0.0% | ✅ | 0 | 0 |
| 20251122_172313_perspective_correction | 27 | 23 | 4 | 85.2% | ✅ | 0 | 0 |
| 20251129_173500_perspective_correction_implementation | 4 | 4 | 0 | 100.0% | ✅ | 4 | 0 |
| 20251217_024343_image_enhancements_implementation | 6 | 6 | 0 | 100.0% | ✅ | 6 | 0 |

## Detailed Violations

### 20251128_005231_perspective_correction

**Compliance**: 0.0% (0/1 artifacts)

**Non-Compliant Artifacts**:

- `incident_reports/20251128_mask_based_edge_detection_robustness_and_affine_trap.md`
  - • Missing required field: ads_version
  - • Missing required field: type
  - • Missing required field: created

### 20251128_220100_perspective_correction

**Compliance**: 0.0% (0/4 artifacts)

**Non-Compliant Artifacts**:

- `assessments/run_01_angle_bucketing_initial.md`
  - • Missing required field: ads_version
  - • Missing required field: type
  - • Missing required field: experiment_id
- `assessments/run_02_coordinate_inversion_fix.md`
  - • Missing required field: ads_version
  - • Missing required field: type
  - • Missing required field: experiment_id
- `assessments/run_03_geometric_synthesis_success.md`
  - • Missing required field: ads_version
  - • Missing required field: type
  - • Missing required field: experiment_id
- `incident_reports/20251129_0218-variable-shadowing-bug-geometric-synthesis.md`
  - • Missing required field: ads_version
  - • Missing required field: type
  - • Missing required field: created

### 20251122_172313_perspective_correction

**Compliance**: 85.2% (23/27 artifacts)

**Non-Compliant Artifacts**:

- `assessments/mask_only_edge_detection_with_heuristics.md`
  - • Missing required field: ads_version
  - • Missing required field: type
  - • Missing required field: created
- `incident_reports/20251124_1338-catastrophic-perspective-correction-failure-on-optimal-orientation-images.md`
  - • Missing required field: ads_version
  - • Missing required field: type
  - • Missing required field: created
- `incident_reports/20251124_1347-improved-edge-based-approach-performance-regression-and-technical-bugs.md`
  - • Missing required field: ads_version
  - • Missing required field: type
  - • Missing required field: created
- `incident_reports/20251124_1925-edge-line-selection-and-area-metric-defects.md`
  - • Missing required field: ads_version
  - • Missing required field: type
  - • Missing required field: created


## Remediation Priorities

### Priority 2: ALL-CAPS Filenames

**Count**: 10 files

**Action**: Rename using lowercase-hyphenated pattern:

**20251129_173500_perspective_correction_implementation**:
- `EXECUTION_SUMMARY.md` → `20251217_1200_guide_execution-summary.md`
- `ISSUES_FIXED.md` → `20251217_1200_guide_issues-fixed.md`
- `TEST_INSTRUCTIONS.md` → `20251217_1200_guide_test-instructions.md`
- `TEST_RESULTS_ANALYSIS.md` → `20251217_1200_guide_test-results-analysis.md`

**20251217_024343_image_enhancements_implementation**:
- `CURRENT_STATE_SUMMARY.md` → `20251217_1200_guide_current-state-summary.md`
- `ENHANCEMENT_QUICK_REFERENCE.md` → `20251217_1200_guide_enhancement-quick-reference.md`
- `EXECUTIVE_SUMMARY.md` → `20251217_1200_guide_executive-summary.md`
- `MASTER_ROADMAP.md` → `20251217_1200_guide_master-roadmap.md`
- `PRIORITY_PLAN_REVISED.md` → `20251217_1200_guide_priority-plan-revised.md`
- `VLM_INTEGRATION_GUIDE.md` → `20251217_1200_guide_vlm-integration-guide.md`

