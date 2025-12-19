---
ads_version: "1.0"
type: plan
title: Border Removal Preprocessing Experiment Plan (Option C)
status: active
created: 2025-12-18T19:05:00+09:00
updated: 2025-12-18T19:05:00+09:00
experiment_id: 20251218_190000_border_removal_preprocessing
phase: phase_0
tags: [border-removal, preprocessing, option-c]
related_artifacts:
 - ../20251217_024343_image_enhancements_implementation/.metadata/plans/20251218_1905_plan_border-removal-experiment.md
---

# Plan: Border Removal Preprocessing (Option C)

source_of_truth:
- ../20251217_024343_image_enhancements_implementation/.metadata/plans/20251218_1905_plan_border-removal-experiment.md

scope:
- Implement and validate border removal methods to prevent deskew failures caused by borders.
- Pipeline placement: Option C (conditional on high skew), with double deskew.

integration_contract:
- input: image (HWC, uint8)
- output: cropped image (HWC, uint8) and metrics
- behavior:
 - if border confidence < threshold: return original
 - if min_area_ratio not met: return original

configuration_schema:
- preprocessing:
 - border_removal:
 - enabled: bool
 - method: one_of [canny, morph, hough]
 - min_area_ratio: float
 - confidence_threshold: float
 - fallback_to_original: bool

phases:
- phase_1_research_and_baseline:
 - collect border cases (search high skew and zero predictions)
 - build synthetic border dataset
 - record baseline skew metrics
- phase_2_implementation:
 - implement canny+contours
 - implement morphological approach
 - optional: implement hough-lines approach
 - unify interface in scripts/border_remover.py
- phase_3_validation:
 - measure boundary detection accuracy
 - measure false crops on border-free controls
 - measure processing time
 - measure skew improvement (000732 primary)
- phase_4_integration_notes:
 - document Option C gating rule: if abs(skew_deg) > 20
 - document failure modes and safe fallbacks

success_criteria:
- 000732: abs(skew_deg_after) < 15
- false crops: < 0.05 on border-free controls
- latency: < 50 ms per image on target hardware
