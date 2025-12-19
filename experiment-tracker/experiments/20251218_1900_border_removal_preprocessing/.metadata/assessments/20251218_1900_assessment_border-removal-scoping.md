---
ads_version: "1.0"
type: assessment
title: Border Removal Experiment Scoping (Emoji-Free Copy)
status: complete
created: 2025-12-18T19:00:00+09:00
updated: 2025-12-18T19:00:00+09:00
experiment_id: 20251218_190000_border_removal_preprocessing
phase: phase_0
priority: medium
evidence_count: 1
tags: [experiment-scoping, border-removal]
related_artifacts:
 - ../20251217_024343_image_enhancements_implementation/.metadata/assessments/20251218_1900_assessment_border-removal-scoping.md
---

# Border Removal Scoping

decision:
- create separate experiment: true

drivers:
- border removal is document boundary detection + crop; distinct from enhancement steps.
- requires method comparison and false-crop evaluation.
- can run in parallel with integration of background normalization and deskewing.

primary_target:
- image_id: 000732
- skew_baseline_deg: -83
- skew_target_abs_deg: < 15

risk_controls:
- confidence threshold + fallback to original
- min_area_ratio gating to prevent over-cropping
- border-free control set to estimate false crops
