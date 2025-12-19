# Border Removal Preprocessing Experiment (Option C)

experiment_id: 20251218_1900_border_removal_preprocessing
parent_experiment_id: 20251217_024343_image_enhancements_implementation
status: active

objective:
- Remove border artifacts that cause extreme skew misdetection (e.g., image 000732).
- Preserve border-free images (minimize false crops).

strategy_option_c:
- pass_1: estimate skew on input
- gate: if abs(skew_deg) > 20
  - apply border removal
  - re-estimate skew on cropped output
- fallback: if border detection confidence < threshold, return original

success_criteria:
- 000732: abs(skew_deg_after) < 15
- false_crops_rate_border_free: < 0.05
- processing_time_ms_per_image: < 50

entrypoints:
- scripts/find_border_cases.py
- scripts/generate_synthetic_borders.py
- scripts/run_border_removal_methods.py
- scripts/compare_methods.py

outputs:
- artifacts/border_cases_manifest.json
- artifacts/synthetic_manifest.json
- artifacts/baseline_metrics.json
- artifacts/method_comparison.json
- outputs/visualizations/

references:
- source_plan: ../20251217_024343_image_enhancements_implementation/.metadata/plans/20251218_1905_plan_border-removal-experiment.md
- source_scoping_assessment: ../20251217_024343_image_enhancements_implementation/.metadata/assessments/20251218_1900_assessment_border-removal-scoping.md
