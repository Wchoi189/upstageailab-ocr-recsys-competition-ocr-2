# Handoff: 20251218_1900_border_removal_preprocessing

context:
- Problem: border artifacts drive extreme skew misdetection (image 000732 reported at -83 deg).
- Goal: detect document boundary, crop to content, then deskew reliably.

work_items:
- Phase 1: identify border-affected images; generate synthetic borders; record baseline metrics.
- Phase 2: implement border removal methods (canny+contours, morphology, optional hough-lines).
- Phase 3: compare methods; select based on accuracy/false-crops/time.
- Phase 4: document integration contract for Option C gating.

key_files:
- state: state.yml
- plan: .metadata/plans/20251218_1905_plan_border-removal-experiment.md
- method API: scripts/border_remover.py
