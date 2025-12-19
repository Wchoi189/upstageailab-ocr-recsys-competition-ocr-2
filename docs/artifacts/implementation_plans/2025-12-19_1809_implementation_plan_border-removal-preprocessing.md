---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation', 'plan', 'development']
title: "Border Removal Preprocessing Experiment (Option C)"
date: "2025-12-19 18:09 (KST)"
branch: "main"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Border Removal Preprocessing Experiment (Option C)**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Border Removal Preprocessing Experiment (Option C)

## Progress Tracker
- **STATUS:** Phase 1 - Dataset Curation
- **CURRENT STEP:** Phase 1, Task 1.1 - Border Case Collection
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** Query experiment registry for images with abs(skew_deg) > 20 and zero predictions

### Implementation Outline (Checklist)

#### **Phase 1: Dataset Curation & Baseline (Days 1-3)**
1. [ ] **Task 1.1: Collect Border Failure Cases**
   - [ ] Query `ocr/experiment_registry.py` for samples: `abs(skew_deg) > 20` AND `predicted_text == ""`
   - [ ] Extract 50 worst performers: sort by `skew_deg DESC`, filter for visible black borders (manual inspection)
   - [ ] Validate with VLM `image_quality` mode: confirm border presence, measure severity (1-10 scale)

2. [ ] **Task 1.2: Generate Synthetic Border Dataset**
   - [ ] Create `scripts/generate_border_dataset.py`: add 10-50px black borders to 100 clean samples
   - [ ] Vary border thickness, aspect ratios, and skew angles (-30° to +30°)
   - [ ] Split: 70 train / 30 test (for boundary detection algorithm validation)

3. [ ] **Task 1.3: Record Baseline Metrics**
   - [ ] Run deskew on border dataset WITHOUT border removal: record `skew_deg_before`, `skew_deg_after`
   - [ ] Measure failure rate: count samples where `abs(skew_deg_after) > 15`
   - [ ] Store in `outputs/experiments/20251218_1900_border_removal_preprocessing/baseline_metrics.json`

#### **Phase 2: Border Removal Implementation (Days 4-7)**
4. [ ] **Task 2.1: Implement Canny+Contours Method**
   - [ ] Create `ocr/datasets/transforms/border_removal.py` with class `CannyBorderRemover`
   - [ ] Algorithm: Canny edge detection → find largest rectangular contour → crop to inner bounds
   - [ ] Expose params: `canny_low=50`, `canny_high=150`, `min_area_ratio=0.75`

5. [ ] **Task 2.2: Implement Morphological Method**
   - [ ] Add class `MorphBorderRemover` to same module
   - [ ] Algorithm: Otsu threshold → morphological closing (kernel=5x5) → find largest connected component
   - [ ] Expose params: `morph_kernel_size=5`, `min_area_ratio=0.75`

6. [ ] **Task 2.3: Implement Hough Lines Method (Optional)**
   - [ ] Add class `HoughBorderRemover` to same module
   - [ ] Algorithm: Canny edges → Hough line detection → find 4 dominant lines → compute intersection rectangle
   - [ ] Expose params: `hough_threshold=100`, `min_line_length=100`, `max_line_gap=10`

7. [ ] **Task 2.4: Unified Interface**
   - [ ] Create factory function `create_border_remover(method: str, **kwargs) -> BorderRemoverBase`
   - [ ] Define `BorderRemoverBase` abstract class: `remove_border(image: np.ndarray) -> Tuple[np.ndarray, dict]`
   - [ ] Return metrics dict: `{cropped: bool, confidence: float, area_ratio: float, method: str}`

#### **Phase 3: Validation & Performance Testing (Days 8-10)**
8. [ ] **Task 3.1: Boundary Detection Accuracy**
   - [ ] Run all 3 methods on synthetic border dataset (30 test samples)
   - [ ] Compute IoU between detected crop region and ground truth
   - [ ] Target: IoU > 0.95 for 90% of samples

9. [ ] **Task 3.2: False Positive Testing**
   - [ ] Run border removal on 100 clean samples (no borders)
   - [ ] Measure false crop rate: count samples where `area_ratio < 0.98`
   - [ ] Target: False crop rate < 0.05

10. [ ] **Task 3.3: Performance Benchmarking** (Optional - not required)
    - [ ] Profile each method on representative image set (if time permits)
    - [ ] Measure: average processing time per image
    - [ ] Note: No latency requirements for this experiment

11. [ ] **Task 3.4: Skew Improvement Validation**
    - [ ] Run Option C pipeline on image_000732: (1) border removal → (2) deskew
    - [ ] Measure `skew_deg_after`: target < 15°
    - [ ] Compare against baseline (no border removal): document improvement

12. [ ] **Task 3.5: VLM Analysis Workflow**
    - [ ] Phase 1: VLM baseline assessment (10 images, ~10 min)
    - [ ] Phase 2: VLM validation analysis (30×3 images, ~1.5 hours)
    - [ ] Phase 3: VLM quality assessment (30 images, ~45 min)
    - [ ] Generate comprehensive VLM reports for all cases
    - [ ] No latency constraints - prioritize quality over speed

---

## **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Abstract base class: `BorderRemoverBase` with `remove_border()` interface
- [ ] Factory pattern: `create_border_remover(method, **kwargs)` for strategy selection
- [ ] Module location: `experiment-tracker/experiments/20251218_1900_border_removal_preprocessing/scripts/border_removal.py` (standalone experiment)
- [ ] Metrics schema: `{cropped: bool, confidence: float, area_ratio: float, method: str, processing_time_ms: float}`
- [ ] Fallback behavior: Return original image if `confidence < threshold` OR `area_ratio < min_area_ratio`

### **Integration Points** (DEFERRED to Options A/B)
- [ ] **NOT IN SCOPE**: Pipeline integration deferred to Options A/B
- [ ] **EXPERIMENT ONLY**: Focus on validating methods and generating reports
- [ ] Metrics export: Log border removal outcomes to `outputs/experiments/{exp_id}/border_metrics.jsonl`
- [ ] VLM analysis: **EXTENSIVE USE** via `AgentQMS.vlm.cli.analyze_image_defects` (no latency constraints)
- [ ] Option C conditional logic: Document findings for future integration

### **Quality Assurance**
- [ ] Unit tests: 3 classes × 5 test cases = 15 tests in `tests/ocr/transforms/test_border_removal.py`
- [ ] Integration test: End-to-end Option C pipeline on 30-sample border dataset
- [ ] Performance test: Latency profiling on 1000 images (p50, p95, p99)
- [ ] Regression test: False crop rate < 0.05 on 100 clean samples

---

## **Success Criteria Validation**

### **Functional Requirements**
- [ ] Image_000732: `abs(skew_deg_after) < 15°` (baseline: 45°+)
- [ ] Boundary detection: IoU > 0.95 on 90% of synthetic border dataset
- [ ] No false crops: False crop rate < 0.05 on clean samples (n=100)
- [ ] VLM validation: Comprehensive reports for baseline, validation, and quality assessment

### **Technical Requirements**
- [ ] Config-driven: All parameters exposed in `configs/data/preprocessing.yaml`
- [ ] Type safety: Full type hints, validated with `pyright --strict`
- [ ] Modularity: 3 border removal algorithms share common interface (`BorderRemoverBase`)
- [ ] Observability: Metrics logged to `outputs/experiments/{exp_id}/border_metrics.jsonl` with timestamps
- [ ] Safe fallback: Original image returned if confidence/area checks fail

---

## **Risk Mitigation & Fallbacks**

### **Current Risk Level**: MEDIUM

### **Active Mitigation Strategies**:
1. **Overfitting to synthetic borders**: Validate on real border failures (image_000732 + 50 worst performers)
2. **False crops on textured images**: Enforce `min_area_ratio=0.75` threshold; log all rejected crops
3. **Latency regression**: Profile each algorithm independently; disable Hough if p95 > 50ms

### **Fallback Options**:
1. **If IoU < 0.95**: Revert to Canny-only method (simplest, fastest)
2. **If false crop rate > 0.05**: Raise `confidence_threshold` to 0.8 or disable border removal
3. **If image_000732 still fails**: Document as known limitation; escalate to manual annotation for ground truth validation

---

## **Blueprint Update Protocol**

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

## **Immediate Next Action**

**TASK:** Query experiment registry for border failure cases (abs(skew_deg) > 20, zero predictions)

**OBJECTIVE:** Collect 50 real-world border failures for algorithm validation

**APPROACH:**
1. Run query: `python -c "from ocr.experiment_registry import get_failures; print(get_failures(min_skew=20, pred_text=''))"`
2. Sort results by `skew_deg DESC`, extract top 50 image paths
3. Copy images to `data/border_failures/real_world_samples/`

**SUCCESS CRITERIA:**
- 50 images extracted with confirmed `abs(skew_deg) > 20`
- Manifest file created: `data/border_failures/real_world_samples/manifest.json` with `{image_id, skew_deg, file_path}[]`

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
