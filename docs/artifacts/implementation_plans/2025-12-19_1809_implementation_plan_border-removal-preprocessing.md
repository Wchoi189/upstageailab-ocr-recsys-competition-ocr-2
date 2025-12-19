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
- **STATUS:** Phase 1 - Initial Validation (In Progress)
- **CURRENT STEP:** Phase 1, Task 1.2 - Dataset Collection
- **LAST COMPLETED TASK:** Task 1.1 - Border removal script validation on 000732
- **NEXT TASK:** Expand dataset beyond zero_prediction_worst_performers (only 1/6 images has skew > 10°)

### Implementation Outline (Checklist)

#### **Phase 1: Initial Validation (Days 1-2)**
1. [x] **Task 1.1: Validate Border Removal Scripts**
   - [x] Test `border_removal.py` on sample image (drp.en_ko.in_house.selectstar_000732.jpg)
   - [x] Verify all 3 methods execute: Canny (89.9ms), Morph (16.1ms), Hough (14.3ms)
   - [x] Generate comparison visualization
   - **Result:** All methods failed to crop (confidence/area < 0.75 threshold). Image may lack clear borders.

2. [ ] **Task 1.2: Expand Dataset Sources**
   - [ ] Scan additional directories: `data/test_large/`, full training set via experiment registry
   - [ ] Target: 50 images with visible borders (black/white frames around document)
   - [ ] Criteria: Manual inspection OR dark edge detection > 20% on 3+ sides

3. [ ] **Task 1.3: Generate Synthetic Border Dataset**
   - [ ] Script: `generate_border_dataset.py` - add 10-50px borders to clean samples
   - [ ] Vary: border thickness, color (black/white), skew angles (-30° to +30°)
   - [ ] Output: 100 samples (70 train / 30 test) with ground truth crop coordinates

#### **Phase 2: Algorithm Refinement & VLM Validation (Days 3-5)**
4. [x] **Task 2.1: Border Removal Methods (Implemented)**
   - [x] Canny edge detection + contour analysis
   - [x] Morphological operations + connected components
   - [x] Hough line transform + rectangular detection
   - **Location:** `experiment-tracker/experiments/20251218_1900_border_removal_preprocessing/scripts/border_removal.py`

5. [ ] **Task 2.2: Tune Detection Thresholds**
   - [ ] Lower `min_area_ratio` from 0.75 to 0.60 (current: all methods reject crops)
   - [ ] Adjust Canny thresholds: test ranges [30-100, 100-200]
   - [ ] Test on synthetic dataset with known borders

6. [ ] **Task 2.3: VLM Quality Assessment (Before/After)**
   - [ ] Run VLM `image_quality` mode on original images
   - [ ] Run VLM `enhancement_validation` mode on border-removed outputs
   - [ ] Compare: border severity scores, text readability, skew measurements

7. [ ] **Task 2.4: Batch Processing Pipeline**
   - [ ] Process all border cases from manifest via `border_removal.py --manifest`
   - [ ] Generate comparison grid: original + 3 methods side-by-side
   - [ ] Export metrics to `border_removal_results.json`

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
- [ ] Validation: Boundary detection accuracy on synthetic dataset (IoU metric)
- [ ] Safety: False positive rate < 0.05 on clean samples (no borders)
- [ ] Performance: Latency profiling on representative image sizes
- [ ] Quality: VLM assessment of border removal effectiveness (readability, artifacts)

---

## **Success Criteria Validation**

### **Functional Requirements**
- [ ] Image_000732: `abs(skew_deg_after) < 15°` (baseline: 45°+)
- [ ] Boundary detection: IoU > 0.95 on 90% of synthetic border dataset
- [ ] No false crops: False crop rate < 0.05 on clean samples (n=100)
- [ ] VLM validation: Comprehensive reports for baseline, validation, and quality assessment

### **Technical Requirements**
- [x] Standalone execution: CLI script with visualization and comparison features
- [x] Safe fallback: Original image returned if `area_ratio < min_area_ratio`
- [ ] Metrics export: JSONL logging for batch processing results
- [ ] VLM integration: Before/after quality assessment workflow
- [ ] Documentation: Failure mode analysis, method comparison, integration recommendations

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

**TASK:** Expand border case dataset beyond zero_prediction_worst_performers

**OBJECTIVE:** Collect 50 images with visible borders for algorithm tuning

**APPROACH:**
1. Scan `data/test_large/` with `collect_border_cases.py --skew-threshold 10`
2. Manual inspection: Identify images with black/white frames (borders)
3. Generate synthetic border dataset: 100 samples with known ground truth

**SUCCESS CRITERIA:**
- 50+ real border cases identified (manual inspection or dark edge detection)
- 100 synthetic border samples created with ground truth crop coordinates
- Updated manifest: `border_cases_manifest.json` with expanded dataset

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
