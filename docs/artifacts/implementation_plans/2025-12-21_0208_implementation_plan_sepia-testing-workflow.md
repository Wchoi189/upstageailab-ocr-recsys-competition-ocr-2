---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation', 'plan', 'development']
title: "Sepia Enhancement Testing and Validation Workflow"
date: "2025-12-21 02:08 (KST)"
branch: "main"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Sepia Enhancement Testing and Validation Workflow**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Sepia Enhancement Testing and Validation Workflow

## Progress Tracker
- **STATUS:** Implementation Complete - Testing Phase
- **CURRENT STEP:** Phase 1 - Isolated Sepia Testing
- **LAST COMPLETED TASK:** Scripts created, documentation written, state updated
- **NEXT TASK:** Run isolated sepia tests on reference samples

### Implementation Outline (Checklist)

#### **Phase 1: Isolated Testing (Immediate - 30 minutes)**
1. [x] **Task 1.1: Script Implementation**
   - [x] Create sepia_enhancement.py with 4 methods
   - [x] Implement metrics calculation
   - [x] Add batch processing support

2. [x] **Task 1.2: Infrastructure Setup**
   - [x] Create output directories
   - [x] Write comprehensive documentation
   - [x] Update experiment state

3. [ ] **Task 1.3: Execute Isolated Tests**
   - [ ] Test all 4 sepia methods on drp...000732.jpg
   - [ ] Test on drp...000712_sepia.jpg reference
   - [ ] Generate and review metrics

#### **Phase 2: Comparative Analysis (Next - 45 minutes)**
4. [x] **Task 2.1: Comparison Framework**
   - [x] Create compare_sepia_methods.py
   - [x] Implement grid visualization
   - [x] Add metrics export (JSON, tables)

5. [ ] **Task 2.2: Run Comparisons**
   - [ ] Generate comparison grids for reference samples
   - [ ] Analyze metrics (tint, contrast, edge strength)
   - [ ] Document quantitative findings

#### **Phase 3: Pipeline Integration (1 hour total)**
6. [x] **Task 3.1: Pipeline Script**
   - [x] Create sepia_perspective_pipeline.py
   - [x] Implement perspective correction stage
   - [x] Add optional deskewing support

7. [ ] **Task 3.2: Pipeline Testing**
   - [ ] Test warm sepia + perspective correction
   - [ ] Compare against gray-world normalization pipeline
   - [ ] Measure total pipeline timing

#### **Phase 4: VLM Validation (1 hour total)**
8. [x] **Task 4.1: VLM Integration**
   - [x] Create vlm_validate_sepia.sh
   - [x] Configure Dashscope API calls
   - [x] Structure VLM prompts for comparison

9. [ ] **Task 4.2: Execute VLM Validation**
   - [ ] Submit comparison grids to Qwen3 VL Plus
   - [ ] Parse and review VLM assessments
   - [ ] Compare VLM scores vs quantitative metrics

#### **Phase 5: OCR End-to-End (2 hours total)**
10. [ ] **Task 5.1: OCR Testing Setup**
    - [ ] Prepare sepia-enhanced test set
    - [ ] Configure inference with epoch-18 checkpoint
    - [ ] Set up results comparison framework

11. [ ] **Task 5.2: Execute OCR Tests**
    - [ ] Run OCR on sepia-enhanced images
    - [ ] Run OCR on gray-world normalized images
    - [ ] Compare prediction accuracy and reliability

12. [ ] **Task 5.3: Analysis and Decision**
    - [ ] Analyze OCR accuracy differences
    - [ ] Document findings in experiment state
    - [ ] Make integration recommendation

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [x] Modular sepia enhancement classes
- [x] Metrics calculation framework
- [x] Pipeline integration architecture
- [x] State management (experiment state.yml)

### **Integration Points**
- [x] Integration with existing perspective correction
- [x] Optional deskewing integration
- [x] VLM validation via Dashscope API
- [x] OCR checkpoint compatibility (epoch-18)

### **Quality Assurance**
- [ ] Isolated testing on reference samples
- [ ] Comparative analysis vs alternatives
- [ ] VLM visual quality validation
- [ ] OCR end-to-end accuracy testing

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [x] 4 sepia methods implemented (classic, adaptive, warm, contrast)
- [x] Comparison framework operational
- [x] Full pipeline integration complete
- [ ] Processing time < 100ms per image (to be validated)
- [ ] OCR accuracy improves on problematic samples (to be validated)

### **Technical Requirements**
- [x] Code documented with docstrings
- [x] Type hints where applicable
- [x] Scripts support single file and batch processing
- [x] Metrics exported to JSON format
- [x] Comprehensive documentation provided

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW

### **Active Mitigation Strategies**:
1. Keep gray-world normalization as fallback if sepia underperforms
2. Test multiple sepia methods to identify best performer
3. VLM validation confirms quantitative metrics
4. Comprehensive documentation enables reproducibility

### **Fallback Options**:
1. If sepia worse than normalization: Document findings, continue with gray-world
2. If processing too slow: Accept higher latency if accuracy improves significantly
3. If method selection unclear: Default to warm sepia (theoretical optimum)
4. If VLM inconclusive: Rely on quantitative metrics + OCR accuracy

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

**TASK:** Run isolated sepia testing on reference samples

**OBJECTIVE:** Validate that all 4 sepia methods execute correctly and generate metrics

**APPROACH:**
```bash
cd experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/scripts

# Test on problematic sample
python sepia_enhancement.py \
  --input ../artifacts/sepia/drp.en_ko.in_house.selectstar_000732_REMBG.jpg \
  --method all \
  --output ../outputs/sepia_tests/

# Test on reference quality sample
python sepia_enhancement.py \
  --input ../artifacts/sepia/drp.en_ko.in_house.selectstar_000712_sepia.webp \
  --method all \
  --output ../outputs/sepia_tests/
```

**SUCCESS CRITERIA:**
- All 4 sepia methods execute without errors
- Metrics generated for each method (tint, contrast, brightness, edge strength)
- Output images saved to sepia_tests/ directory
- Processing time < 100ms per method
- Visual inspection shows sepia tone applied correctly

**EXPECTED OUTPUTS:**
- `*_sepia_classic.jpg`
- `*_sepia_adaptive.jpg`
- `*_sepia_warm.jpg`
- `*_sepia_contrast.jpg`
- Console output with metrics for each method

**NEXT TASK AFTER COMPLETION:**
Run comparison analysis (Task 2.2) to compare sepia methods against gray-scale and normalization

---

## 📍 **Quick Reference Commands**

### Isolated Testing
```bash
python sepia_enhancement.py --input <image> --method all --output ../outputs/sepia_tests/
```

### Comparison Analysis
```bash
python compare_sepia_methods.py --input <image> --output ../outputs/sepia_comparison/ --save-metrics
```

### Full Pipeline
```bash
python sepia_perspective_pipeline.py --input <image> --sepia-method warm --output ../outputs/sepia_pipeline/
```

### VLM Validation
```bash
export DASHSCOPE_API_KEY='your_key'
./vlm_validate_sepia.sh ../outputs/sepia_comparison/
```

---

*This implementation plan tracks the sepia enhancement testing workflow for experiment 20251217_024343_image_enhancements_implementation.*
