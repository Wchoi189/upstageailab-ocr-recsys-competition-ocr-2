---
ads_version: "1.0"
type: assessment
title: Border Removal Experiment Scoping Assessment
status: complete
created: 2025-12-18T19:00:00+09:00
updated: 2025-12-18T19:00:00+09:00
experiment_id: 20251217_024343_image_enhancements_implementation
phase: phase_0
priority: medium
evidence_count: 1
tags: [experiment-scoping, border-removal, week3]
related_artifacts:
 - 20251218_1850_report_week1-2-completion-integration-plan.md
---

# Border Removal Experiment Scoping Assessment

## Question: New Experiment vs Direct Implementation?

**Answer**: **New Experiment Required** 

## Rationale

### Why New Experiment?

#### 1. Different Problem Domain
- **Current experiment**: Image quality enhancements (tint, contrast, skew)
- **Border removal**: Document boundary detection and extraction
- **Distinction**: Enhancement vs correction/preprocessing cleanup

#### 2. Specific, Well-Defined Problem
- **Root cause**: Black borders causing -83° skew misdetection on image 000732
- **Solution scope**: Border detection → document extraction → crop
- **Clear success criteria**: Resolve -83° to <15° baseline skew

#### 3. Natural Completion Point
- Current experiment reached logical conclusion:
 - Week 1: Background normalization (4.75/5) 
 - Week 2: Text deskewing (4.6/5) 
 - Both validated (quantitative + VLM) 
- Continuing as "Week 3" would bloat experiment scope

#### 4. Independent Implementation Path
- Border removal can be developed/tested independently
- Doesn't require background norm or deskewing to be integrated first
- Can run in parallel with Options A/B (pipeline integration, OCR validation)

#### 5. Different Testing Requirements
- Current experiment: 6 worst-performer images
- Border removal: Needs specific test set with border images
- May require synthetic data generation (images with various border types)

### Why NOT Direct Implementation?

#### 1. Requires Research Phase
- Multiple approaches: Canny edges, contour detection, morphological ops, ML-based
- Need to test which works best for receipts/invoices
- Not a "known solution" - requires experimentation

#### 2. Validation Complexity
- Must ensure no false positives (cropping actual content)
- Edge cases: Rotated documents, partial borders, noise
- Requires careful metric design beyond simple before/after

#### 3. Integration Dependencies
- May need to run BEFORE deskewing (borders affect angle detection)
- Pipeline ordering: border removal → background norm → deskewing
- Requires architecture decisions

## Proposed New Experiment

### Experiment ID
`20251218_1900_border_removal_preprocessing`

### Scope
**Problem**: Black borders cause severe skew misdetection (e.g., -83° on image 000732)

**Goal**: Detect document boundaries and crop to content area, eliminating border interference

**Approach**:
1. Research phase: Survey border detection methods (Canny + contours, morphological, ML)
2. Implementation: Test 2-3 methods on images with/without borders
3. Validation: Measure skew improvement on border-affected images
4. Integration: Determine pipeline placement (before/after deskewing)

### Success Criteria
- Resolve -83° misdetection to <15° on 000732
- No false crops on border-free images (100% preservation)
- Processing time <50ms per image
- Works on various border types (black, white, colored)

### Test Data Requirements
- **Existing**: 1 confirmed border case (000732)
- **Additional**: Find 5-10 more images with borders from test set
- **Synthetic**: Generate borders on clean images (validation set)

### Timeline
- Research phase: 1-2 days
- Implementation: 2-3 days
- Validation: 1-2 days
- Integration: 1 day
- **Total**: 5-8 days

### Dependencies
- **Independent**: Can run in parallel with current experiment integration
- **Pipeline impact**: May require reordering (border removal first)
- **OCR validation**: Should retest after border removal integrated

## Parallel Execution Plan

### Track 1: Current Experiment Integration (Options A/B)
**Owner**: Primary development focus
**Timeline**: Week 3 (7 days)

Tasks:
1. Integrate gray-world + Hough deskewing (2 days)
2. OCR end-to-end validation (3 days)
3. Production deployment prep (2 days)

### Track 2: Border Removal Experiment (Option C)
**Owner**: Can be delegated or done in parallel
**Timeline**: Overlaps with Week 3

Tasks:
1. Create new experiment structure (AgentQMS) (0.5 days)
2. Research border detection methods (1 day)
3. Implement candidate methods (2 days)
4. Validate on test images (1.5 days)
5. Document findings and integration plan (1 day)

### Synchronization Points
- **Day 3**: Share findings from Track 2 research with Track 1 (may inform pipeline design)
- **Day 5**: Track 1 OCR baseline complete, Track 2 implementation complete
- **Day 7**: Integrate Track 2 results into Track 1 pipeline, revalidate OCR

## Recommendation

### Immediate Actions (This Conversation)
1. Create implementation plan for border removal experiment
2. Update current experiment state.json (mark Week 3 as deferred)
3. Document handoff information for new conversation

### New Conversation (Border Removal)
1. Create new experiment via AgentQMS tools
2. Establish baseline (identify all border-affected images)
3. Research and implement border detection methods
4. Validate and document results
5. Create integration guide for main pipeline

### Continuation Conversation (Integration)
1. Implement pipeline integration (gray-world + Hough)
2. Run OCR validation with epoch-18_step-001957.ckpt
3. Incorporate border removal when ready (Track 2 output)
4. Final production deployment

## Decision Matrix

| Factor | New Experiment | Direct Implementation |
|--------|----------------|----------------------|
| Problem complexity | High (requires research) | Too complex for direct impl |
| Testing requirements | Needs dedicated test set | Would bloat current experiment |
| Independence | Can run parallel | Would block integration |
| Traceability | Clear experiment boundary | Mixed with enhancements |
| Maintainability | Focused scope | Bloated experiment |
| **Recommendation** | ** PROCEED** | Not recommended |

---

**Conclusion**: Border removal should be a **new experiment** (`20251218_1900_border_removal_preprocessing`) that runs in parallel with current experiment integration. This allows clean separation of concerns, independent testing, and better traceability.

---
*EDS v1.0 compliant | Experiment Scoping Analysis | Recommendation: New Experiment*
