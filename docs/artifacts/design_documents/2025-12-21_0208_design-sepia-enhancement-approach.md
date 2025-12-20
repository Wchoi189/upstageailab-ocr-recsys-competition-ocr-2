---
ads_version: "1.0"
type: "design"
category: "architecture"
status: "active"
version: "1.0"
tags: ['design', 'architecture', 'specification']
title: "Sepia Enhancement Approach for OCR Preprocessing"
date: "2025-12-21 02:08 (KST)"
branch: "main"
---

# Sepia Enhancement Approach for OCR Preprocessing

## Overview

This document describes the design approach for implementing sepia tone enhancement as an alternative to gray-scale conversion and gray-world normalization in the OCR preprocessing pipeline. Based on user observations, sepia enhancement provides more reliable OCR results than existing normalization methods.

**Experiment Context**: 20251217_024343_image_enhancements_implementation
**Related Artifacts**: experiment-tracker/experiments/.../scripts/sepia_*.py

## Problem Statement

Current preprocessing methods (gray-scale conversion and gray-world normalization) do not consistently produce reliable OCR predictions. User testing has identified that:

1. **Gray-scale conversion** eliminates color information that may be useful for OCR
2. **Gray-world normalization** achieves 75% tint reduction but still produces zero predictions on problematic images
3. **Combined approach** (normalization + gray-scale) provides inconsistent results

**Key Issue**: Need a more reliable enhancement method that maintains consistent OCR performance across varying document conditions (background tints, lighting variations, contrast issues).

**Reference Cases**:
- Problematic: `drp.en_ko.in_house.selectstar_000732` (poor OCR results)
- Target: `drp.en_ko.in_house.selectstar_000712_sepia.jpg` (good OCR results)

## Design Goals

1. **Reliability**: Consistent OCR predictions across document variations
2. **Simplicity**: Maintain simple pipeline (perspective correction + enhancement)
3. **Performance**: Processing time < 100ms per image
4. **Quality**: VLM validation score > 4.5/5
5. **Superiority**: Outperform gray-scale and normalization methods

## Architecture

### High-Level Architecture

```
Input Image
    │
    ├─> Perspective Correction
    │       │
    │       └─> Document boundary detection (contour-based)
    │       └─> Quadrilateral approximation
    │       └─> Perspective transformation
    │
    ├─> Sepia Enhancement (Method Selection)
    │       │
    │       ├─> Classic Sepia (traditional matrix)
    │       ├─> Adaptive Sepia (intensity-based)
    │       ├─> Warm Sepia (OCR-optimized) ⭐
    │       └─> Contrast Sepia (CLAHE-enhanced)
    │
    ├─> [Optional] Deskewing
    │       └─> Hough lines text alignment
    │
    └─> Enhanced Output → OCR Model
```

**Pipeline Philosophy**: Minimize processing stages while maximizing OCR reliability.

### Components

#### SepiaEnhancer Class
- **Purpose**: Apply sepia tone transformations to document images
- **Responsibilities**:
  - Implement 4 sepia methods with distinct characteristics
  - Calculate enhancement metrics (tint, contrast, brightness, edge strength)
  - Process single images or batches
  - Track processing time
- **Interfaces**:
  - Input: NumPy array (BGR image)
  - Output: Dict[method_name: (enhanced_image, metrics)]
  - Methods: `enhance(img) -> results`

#### ImageEnhancementComparator Class
- **Purpose**: Compare sepia methods against alternatives
- **Responsibilities**:
  - Execute 7 enhancement methods (raw, grayscale, gray-world, 4x sepia)
  - Generate comparison grids and visualizations
  - Calculate comparative metrics
  - Export results (images, JSON, tables)
- **Interfaces**:
  - Input: NumPy array (BGR image)
  - Output: Comparison grid, metrics, individual images
  - Methods: `compare_all(img) -> results`

#### SepiaPerspectivePipeline Class
- **Purpose**: Integrate sepia with perspective correction
- **Responsibilities**:
  - Stage 1: Perspective correction
  - Stage 2: Sepia enhancement (configurable method)
  - Stage 3: Optional deskewing
  - Pipeline metrics and timing
- **Interfaces**:
  - Input: Raw document image
  - Output: Fully processed image ready for OCR
  - Configuration: sepia_method, enable_deskewing
  - Methods: `process(img) -> staged_results`

#### VLM Validator (Bash Script)
- **Purpose**: Visual quality assessment via Dashscope API
- **Responsibilities**:
  - Submit comparison grids to VLM (Qwen3 VL Plus)
  - Evaluate text clarity, background quality, tint impact, OCR suitability
  - Generate structured JSON assessments
  - Rank methods and provide recommendations
- **Interfaces**:
  - Input: Comparison grid images
  - Output: JSON validation reports
  - API: Dashscope multimodal generation endpoint

## Design Decisions

### Decision 1: Four Sepia Methods Instead of One
- **Context**: Different document types may benefit from different sepia characteristics
- **Options Considered**:
  1. Single classic sepia method
  2. Multiple sepia variations with different properties
  3. Adaptive method that auto-selects approach
- **Decision**: Implement 4 distinct methods (classic, adaptive, warm, contrast)
- **Rationale**:
  - Allows empirical testing to determine best method
  - Provides flexibility for different document types
  - Warm sepia theoretically optimal for OCR (enhanced red/yellow, reduced blue)
  - Contrast variant useful for low-contrast documents
- **Consequences**:
  - More code to maintain
  - Longer testing phase to evaluate all methods
  - Better documentation of method trade-offs

### Decision 2: Sepia vs Advanced Color Correction
- **Context**: Could implement sophisticated color correction (white balance, color constancy algorithms)
- **Options Considered**:
  1. Simple sepia transformation
  2. Advanced illumination-invariant color spaces
  3. Machine learning-based enhancement
- **Decision**: Use sepia transformation with matrix operations
- **Rationale**:
  - Extremely fast processing (~50ms)
  - Deterministic and reproducible results
  - User observations show effectiveness
  - Avoids complexity of ML models or advanced algorithms
- **Consequences**:
  - May not be optimal for all document types
  - Fixed transformation (not adaptive to image content)
  - Easy to understand and debug

### Decision 3: Integration with Existing Pipeline
- **Context**: Experiment already has perspective correction and deskewing
- **Options Considered**:
  1. Replace gray-world normalization completely
  2. Add sepia as alternative option
  3. Chain sepia after normalization
- **Decision**: Add sepia as alternative to normalization, keep both available
- **Rationale**:
  - Allows A/B testing against existing method
  - Preserves fallback if sepia underperforms
  - User can select method based on document type
  - Maintains experimental rigor
- **Consequences**:
  - Two parallel enhancement paths in code
  - Configuration complexity increases
  - Need clear documentation of when to use each

### Decision 4: VLM Validation Strategy
- **Context**: Need visual quality assessment beyond quantitative metrics
- **Options Considered**:
  1. Quantitative metrics only (tint, contrast, etc.)
  2. Human visual inspection
  3. VLM-based quality scoring
- **Decision**: Use VLM (Qwen3 VL Plus) for visual validation
- **Rationale**:
  - Previous experiment showed 96-100% correlation with quantitative metrics
  - Provides human-like perception of quality
  - Can assess text legibility and OCR suitability directly
  - Automated and reproducible
  - Already integrated in experiment infrastructure
- **Consequences**:
  - Requires API access and costs
  - Slower than quantitative-only approach (~30-45s per image)
  - JSON parsing complexity for structured results

### Decision 5: Optional Deskewing Stage
- **Context**: Week 2 deskewing showed no OCR improvement in user testing
- **Options Considered**:
  1. Always include deskewing
  2. Exclude deskewing completely
  3. Make deskewing optional
- **Decision**: Deskewing is optional, disabled by default
- **Rationale**:
  - User testing showed no OCR benefit
  - Sepia might interact differently with skewed text
  - Preserves ability to test combination if needed
  - Reduces default processing time
- **Consequences**:
  - Simpler default pipeline (2 stages instead of 3)
  - Need to document when to enable deskewing
  - Potential missed optimization if combination beneficial

## Implementation Details

### Sepia Transformation Matrices

**Classic Sepia (Traditional)**:
```
R' = 0.393*R + 0.769*G + 0.189*B
G' = 0.349*R + 0.686*G + 0.168*B
B' = 0.272*R + 0.534*G + 0.131*B
```

**Warm Sepia (OCR-Optimized)**:
```
R' = 0.450*R + 0.850*G + 0.200*B  (strong boost)
G' = 0.350*R + 0.750*G + 0.150*B  (boosted)
B' = 0.200*R + 0.450*G + 0.100*B  (reduced)
```

Key difference: Warm sepia amplifies red/yellow channels more aggressively while further reducing blue channel to eliminate cold tints.

### Metrics Calculation

For each enhancement method, calculate:

1. **Color Tint Score**: `std_dev(channel_means)` - Lower is better, target < 20
2. **Background Variance**: `std_dev(all_pixels)` - Spatial uniformity
3. **Contrast**: `std_dev(grayscale_values)` - Higher indicates better text definition
4. **Brightness**: `mean(grayscale_values)` - Target 150-180 range
5. **Edge Strength**: `variance(Laplacian)` - Proxy for text clarity

### Performance Targets

- **Processing Time**: < 100ms per image (perspective + sepia)
- **Throughput**: > 10 images/second on standard hardware
- **Memory**: < 500MB for batch processing
- **Accuracy**: Maintain 97% hmean on test dataset (epoch-18 checkpoint)

## Testing Strategy

### Phase 1: Isolated Testing
- Test all 4 sepia methods on reference samples
- Generate metrics for each method
- Visual inspection of outputs

### Phase 2: Comparative Analysis
- Compare sepia methods vs gray-scale and gray-world normalization
- Generate comparison grids
- Quantitative metric analysis

### Phase 3: Pipeline Validation
- Test full sepia + perspective correction pipeline
- Compare against current pipeline (perspective + normalization)
- Optional deskewing evaluation

### Phase 4: VLM Validation
- Submit comparison grids to Qwen3 VL Plus
- Evaluate text clarity, background quality, tint impact, OCR suitability
- Generate method rankings and recommendations

### Phase 5: OCR End-to-End
- Run OCR inference with epoch-18 checkpoint
- Compare prediction accuracy vs baseline
- Test on problematic samples and full test set

## Success Criteria

Sepia enhancement is considered successful if:

1. **OCR Accuracy**: Predictions improve on problematic samples
2. **Reliability**: Consistent results across document variations
3. **Metrics**: Color tint < 20, maintained or improved contrast
4. **VLM Score**: > 4.5/5 visual quality validation
5. **Performance**: Processing time < 100ms per image
6. **Comparison**: Outperforms gray-scale and gray-world normalization

## Risks and Mitigation

### Risk 1: Sepia May Not Improve OCR
- **Mitigation**: Keep gray-world normalization as fallback
- **Fallback**: Document findings and continue with existing pipeline

### Risk 2: Processing Time Exceeds Target
- **Mitigation**: Optimize matrix operations, use NumPy vectorization
- **Fallback**: Accept slightly higher latency if accuracy improves

### Risk 3: Method Selection Unclear
- **Mitigation**: Comprehensive testing of all 4 methods
- **Fallback**: Default to warm sepia based on theoretical optimality

### Risk 4: VLM Validation Inconclusive
- **Mitigation**: Rely on quantitative metrics and OCR accuracy
- **Fallback**: Previous experiments showed high correlation

## Future Considerations

1. **Adaptive Method Selection**: Auto-select sepia method based on image analysis
2. **Hybrid Approaches**: Combine sepia with selective normalization
3. **ML-Based Enhancement**: Train model to predict optimal enhancement
4. **Real-time Configuration**: Allow OCR pipeline to switch methods dynamically

## References

- Experiment: `20251217_024343_image_enhancements_implementation`
- Scripts: `experiment-tracker/.../scripts/sepia_*.py`
- Documentation: `experiment-tracker/.../docs/SEPIA_TESTING_GUIDE.md`
- State: `experiment-tracker/.../state.yml`
- **Decision**: What was chosen
- **Rationale**: Why this option was selected
- **Consequences**: Implications of this choice

## Implementation Considerations

### Technical Requirements
- Requirement 1
- Requirement 2

### Dependencies
- Dependency 1
- Dependency 2

### Constraints
- Constraint 1
- Constraint 2

## Testing Strategy

### Unit Testing
- Test approach for individual components

### Integration Testing
- Test approach for component interactions

### End-to-End Testing
- Test approach for complete workflows

## Deployment

### Deployment Strategy
- How this will be deployed

### Rollback Plan
- How to rollback if issues occur

## Monitoring & Observability

### Metrics
- Key metrics to monitor

### Logging
- Logging strategy

### Alerting
- Alert conditions and thresholds

## Future Considerations

### Scalability
- How this design will scale

### Extensibility
- How this design can be extended

### Maintenance
- Maintenance considerations

---

*This design document follows the project's standardized format for architectural documentation.*
