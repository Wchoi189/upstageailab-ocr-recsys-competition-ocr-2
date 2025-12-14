# Session Handover: Perspective Correction Algorithm Evaluation & Refinement

**Date**: 2025-11-22
**Session Type**: Evaluation & Implementation
**Status**: ✅ Testing Complete, ⚠️ Algorithm Refinement Needed
**Next Session**: Algorithm Refinement & ScanTailor Integration

---

## Executive Summary

Comprehensive evaluation of perspective correction algorithms on rembg-processed images has been completed. Key findings:

- **60% success rate** for both DocTR and regular methods
- **40% failure rate** with identical failure patterns (corner detection issue)
- **Fallback mechanism implemented** and working correctly
- **DocTR is 1.66x faster** on valid cases
- **ScanTailor Advanced installed** (GUI version, CLI integration pending)

---

## Completed Work

### 1. Extended Testing Framework ✅

**Script**: `scripts/test_perspective_comprehensive.py`

- Tests both DocTR and regular perspective correction methods
- Validates results with multiple metrics (area retention, dimension ratios, content preservation)
- Implements automatic fallback to rembg version when both methods fail
- Generates detailed JSON results for analysis
- Creates side-by-side comparison images

**Test Results (10 samples)**:
- Regular method valid: 6/10 (60.0%)
- DocTR method valid: 6/10 (60.0%)
- Fallback used: 4/10 (40.0%)
- Average regular time: 0.007s
- Average DocTR time: 0.004s (1.66x faster)

**Output**: `outputs/perspective_comprehensive/results.json` (22KB)

### 2. Fallback Mechanism ✅

**Implementation**: Automatic fallback to rembg-processed image when both methods fail validation

**Validation Criteria**:
- Area retention >50% (output area / input area)
- Dimension ratios >50% (width and height)
- Content preservation >10% (non-zero pixels)

**Status**: Working correctly, prevents bad outputs

### 3. Failure Analysis ✅

**Root Cause Identified**: Corner detection finds small regions (text blocks, artifacts) instead of full document boundaries

**Common Failure Pattern**: "Area loss too large" - Both methods produce identical invalid results (17-45% area loss)

**Failed Images**:
1. `drp.en_ko.in_house.selectstar_000006.jpg` - 35.4% area loss
2. `drp.en_ko.in_house.selectstar_000008.jpg` - 17.2% area loss (catastrophic)
3. `drp.en_ko.in_house.selectstar_000011.jpg` - 45.2% area loss
4. `drp.en_ko.in_house.selectstar_000015.jpg` - 42.1% area loss

**Key Finding**: Both methods fail on the same images with identical failure reasons, suggesting the issue is in shared corner detection logic, not method-specific.

### 4. Supporting Scripts Created ✅

1. **`scripts/test_perspective_comprehensive.py`**
   - Main evaluation script with fallback mechanism
   - Tests both methods, validates results, generates JSON

2. **`scripts/test_perspective_robust.py`**
   - Pre and post-correction validation
   - Skew angle detection
   - Ready for integration into main pipeline

3. **`scripts/test_perspective_doctr_rembg.py`**
   - DocTR vs regular comparison
   - Performance metrics

4. **`scripts/test_perspective_on_rembg.py`**
   - Basic perspective correction test
   - Metrics calculation

5. **`scripts/test_scantailor_integration.py`**
   - ScanTailor integration wrapper (experimental)
   - **Note**: Currently expects CLI, but ScanTailor Advanced is GUI-only

### 5. Documentation Created ✅

1. **`docs/assessments/perspective_correction_evaluation_summary.md`**
   - Complete evaluation summary
   - Implementation plan with phases
   - Success criteria

2. **`docs/assessments/perspective_correction_failures_analysis.md`**
   - Root cause analysis
   - Statistical measurements
   - Alternative solutions

3. **`scripts/SCANTAILOR_INSTALL.md`**
   - Installation guide (Option 4 used - ScanTailor Advanced)

---

## Current State

### ScanTailor Advanced Installation

**Status**: ✅ Installed

**Location**: `/usr/local/bin/scantailor`

**Type**: GUI application (not CLI)

**Implications**:
- Current integration script (`test_scantailor_integration.py`) expects CLI version
- Need to adapt for GUI version or use X11 forwarding
- Alternative: Check if original ScanTailor has CLI option

**Next Steps for ScanTailor**:
1. Test GUI version with X11 forwarding or virtual display
2. Check if original ScanTailor has CLI (`scantailor-cli`)
3. Adapt integration script for GUI workflow
4. Compare performance on failed cases

### Algorithm Status

**Current Success Rate**: 60% (both methods)

**Failure Rate**: 40% (both methods fail identically)

**Root Cause**: Corner detection finds small regions instead of document boundaries

**Fallback**: Working correctly, prevents bad outputs

---

## Immediate Next Steps

### Priority 1: Algorithm Refinement (High)

#### 1.1 Pre-Correction Validation Integration
**Goal**: Prevent bad corrections before they happen

**Tasks**:
- [ ] Integrate validation from `test_perspective_robust.py` into main pipeline
- [ ] Add corner area ratio check (>30% of image)
- [ ] Add aspect ratio validation
- [ ] Add skew angle check (skip if <2°)
- [ ] Test on failed cases

**Script Reference**: `scripts/test_perspective_robust.py` (lines 50-120)

**Expected Impact**: Reduce failure rate from 40% to <20%

#### 1.2 Corner Detection Improvement
**Goal**: Fix root cause of failures

**Tasks**:
- [ ] Test DocTR text-based detection (`use_doctr_text=True`)
- [ ] Increase `min_area_ratio` from 0.1 to 0.3 in DocumentDetector
- [ ] Add adaptive thresholds based on image size
- [ ] Test on failed cases

**Code Location**: `ocr/datasets/preprocessing/detector.py`

**Expected Impact**: Improve corner detection accuracy

#### 1.3 Extended Testing
**Goal**: Validate improvements on larger dataset

**Command**:
```bash
python scripts/test_perspective_comprehensive.py \
    --input-dir data/datasets/images/train \
    --output-dir outputs/perspective_comprehensive \
    --num-samples 50 \
    --use-gpu \
    --save-json
```

**Expected Outcome**: Statistical validation of improvements

### Priority 2: ScanTailor Integration (Medium)

#### 2.1 GUI Integration
**Goal**: Test ScanTailor Advanced on failed cases

**Options**:
1. **X11 Forwarding** (if SSH with X11):
   ```bash
   export DISPLAY=:0
   scantailor --help
   ```

2. **Virtual Display** (Xvfb):
   ```bash
   sudo apt-get install xvfb
   xvfb-run -a scantailor [options]
   ```

3. **Adapt Integration Script**:
   - Modify `test_scantailor_integration.py` to use GUI workflow
   - May require interactive mode or batch processing

**Tasks**:
- [ ] Test ScanTailor Advanced on failed cases
- [ ] Compare performance vs current methods
- [ ] Document integration approach

#### 2.2 Alternative: Original ScanTailor CLI
**Goal**: Check if original ScanTailor has CLI option

**Command**:
```bash
apt-cache search scantailor
# Or check if scantailor-cli exists
which scantailor-cli
```

**If Available**: Use CLI version for easier integration

### Priority 3: Fallback Enhancement (Medium)

#### 3.1 Quality-Based Selection
**Goal**: Choose best valid result instead of just first valid

**Tasks**:
- [ ] Compare quality metrics between valid results
- [ ] Select best result based on quality scores
- [ ] Add logging for selection rationale

**Expected Impact**: Better output quality when both methods succeed

---

## Key Files & Locations

### Scripts
- `scripts/test_perspective_comprehensive.py` - Main evaluation script
- `scripts/test_perspective_robust.py` - Validation script (ready for integration)
- `scripts/test_perspective_doctr_rembg.py` - DocTR comparison
- `scripts/test_perspective_on_rembg.py` - Basic test
- `scripts/test_scantailor_integration.py` - ScanTailor wrapper (needs GUI adaptation)
- `scripts/optimized_rembg.py` - rembg optimization wrapper

### Documentation
- `docs/assessments/perspective_correction_evaluation_summary.md` - Full evaluation
- `docs/assessments/perspective_correction_failures_analysis.md` - Failure analysis
- `scripts/SCANTAILOR_INSTALL.md` - Installation guide

### Outputs
- `outputs/perspective_comprehensive/` - Comprehensive test results
  - `results.json` - Detailed JSON results (22KB)
  - `*_comparison.jpg` - Side-by-side comparisons
  - `*_final_*.jpg` - Final results (with fallback indicator)

### Code Locations
- `ocr/datasets/preprocessing/detector.py` - DocumentDetector (corner detection)
- `ocr/datasets/preprocessing/perspective.py` - PerspectiveCorrector
- `ocr/datasets/preprocessing/external.py` - DocTR integration

---

## Technical Details

### Validation Metrics

**Pre-Correction** (from `test_perspective_robust.py`):
- Corner area ratio >30%
- Aspect ratio match (within 50-200%)
- Skew angle >2° (skip if too small)

**Post-Correction** (from `test_perspective_comprehensive.py`):
- Area retention >50%
- Dimension ratios >50%
- Content preservation >10%

### Performance Metrics

**DocTR**:
- Average time: 0.004s (valid cases)
- Method: `doctr_rcrop`
- Speedup: 1.66x vs regular

**Regular**:
- Average time: 0.007s (valid cases)
- Method: `opencv`
- Detection: Edge-based contour

### Failure Patterns

**Common**: "Area loss too large" (17-45% area loss)

**Root Cause**: Corner detection finds small regions instead of document boundaries

**Both Methods**: Fail identically (shared detector issue)

---

## Testing Commands

### Comprehensive Evaluation
```bash
# Test on 50 images
python scripts/test_perspective_comprehensive.py \
    --input-dir data/datasets/images/train \
    --output-dir outputs/perspective_comprehensive \
    --num-samples 50 \
    --use-gpu \
    --save-json
```

### Robust Validation Test
```bash
# Test with validation
python scripts/test_perspective_robust.py \
    --input-dir outputs/perspective_test \
    --output-dir outputs/perspective_robust_test \
    --num-samples 10
```

### DocTR Comparison
```bash
# Compare DocTR vs regular
python scripts/test_perspective_doctr_rembg.py \
    --input-dir data/datasets/images/train \
    --output-dir outputs/doctr_rembg_test \
    --num-samples 10 \
    --use-gpu
```

### ScanTailor Test (when GUI is available)
```bash
# Test ScanTailor (requires GUI/X11)
xvfb-run -a python scripts/test_scantailor_integration.py \
    --input-dir outputs/perspective_comprehensive \
    --output-dir outputs/scantailor_test \
    --num-samples 5
```

---

## Known Issues & Limitations

1. **Corner Detection**: Finds small regions instead of full document (40% failure rate)
2. **ScanTailor Integration**: GUI-only, needs X11 or virtual display
3. **No Pre-Correction Validation**: Currently only post-correction validation
4. **Identical Failures**: Both methods fail on same images (shared detector)

---

## Success Criteria

### Immediate (Next Session)
- [ ] Pre-correction validation integrated
- [ ] Corner detection improved (test DocTR text-based)
- [ ] Extended test on 50+ images
- [ ] Failure rate reduced to <20%

### Short-term
- [ ] ScanTailor tested on failed cases
- [ ] Performance comparison documented
- [ ] Quality-based result selection implemented

### Long-term
- [ ] Production-ready pipeline with <10% failure rate
- [ ] Clear performance benchmarks
- [ ] Comprehensive documentation

---

## Questions & Decisions Needed

1. **ScanTailor Integration Approach**:
   - Use X11 forwarding?
   - Use virtual display (Xvfb)?
   - Adapt for GUI workflow?
   - Try original ScanTailor for CLI?

2. **Validation Thresholds**:
   - Current: 50% area retention - is this appropriate?
   - Should thresholds be adaptive based on image characteristics?

3. **Fallback Strategy**:
   - Current: Use rembg version when both fail
   - Should we try additional methods before fallback?
   - Should we log/notify when fallback is used?

4. **Priority**:
   - Focus on algorithm refinement first?
   - Or explore ScanTailor integration first?

---

## Resources & References

- **ScanTailor Advanced**: https://github.com/4lex4/scantailor-advanced
- **Original ScanTailor**: https://github.com/scantailor/scantailor
- **DocTR Documentation**: https://github.com/mindee/doctr
- **Evaluation Summary**: `docs/assessments/perspective_correction_evaluation_summary.md`
- **Failure Analysis**: `docs/assessments/perspective_correction_failures_analysis.md`

---

## Notes

- ScanTailor Advanced is installed but GUI-only (needs X11/virtual display)
- Current integration script expects CLI - needs adaptation
- Both perspective correction methods have identical failure patterns
- Fallback mechanism is working and prevents bad outputs
- DocTR is faster but doesn't solve the corner detection issue

---

**End of Session Handover**

