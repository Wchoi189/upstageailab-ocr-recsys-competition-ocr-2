# BUG-001: OCR Overlay Misalignment – Technical Analysis

You are a meticulous computer vision engineer debugging an OCR visualization issue.

**Input**: One image containing:
- A receipt (white paper) on a dark background
- Green rectangular overlays marking OCR text regions
- Yellow text labels showing recognized text and confidence scores

**Objective**: Analyze the geometric relationship between overlays and actual text to identify coordinate frame misalignment issues.

---

## 1. Visual Overview

Provide a concise description covering:
- **Layout**: Receipt positioning within the image frame
- **Overlay behavior**: How green boxes relate to printed text (aligned, systematically shifted, scaled incorrectly, etc.)
- **Overall pattern**: Any immediately obvious systematic issues

## 2. Quantitative Alignment Analysis

Select **3-4 representative text lines** (top, middle, bottom of receipt) and for each:

**Text vs. Box Positioning:**
- Where does the actual text line start/end horizontally?
- Where does the corresponding green box start/end?
- Calculate approximate pixel offset: box_start - text_start, box_end - text_end

**Pattern Recognition:**
- Is there a consistent horizontal shift across all lines? (e.g., "+15px left, +20px right")
- Do boxes maintain correct width relative to text, or are they systematically too wide/narrow?
- Any correlation between offset magnitude and vertical position?

## 3. Coordinate Frame Analysis

**Image Dimensions** (estimate in pixels):
- Full image: W × H
- Receipt content area: left_edge to right_edge, approximate width
- Padding: left_padding, right_padding

**Frame Hypothesis Testing:**
Compare your measured offsets against:
- Left padding value
- Right padding value
- Receipt width ratios

**Key Question**: Do the green boxes appear positioned as if they were:
- A) Calculated for the content area but rendered in the full padded frame?
- B) Calculated for the full frame but interpreted as content-only coordinates?
- C) Correctly positioned but with an additional systematic offset?

## 4. Root Cause Assessment

**Primary Misalignment Pattern:**
- Consistent offset direction and magnitude
- Whether offset correlates with padding dimensions

**Coordinate System Diagnosis:**
- Most likely explanation for the observed pattern
- Whether this suggests a scaling issue, translation issue, or coordinate frame confusion

## 5. Technical Recommendations

**Immediate Verification Steps:**
1. **Coordinate bounds check**: Compare min/max X coordinates of overlay polygons to content area boundaries
2. **Scaling factor verification**: Check if overlay coordinates need to be scaled by (content_width / padded_width)

**Expected Fix Category:**
- Coordinate frame translation
- Scaling factor correction
- Padding compensation
- Other (specify)

**Confidence Level**: Rate your diagnosis confidence (High/Medium/Low) and note any ambiguities requiring additional data.
