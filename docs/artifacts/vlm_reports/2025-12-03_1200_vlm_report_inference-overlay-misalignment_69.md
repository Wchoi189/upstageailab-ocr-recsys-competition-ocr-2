---
ads_version: "1.0"
title: "2025 12 03 Bug 001 Inference Overlay Misalignment 69"
date: "2025-12-06 18:08 (KST)"
type: "vlm_report"
category: "evaluation"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---





# Analysis Result

**Mode:** bug_001
**Backend:** openrouter
**Processing Time:** 77.28s

## Analysis

## BUG-001: OCR Overlay Misalignment – Technical Analysis

**1. Visual Overview**

* **Layout**: The receipt is positioned centrally within the image frame, with a significant dark border (padding) around it. The receipt appears vertically oriented.
* **Overlay behavior**: The green boxes consistently lag *behind* the actual text. The left edge of the boxes is noticeably to the right of the start of the text lines. The boxes generally appear to be of appropriate width relative to the text they are attempting to highlight.
* **Overall pattern**: A systematic horizontal shift of the green boxes to the right relative to the text is immediately apparent. This shift appears relatively consistent across the entire receipt.

**2. Quantitative Alignment Analysis**

Let's analyze three representative lines:

* **Line 1 (Top):** "상호 (주)신화"
    * Text Start: ~60px, Text End: ~200px
    * Box Start: ~85px, Box End: ~225px
    * Offset: +25px (box_start - text_start), +25px (box_end - text_end)
* **Line 2 (Middle):** "판매금액"
    * Text Start: ~350px, Text End: ~450px
    * Box Start: ~375px, Box End: ~475px
    * Offset: +25px (box_start - text_start), +25px (box_end - text_end)
* **Line 3 (Bottom):** "서비스받으신"
    * Text Start: ~600px, Text End: ~750px
    * Box Start: ~625px, Box End: ~775px
    * Offset: +25px (box_start - text_start), +25px (box_end - text_end)

**Pattern Recognition:**

* **Consistent Horizontal Shift**: There is a consistent horizontal shift of approximately +25 pixels to the right across all three lines.
* **Box Width**: The boxes maintain a correct width relative to the text. They are not systematically too wide or narrow.
* **Vertical Correlation**: No apparent correlation between offset magnitude and vertical position. The offset remains consistent throughout the receipt.

**3. Coordinate Frame Analysis**

**Image Dimensions (estimated):**

* Full image: ~1000px x ~1400px (W x H)
* Receipt content area: left_edge ~50px, right_edge ~950px, approximate width ~900px
* Padding: left_padding ~50px, right_padding ~50px

**Frame Hypothesis Testing:**

The consistent +25px offset strongly suggests that the green boxes are calculated based on the receipt content area but are being rendered within the full padded image frame.  The offset is approximately half the left padding (50px / 2 = 25px).

* **A) Calculated for the content area but rendered in the full padded frame?** – **Highly Likely.** This aligns with the observed +25px offset.
* **B) Calculated for the full frame but interpreted as content-only coordinates?** – Unlikely. This would result in a larger offset, closer to the full left padding.
* **C) Correctly positioned but with an additional systematic offset?** – Less likely, given the consistency and relationship to the padding.

**4. Root Cause Assessment**

**Primary Misalignment Pattern:** Consistent offset direction and magnitude (+25px right).

**Coordinate System Diagnosis:** The most likely explanation is a coordinate frame mismatch. The OCR engine is likely calculating bounding box coordinates relative to the receipt content area, but these coordinates are being used directly to render the boxes in the full image frame, without accounting for the left padding.

This is a translation issue – the bounding box coordinates need to be translated to the left by the amount of the left padding to align with the full image frame.

**5. Technical Recommendations**

**Immediate Verification Steps:**

1. **Coordinate bounds check**: Verify that the minimum X coordinate of the overlay polygons is greater than or equal to the left edge of the receipt content area (50px).
2. **Scaling factor verification**: Scaling is not the primary issue here, but it's good to confirm that the scaling factor (if any) is correctly applied before the translation.

**Expected Fix Category:** Coordinate frame translation.  Specifically, subtract the left padding value from the X coordinate of each bounding box before rendering.

**Confidence Level**: **High**. The consistent offset and its relationship to the padding strongly suggest a coordinate frame issue. However, further investigation of the OCR engine's coordinate system and rendering pipeline is recommended to confirm.  Ambiguity remains regarding the exact implementation details of the OCR engine.
