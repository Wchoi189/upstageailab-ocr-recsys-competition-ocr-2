---
ads_version: "1.0"
title: "2025 12 03 Bug 001 Inference Overlay Misalignment Gemma1"
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
**Processing Time:** 73.35s

## Analysis

## BUG-001: OCR Overlay Misalignment – Technical Analysis

Here's a detailed analysis of the OCR overlay misalignment issue based on the provided image.

## 1. High-Level Description

The image shows a receipt (white paper) placed on a dark background. Green rectangular overlays are drawn on top of the receipt, attempting to highlight the regions of recognized text.  The overlays contain yellow text displaying the OCR result and confidence score.  The overlays are consistently shifted to the right of the actual text, and appear slightly too wide. The misalignment is noticeable across the entire receipt.

## 2. Horizontal Alignment Analysis

Let's analyze three representative rows: top, middle, and bottom.

**Row 1 (Top - "U PLEX"):**

*   **Text Line:** Appears to start around pixel 60 from the left image edge.
*   **Green Box:** Starts around pixel 85, ends around pixel 130.
*   **Offset:** Shifted right by approximately 25 pixels.  The box is wider than the text.

**Row 2 (Middle - "카드결제"):**

*   **Text Line:** Appears to start around pixel 60 from the left image edge.
*   **Green Box:** Starts around pixel 85, ends around pixel 130.
*   **Offset:** Shifted right by approximately 25 pixels. The box is wider than the text.

**Row 3 (Bottom - "현금결제"):**

*   **Text Line:** Appears to start around pixel 60 from the left image edge.
*   **Green Box:** Starts around pixel 85, ends around pixel 130.
*   **Offset:** Shifted right by approximately 25 pixels. The box is wider than the text.

**Overall Observations:**

*   There is a **consistent horizontal offset** of approximately 25 pixels to the right across most lines.
*   The offset does *not* roughly match any obvious padding. The left padding is significantly larger than 25 pixels (see section 3).

## 3. Content Box and Padding Estimation

1.  **Full Image Dimensions:** Approximately 640 x 480 pixels.
2.  **Receipt Content Dimensions:**
    *   Left Edge: ~60 pixels from the image border.
    *   Right Edge: ~570 pixels from the image border.
    *   Approximate Width: ~510 pixels.
3.  **Padding:**
    *   Left Padding: ~60 pixels.
    *   Right Padding: ~70 pixels.

**Coordinate Frame:**

The green boxes appear to be positioned in the coordinate frame of the **entire image** (including background).  The coordinates are not relative to the receipt content area.

## 4. Relationship Between Boxes and Padding

The typical horizontal offset (25 pixels) is *not* similar to either the left padding (60 pixels) or the right padding (70 pixels). It's also not a significant fraction of the receipt width (510 pixels).

This suggests that **coordinates were scaled only for the content area but drawn as if they referred to the full padded frame**. The OCR engine likely calculated bounding box coordinates relative to the receipt content, but these coordinates were then directly used to draw the boxes on the full image canvas without accounting for the offset caused by the padding.

## 5. Final Technical Hypothesis

**Observed Misalignments**

*   Consistent rightward shift of green boxes by approximately 25 pixels.
*   Green boxes are consistently wider than the actual text they enclose.
*   Misalignment is present across the entire receipt, from top to bottom.

**Estimated Content Box & Padding**

*   Image: 640 x 480 pixels
*   Receipt Content: Left ~60, Right ~570, Width ~510 pixels
*   Left Padding: ~60 pixels
*   Right Padding: ~70 pixels

**Likely Coordinate Frame**

(A) defined in the resized content box but drawn in the padded frame.

**Recommended Next Check**

The developer should check if the OCR bounding box coordinates are being translated by the left padding value before being used to draw the green rectangles. Specifically, compare the minimum X coordinate of each polygon (green box) to the left edge of the receipt content (60 pixels). If the minimum X values are close to 60, it confirms that the coordinates are being interpreted as absolute image coordinates instead of relative content coordinates.  Another check would be to subtract the left padding from the X coordinates before drawing the boxes.
