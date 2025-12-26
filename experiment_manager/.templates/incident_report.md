---
title: "[Short Title, e.g., Perspective Overshoot]"
date: "YYYY-MM-DD HH:MM (KST)"
experiment_id: "YYYYMMDD_HHMM_experiment_type"
severity: "medium"
status: "open"
tags: []
author: "AI Agent"
---

## Defect Analysis: [Short Title, e.g., Perspective Overshoot]

### 1. Visual Artifacts (What does the output look like?)

* **Distortion Type:** [e.g., Shearing, Pincushion, Stretching, Blank Output]

* **Key Features:** [e.g., Text is diagonal, pixel smearing, ROI cropped out]

* **Comparison:** [e.g., Worse than baseline, or regression from previous version]

### 2. Input Characteristics (What is unique about the source?)

* **ROI Coverage:** [e.g., Subject fills 90% of frame]

* **Contrast/Lighting:** [e.g., Low contrast between paper and table]

* **Geometry:** [e.g., Image is already cropped/rectified]

### 3. Geometric/Data Analysis (The Math)

* **Mask Topology:** [e.g., Mask touches all 4 image borders]

* **Corner Detection:** [e.g., Detected points are collinear]

* **Transform Matrix:** [e.g., Matrix appears singular or ill-conditioned]

### 4. Hypothesis & Action Items

* **Theory:** [Why did the logic fail?]

* **Proposed Fix:** [e.g., Add threshold, adjust epsilon, clamp coordinates]

---

## Related Resources

### Related Artifacts

* (No related artifacts)

### Related Assessments

* (No related assessments)

