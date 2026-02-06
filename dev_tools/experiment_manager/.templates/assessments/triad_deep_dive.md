# Deep Dive Diagnostic: [Image ID]

## 1. The Raw Input

| Original Image | Binary Mask |
| :---: | :---: |
| ![Original](path/to/original.jpg) | ![Mask](path/to/mask.jpg) |

**Input Characteristics:**

* **Shape:** [e.g., 5-sided polygon with folded corner]
* **Noise:** [e.g., Clean edges, but significant curvature on left side]

## 2. The Algorithm Output

![Result](path/to/result_vis.jpg)

**Telemetry:**

* **Epsilon Used:** [e.g., 10px]
* **Reason Code:** `line_quality_fallback`
* **Failed Metric:** Edge Support (0.42)

## 3. The Discrepancy Analysis

* **Expected Behavior:** [Describe the intended geometric relationship.]
* **Actual Behavior:** [Describe the observed drift or missing structure.]
* **Actionable Insight:** [Define a concrete change to test next.]


