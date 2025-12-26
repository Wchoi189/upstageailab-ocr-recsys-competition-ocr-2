# Visual Evidence Draft: [Experiment Name, e.g., Dominant Edge Extension]

## Cluster A: The "Lever Arm" Instability

**Frequency:** ~17/25 cases
**Description:** Cases where the "Dominant Edge" extension caused lines to fly off into the background.
**Selection Criteria:** High deviation from mask, low support score (<0.50).

| Sample ID | Visual Evidence | Observations |
| :--- | :--- | :--- |
| `selectstar_000138` | ![000138](path/to/vis_000138.jpg) | **Visual:** The bottom line is angled 15° too sharp.<br>**Metric:** Edge Support = 0.32.<br>**Analysis:** Short hull segment caused extension drift. |
| `selectstar_000216` | ![000216](path/to/vis_000216.jpg) | **Visual:** Top corners are "virtual" and exist in dark background.<br>**Metric:** Linearity RMSE = 8.4.<br>**Analysis:** Perspective foreshortening was misinterpreted. |

## Cluster B: The "Topology Trap"

**Frequency:** ~8/25 cases
**Description:** Cases where filtering left < 4 segments or > 4 segments.

| Sample ID | Visual Evidence | Observations |
| :--- | :--- | :--- |
| `selectstar_000006` | ![000006](path/to/vis_000006.jpg) | **Visual:** Mask is triangular/blobby.<br>**Reason:** `dominant_extension_failed_using_bbox`<br>**Analysis:** Hull simplification resulted in only 3 valid segments. |

---

**Hypothesis for Agent:** Algorithm fails when local segment noise is extrapolated globally.


