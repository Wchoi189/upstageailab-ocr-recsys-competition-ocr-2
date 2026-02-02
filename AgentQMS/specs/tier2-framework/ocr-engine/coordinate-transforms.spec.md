---
ads_version: '2.0'
id: 'FW-012'
type: 'component_interface'
tier: 2
priority: 'high'
updated: '2026-02-03'
description: 'Handles mapping between resized/padded images and original coordinates.'
---

# Coordinate Transforms

> Handles mapping between resized/padded images and original coordinates.

## Specification

```yaml
agent: all
last_updated: '2026-01-20'
component: coordinate_transforms
role: 'Ensures that bounding boxes detected on the 640x640 model input are accurately
  mapped back to the original full-resolution image, accounting for padding and resizing.

  '
critical_logic:
- id: padding-calculation
  description: 'Calculates padding (top, bottom, left, right) to maintain aspect ratio
    while fitting into target square.

    Logic: scale = target / max(h, w). new_size = older * scale. pad = (target - new)
    / 2.

    '
- id: inverse-mapping
  description: 'Applies the inverse of the affine transformation matrix used in preprocessing.

    Must handle both ''resize'' and ''pad'' operations.

    '
- id: coordinate-clamping
  description: 'Ensures transformed coordinates do not exceed original image dimensions
    (0, 0, W, H).

    '
data_contract:
  input:
  - field: polygons
    type: List[List[float]]
    description: Coordinates in processed space (640x640)
  - field: metadata
    type: InferenceMetadata
    description: Contains scale factor and padding values
  output:
  - field: original_polygons
    type: List[List[float]]
    description: Coordinates in original image space

```