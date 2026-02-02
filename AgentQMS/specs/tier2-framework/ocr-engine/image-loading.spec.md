---
ads_version: '2.0'
id: 'FW-020'
type: 'component_interface'
tier: 2
priority: 'high'
updated: '2026-02-03'
description: 'Handles loading and validation of input images.'
---

# Image Loading Standards

> Handles loading and validation of input images.

## Specification

```yaml
agent: all
last_updated: '2026-01-20'
component: image_loading
role: 'The entry point for raw data. Responsible for reading image bytes, validating
  formats, and correcting orientation (EXIF).

  '
critical_logic:
- id: format-support
  description: 'Must support: JPG, PNG, BMP, TIFF, WebP.

    Must reject: PDF (handled by separate PDF pipeline), GIF (unless first frame).

    '
- id: safe-loading
  description: 'Use ''Pillow'' or ''OpenCV'' with strict limits on image dimensions
    (Decompression Bomb protection).

    Max dims default: 10000x10000.

    '
- id: exif-correction
  description: 'Must apply EXIF orientation tag (e.g., rotate 90 deg for phone photos)
    BEFORE any processing.

    '
- id: channel-standardization
  description: 'Convert all inputs to BGR (OpenCV standard) or RGB (Pillow standard).

    Current Standard: BGR (np.array).

    '
data_contract:
  input:
  - field: image_source
    type: Union[str, bytes, Path]
    description: File path, url, or raw bytes
  output:
  - field: image_array
    type: np.ndarray
    description: BGR image array (H, W, 3)

```