---
ads_version: '2.0'
id: 'FW-028'
type: 'component_interface'
tier: 2
priority: 'high'
updated: '2026-02-03'
description: 'Prepares raw images for inference.'
---

# Preprocessing Logic

> Prepares raw images for inference.

## Specification

```yaml
agent: all
last_updated: '2026-01-20'
component: preprocessing_logic
role: 'Applies resolution standardization, padding, normalization, and optional geometric
  corrections.

  '
critical_logic:
- id: pipeline-stages
  description: "1. Decode: Convert Base64/Bytes to BGR Numpy Array (Opencv format).\n\
    2. Perspective Correction (Optional):\n   - If 'enable_perspective_correction=True'.\n\
    \   - Uses 'rembg' or corner detection logic to find document corners.\n   - Warps\
    \ image to 'flat' view.\n   - Note: This changes the 'original_size' effective\
    \ for downstream coordinate mapping.\n3. Resize & Pad:\n   - Resizes to 'target_size'\
    \ (default 640) preserving aspect ratio.\n   - Pads right/bottom to reach exact\
    \ 640x640 square.\n4. Normalize:\n   - Standard ImageNet mean/std (if generic\
    \ model) or custom stats.\n   - Converts to Float32 Tensor (1, C, H, W).\n"
- id: configuration
  description: 'Configuration is loaded from Hydra V5.0 domain configs (domain/detection.yaml,
    etc.) or can be overridden per request.

    - target_size: int (default 640)

    - transform_pipeline: List[str] (names of albumentations transforms)

    See: AgentQMS/standards/tier2-framework/hydra-v5-rules.yaml

    '
- id: display-modes
  description: 'Corrected: The preview image shows the ''After Warp'' view.

    Original: The preview image shows the raw input, requiring inverse-inverse mapping
    (complex). Currently defaults to Corrected view for simplicity.

    '
data_contract:
  input:
  - field: image
    type: np.ndarray
  - field: settings
    type: PreprocessSettings
  output:
  - field: result
    type: PreprocessingResult
    description: Tensor + Metadata + OriginalImage

```