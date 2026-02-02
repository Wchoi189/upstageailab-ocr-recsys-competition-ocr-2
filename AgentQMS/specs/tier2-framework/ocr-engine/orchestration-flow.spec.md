---
ads_version: '2.0'
id: 'FW-024'
type: 'component_interface'
tier: 2
priority: 'high'
updated: '2026-02-03'
description: 'The central controller for the OCR inference pipeline.'
---

# Orchestration Flow

> The central controller for the OCR inference pipeline.

## Specification

```yaml
agent: all
last_updated: '2026-01-20'
component: orchestration_flow
role: 'The central ''brain'' that wires together the individual components. It ensures
  the data flows correctly from ImageLoader -> Preprocessing -> Model -> Postprocessing
  -> Preview.

  '
critical_logic:
- id: request-parsing
  description: Receives 'InferenceRequest' object.
- id: resource-check
  description: Verifies ModelManager has the correct checkpoint loaded. If not, triggers
    load (blocking).
- id: pipeline-execution
  description: '1. Load and Fix Orientation (ImageLoader)

    2. Preprocess (Resize/Pad) (PreprocessingPipeline)

    3. Inference (ModelManager)

    4. Postprocess (Decode Polygons) (PostprocessingPipeline)

    5. Preview (Optional Visualization) (PreviewGenerator)

    '
- id: error-handling
  description: Catches internal decoding errors and wraps them in 'OCRBackendError'.
data_contract:
  input:
  - field: InferenceRequest
    type: object
  output:
  - field: InferenceResponse
    type: Dict

```