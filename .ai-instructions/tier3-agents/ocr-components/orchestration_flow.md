# Component: Orchestration Flow (InferenceOrchestrator)

## Role
The central "brain" that wires together the individual components. It ensures the data flows correctly from ImageLoader -> Preprocessing -> Model -> Postprocessing -> Preview.

## Critical Logic
1.  **Request Parsing**: Receives `InferenceRequest` object.
2.  **Resource Check**: Verifies ModelManager has the correct checkpoint loaded. If not, triggers load (blocking).
3.  **Pipeline Execution**:
    ```python
    # 1. Load and Fix Orientation
    img = ImageLoader.load(request.image)

    # 2. Preprocess (Resize/Pad)
    tensor, meta = PreprocessingPipeline.process(img)

    # 3. Inference
    raw_output = ModelManager.forward(tensor)

    # 4. Postprocess (Decode Polygons)
    regions = PostprocessingPipeline.process(raw_output, meta)

    # 5. Preview (Optional Visualization)
    if request.preview:
        b64 = PreviewGenerator.draw(img, regions)
    ```
4.  **Error Handling**: Catches internal decoding errors and wraps them in `OCRBackendError`.

## Data Contract
**Input**: `InferenceRequest`
**Output**: `InferenceResponse` (Dict format)
