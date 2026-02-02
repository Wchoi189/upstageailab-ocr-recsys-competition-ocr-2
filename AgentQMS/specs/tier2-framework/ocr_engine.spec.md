# OCR Engine Specification

**Tier**: 2 (Framework)
**Scope**: OCR Pipeline, Models, and Processing Logic.

## 1. Pipeline Contracts

**Core Interface**: `BasePipeline`
*   **Method**: `process(images: T) -> Result`
*   **State**: Pipelines must be stateless.
*   **Config**: All hyperparameters injected via `conf`.

### Stages
| Stage | Input | Output | Responsibility |
| :--- | :--- | :--- | :--- |
| **Preprocessing** | `np.ndarray` | `np.ndarray` | Resize, Norm, Pad |
| **Inference** | `Tensor` | `Tensor` | Model Forward Pass |
| **Postprocessing** | `Tensor` | `List[Box]` | Decode, NMS, Format |

## 2. Model Management

**Class**: `ModelManager`
*   **Loading**: Lazy loading restricted to `setup()`.
*   **Weights**: Must use `torch.load(..., map_location='cpu')` initially.
*   **Device**: Strict `device` arg passing (no `.cuda()` calls hardcoded).

### Artifacts
*   **ONNX**: Supported for exported models.
*   **TorchScript**: Legacy support only.
