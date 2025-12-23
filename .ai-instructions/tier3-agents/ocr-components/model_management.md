# Component: Model Management (ModelManager)

## Role
Manages the singleton instance of the AI model to prevent expensive reload times. Handles checkpoint discovery and auto-configuration.

## Critical Logic

### 1. Caching Strategy
- **Key**: `checkpoint_path` (absolute path).
- **Behavior**: If `load_model(A)` is called and model A is already loaded, it returns immediately (0ms latency). If B is requested, A is unloaded, GPU cache cleared, and B is loaded.
- **Reason**: Serverless-like behavior (cold start vs warm start).

### 2. Configuration Inference
- Models require 4 components: `Encoder`, `Decoder`, `Head`, `Transforms`.
- These are defined in a YAML config.
- **Auto-Discovery**: If `checkpoint.pth` is at `/foo/bar/checkpoints/epoch=1.ckpt`, manager looks for `.hydra/config.yaml` or `config.yaml` in parent directories.

### 3. Resource Management
- **Cleanup**: Calls `torch.cuda.empty_cache()` on model swap to prevent OutOfMemory (OOM) errors on 8GB GPUs.
- **Device**: Defaults to `cuda` if available, else `cpu`.

## Data Contract
**Input**: `checkpoint_path`
**Output**: `LightingModule` (loaded on device)
