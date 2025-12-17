---
type: api_reference
component: model_manager
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# ModelManager

## Purpose

Manages model lifecycle including loading, caching, device placement, and cleanup for OCR inference.

## Interface

| Method | Signature | Returns | Raises |
|--------|-----------|---------|--------|
| `__init__` | `__init__(device: str \| None = None)` | None | - |
| `load_model` | `load_model(checkpoint_path: str, config_path: str \| None = None)` | bool | - |
| `get_config_bundle` | `get_config_bundle()` | ModelConfigBundle \| None | - |
| `is_loaded` | `is_loaded()` | bool | - |
| `get_current_checkpoint` | `get_current_checkpoint()` | str \| None | - |
| `cleanup` | `cleanup()` | None | - |

## Dependencies

### Imports
- logging
- time
- pathlib
- typing

### Internal Components
- config_loader (ModelConfigBundle, load_model_config, resolve_config_path)
- dependencies (OCR_MODULES_AVAILABLE, torch)
- model_loader (instantiate_model, load_checkpoint, load_state_dict)
- ocr.utils.path_utils (get_path_resolver)

### External Dependencies
- torch: Model execution and state management
- Model checkpoint files: `.pth` or `.ckpt` format
- Config files: YAML format with model architecture specifications

## State

- **Stateful**: Yes (caches loaded model, config, checkpoint path)
- **Thread-safe**: No
- **Lifecycle**: unloaded → loading → loaded → cleaned_up

### State Management

| State Variable | Type | Purpose |
|---------------|------|---------|
| model | Any \| None | Loaded PyTorch model instance |
| config | Any \| None | Parsed configuration object |
| _current_checkpoint_path | str \| None | Normalized path to loaded checkpoint |
| _config_bundle | ModelConfigBundle \| None | Full config bundle (model + preprocess + postprocess) |
| device | str | Device for inference ("cuda" or "cpu") |

## Constraints

- Requires OCR modules (torch, lightning) installed for model loading
- Auto-detects device if not specified (CUDA if available, else CPU)
- Caches single model at a time (loads new model replaces cached)
- Config file auto-detected from checkpoint location if not provided
- Checkpoint path normalized and cached to enable cache hit detection
- GPU memory cleared on cleanup (calls `torch.cuda.empty_cache()`)

## Backward Compatibility

✨ **New Component**: Introduced in v2.0
- Extracted from InferenceEngine model loading logic
- No public API changes to InferenceEngine
- Caching behavior improves performance over v1.x (avoids redundant loads)
