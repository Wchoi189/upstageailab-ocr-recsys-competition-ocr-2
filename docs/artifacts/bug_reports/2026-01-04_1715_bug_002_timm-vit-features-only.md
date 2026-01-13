---
ads_version: "1.0"
type: "bug_report"
category: "troubleshooting"
status: "completed"
severity: "major"
version: "1.0"
tags: ['encoder', 'timm', 'vit', 'initialization']
title: "TimmBackbone Failure with ViT Models (features_only support)"
date: "2026-01-04 17:15 (KST)"
branch: "main"
summary: "TimmBackbone failed to initialize Vision Transformer (ViT) models because it forcibly set `features_only=True`, which is not supported by standard ViT implementations in the `timm` library."
---

# Details

## Symptoms
When using `vit_small_patch16_224` as the encoder backbone:
```text
RuntimeError: features_only not implemented for Vision Transformer models.
```

## Root Cause
The `TimmBackbone` class was designed for CNNs (like ResNet) and unconditionally passed `features_only=True` to `timm.create_model` to extract intermediate feature maps. ViT models in `timm` do not support this argument/mode.

## Fix Implementation
Updated `ocr/models/encoder/timm_backbone.py`:
1. Wrapped `timm.create_model` in a `try-except RuntimeError` block.
2. If `features_only` fails, fallback to `features_only=False`.
3. In `forward()`, detect the non-features-only mode and use `self.model.forward_features(x)` to extract the sequence of patch embeddings.
4. Correctly populated `out_channels` and `strides` for ViT models (using `embed_dim` and `patch_size`).

## Verification
Confirmed via `tests/check_timm_vit.py` and `fast_dev_run` that the model initializes and passes data correctly.
