# Debugging Workspace: Recognition Model Collapse (BOS-EOS)

## Executive Summary
The Recognition model (PARSeq) was failing to learn, consistently predicting empty strings (`BOS` -> `EOS`) regardless of input. Even when verifying on a single batch (Overfitting), the model converged to a high loss (~3.4) and zero accuracy.

**Key Symptoms:**
- **Empty Predictions**: Output was always `''` (Empty string).
- **High Entropy Loss**: Loss plateaued around 3.0-3.5, indicating failure to extract meaningful signals.
- **Zero Accuracy**: Metric remained at 0.0 even after 100+ epochs on a single batch.

## Technical References
- **Architecture**: `ocr/features/recognition/models/architecture.py` (PARSeq)
- **Encoder**: `ocr/core/models/encoder/timm_backbone.py` (ViT vs ResNet)
- **Decoder**: `ocr/features/recognition/models/decoder.py` (Transformer Decoder)

## Resolution
The root cause was identified as the **ViT Backbone (Small, Patch 16)** being unsuitable for the input resolution (`32x128`), resulting in an extremely coarse feature grid (`2x8`) that lacked spatial detail for character recognition.

**Fix**: Switched Encoder to **ResNet18** (Standard CNN), resulting in immediate learning (Loss drops to 1.7).
