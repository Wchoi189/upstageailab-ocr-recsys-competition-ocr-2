# Decoder Refactoring and Segfault Fix Walkthrough

## Overview
We refactored the [OCRModel](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py#16-284) and [PARSeqDecoder](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/decoder.py#7-111) to fix a potential segmentation fault caused by incorrect mask types in PyTorch 2.x and to correct the architectural coupling between the decoder and the head.

## Changes

### 1. OCRModel Refactoring
We modified `OCRModel.forward` in [architecture.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py) to:
*   Pass `targets` to the decoder during training, enabling proper autoregressive (AR) training.
*   Implement [generate()](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py#75-122) method to handle the AR generation loop (greedy decoding) during inference.
*   Conditionally trigger [generate()](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py#75-122) only for AR models (checking `bos_token_id`) to maintain compatibility with detection models.

```python
# Before
decoded_features = self.decoder(encoded_features) # No targets passed

# After
if targets is not None:
    decoded_features = self.decoder(encoded_features, targets=targets)
```

### 2. Decoder Clean-up
We updated [decoder.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/decoder.py) to:
*   Remove the broken [generate](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py#75-122) method (now handled by [OCRModel](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py#16-284)).
*   Enforce that `targets` are provided during [forward](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/base_classes.py#191-209).
*   Ensure `tgt_key_padding_mask` is cast to [float](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/train.py#145-155) to avoid PyTorch 2.x segfaults.

```python
# Fix for segfault
tgt_key_padding_mask = (targets == self.pad_token_id).float()
```

### 3. Detection Compatibility
We updated [BaseDecoder](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/base_classes.py#71-111) and all detection decoders ([FPNDecoder](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/fpn_decoder.py#23-73), [PANDecoder](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/decoder/pan_decoder.py#35-100), `CRAFTDecoder`, [DBPPDecoder](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/detection/models/decoders/dbpp_decoder.py#36-98)) to accept optional `targets` to align with the refined [OCRModel](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py#16-284) interface.

## Verification Results

### Training with `num_workers=4`
We ran a test training command to verify that the segregation fault (which occurred when `num_workers > 0`) is resolved.

Command:
```bash
python runners/train.py domain=recognition model/architectures=parseq +data.num_workers=4 +trainer.fast_dev_run=True
```

Result:
> [!NOTE]
> Training completed successfully with `num_workers=4`, confirming the fix. We also verified that Detection models continue to work correctly.

![Verification Success](/home/vscode/.gemini/antigravity/brain/529c07cd-8445-40d6-a7f7-882dad22fb47/segfault_verification_success_1768418435014.png)

## Next Steps
*   Run full training to verify convergence.
*   Implement beam search in `OCRModel.generate` if needed.
