# Findings & Recommendations

## Root Cause: ViT Patch Resolution
The primary failure mode was the use of `vit_small_patch16_224` on small OCR images (`32x128`).
-   ViT divides the image into non-overlapping `16x16` patches.
-   For a `32` pixel high image, this results in only **2 vertical patches**.
-   This vertical resolution is insufficient to capture the structural details of Korean characters (Hangul), leading to "feature collapse" where the model cannot distinguish tokens and defaults to the empty string prior.

## Comparison: ResNet18
-   ResNet18 (as configured) downsamples typically by 32x, but effectively preserves spatial locality in a way that aligns better with the Decoder's cross-attention mechanisms for OCR text lines.
-   Even with similar theoretical downsampling, the CNN's sliding window receptive field provides better feature overlap than non-overlapping ViT patches for this resolution.

## Action Taken
-   **Permanently Switched to ResNet18**: Updated `parseq.yaml` to use `resnet18` with `features_only=True` and `out_indices=[4]`.
-   **Adjusted Dimensions**: Updated Decoder `d_model` to `512` to match ResNet18 output.

## Future Recommendations
-   **Avoid Vanilla ViT for Low-Res OCR**: Unless using small patches (`patch_size=4` or `8`) or larger input images, standard ViT is ill-suited for strip-OCR.
-   **Stick to CNN Backbones**: ResNet or specialized OCR backbones (like SVTR with custom strides) are robust defaults.
