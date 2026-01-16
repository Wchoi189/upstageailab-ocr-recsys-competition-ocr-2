# Investigation Steps

## 1. Autoregressive Shift Fix
-   **Action**: Corrected `architecture.py` to target `t[1:]` (EOS) while inputting `t[:-1]` (BOS).
-   **Result**: No change in convergence. Model still predicted empty strings.

## 2. Tokenizer & Vocabulary
-   **Action**: Enforced `vocab_size` argument to avoid default mismatches.
-   **Result**: Confirmed `vocab_size=1027`. Loss remained high.

## 3. ViT [CLS] Token Removal
-   **Action**: Detected `TimmBackbone` (ViT) prepends `[CLS]` token.
-   **Fix**: Modified `architecture.py` to slice `visual_feat[:, 1:, :]`.
-   **Result**: Spatial alignment improved theoretically, but loss did not improve.

## 4. Positional Embedding Scaling
-   **Action**: Scaled `pos_emb` by `sqrt(d_model)` to match Token Embedding magnitude (standard Transformer practice).
-   **Result**: Slight improvement in gradient flow stability, but still no convergence.

## 5. Initialization (Xavier)
-   **Action**: Switched from `trunc_normal(std=0.02)` to `xavier_uniform`.
-   **Result**: Prevented potential vanishing gradients, but loss plateaued at ~3.0.

## 6. Resolution & Backbone Swap
-   **Hypothesis**: `32x128` image + `Patch 16` = `2x8` grid (Too coarse).
-   **Action A**: Force `224x224` input (14x14 grid). Result: Failed to improve (possibly due to massive padding or optimization difficulty).
-   **Action B**: Swap to **ResNet18** (Standard CNN).
-   **Result**: **SUCCESS**.
    -   Loss dropped from ~3.4 to ~1.76 rapidly.
    -   Confirmed ViT configuration/suitability was the root cause.
