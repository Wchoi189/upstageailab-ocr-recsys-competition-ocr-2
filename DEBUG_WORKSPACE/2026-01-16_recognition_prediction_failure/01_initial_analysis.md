# Initial Analysis: BOS-EOS Model Collapse

## Failure Mechanisms

### 1. The "Empty Prediction" Loop
-   **Symptom**: The model immediately predicts `EOS` (Token ID 2) after the forced `BOS` (Token ID 1).
-   **Mechanism**: The Decoder learned that the safest generic prediction (to minimize loss in the absence of strong visual usage) was to end the sequence immediately.
-   **Observation**:
    ```
    GT: "Change and Creation"
    Pred: ""
    ```

### 2. Loss Plateau
-   **Symptom**: Training Loss stuck at ~3.4.
-   **Analysis**: If the model was completely broken, loss might explode. If it was learning well, it should drop < 0.1 on a single batch. A plateau at 3.4 suggests it was predicting a "safe" distribution (likely the prior of the vocab) without conditioning on the image.

### 3. Resolution Mismatch Hypothesis
-   **Hypothesis**: The input image size (`32x128`) combined with ViT Patch Size (`16`) results in a `2x8` feature grid.
-   **Impact**: Complex Korean characters cannot be resolved in a 2-pixel high vertical space. The model effectively sees "blur" and collapses to the dataset prior (which is short sequences -> EOS).

## Initial Checks
-   **AR Logic**: Verified `targets[:, :-1]` shifts. Correct.
-   **Tokenizer**: Verified `vocab_size` passed to Decoder. Correct.
-   **Data Loading**: Verified images are loaded and normalized. Correct.
