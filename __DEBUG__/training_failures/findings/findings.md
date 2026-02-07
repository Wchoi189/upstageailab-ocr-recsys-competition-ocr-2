# Critical Findings: 0% Accuracy Failure

> **Status:** 🔴 Active & Critical
> **Date:** 2026-02-07
> **Hardware:** RTX 3090

## The Issue
Despite a successful refactor and stable training loop, the model **fails to learn anything** even after 6 epochs.

**Symptoms:**
- `val/acc`: **0.000** (Flatline)
- `val/cer`: **1.000** (100% Error)
- Predictions: Repetitive characters (e.g., '33333333', '설설설설설설') or empty strings.
- Loss: Converging to ~6.78 but effectively high.

## Validated Context
- **Data Pipeline**:
    - Images are normalized (approx -2 to 2).
    - Tokenizer works (batch 0 decodes correctly).
    - BOS/EOS tokens are present in ground truth.
- **Model**:
    - PARSeq + ResNet + Transformer Decoder.
    - Vocab Size: 1027.

## Hypotheses & Action Plan

### 1. [INVALID] Data Pipeline Corruption
- **Status**: Disproven.
- **Evidence**: Diagnostic script confirms images are loaded, normalized, and labels are tokenized correctly with BOS/EOS.

### 2. [INVALID] Metric Calculation Error
- **Status**: Disproven.
- **Evidence**: Manual debug prints show predictions are indeed garbage ('3333333') vs correct GT ('"변화와'). The 0% accuracy is real.

### 3. [ACTIVE] Model Collapse / Initialization
- **Status**: High Probability.
- **Hypothesis**: The model is collapsing to a trivial solution (predicting the same token repeatedly) or gradients are exploding/vanishing.
- **Evidence**: Predictions are '33333333' or '설설설설설설'. This is classic mode collapse in autoregressive models.
- **Action**:
    - Check Transformer Masking: Is the causal mask correct?
    - Check Learning Rate: Is it too high? (Default 1e-4 might be too high for this vocab/batch).
    - Check Gradient Clipping.

### 5. [CONFIRMED] Optimization Barrier & Signal Mismatch
- **Status**: Critical.
- **Hypothesis**: The model is stuck in a local minimum where it ignores the image and predicts a unigram distribution (repetition or empty string).
- **Evidence**:
    - **Runs 22-30**: Varying LR (1e-4 to 1e-2) either caused slow convergence to unigram or explosion.
    - **Run 31**: Gradient logs showed `Pos Encoder` and `Embed Tokens` grads were 0.0 or negligible.
    - **Run 35**: Debug prints revealed `Visual Features` (Mean ~1.5) dwarfed `Learned Positional Embeddings` (Mean ~0.2), making spatial attention impossible.
    - **Run 36-39**: Scaling PE x5 and using Sinusoidal Init (Magnitude ~8.0) did *not* fix the unigram collapse.
    - **Run 40**: Input Images are verified valid (Mean ~1.6, Std ~1.2).
    - **Run 41**: Synthetic Data (Random Noise + Fixed Token Sequence) *failed to learn*. The model predicted empty strings even on fixed synthetic data.

## Conclusion
The issue is likely **not** data corruption but a fundamental flaw in the **PARSeq Decoder implementation** (masking, alignment logic) or a **Need for Pretraining**. The model starts in a basin where "Ignore Image, Predict Frequent Token" is the steepest gradient descent path, and it cannot escape.

## Recommended Next Actions
1.  **Switch Architecture**: Try a simpler CRNN (CNN+RNN+CTC) to verify the dataset and optimization loop on a simpler topology.
2.  **Pretrained Weights**: Initialize with weights trained on Synthetic Data (MJSynth) to bypass the alignment cold-start problem.
3.  **Deep Code Audit**: Re-verify `PARSeqDecoder` masking and cross-attention logic line-by-line.
