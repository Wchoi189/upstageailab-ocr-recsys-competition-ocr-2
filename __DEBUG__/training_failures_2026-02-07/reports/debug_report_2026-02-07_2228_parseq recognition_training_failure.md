# Debugging Report: PARSeq Recognition Training Failure

## Executive Summary
The PARSeq recognition model fails to learn from a single training sample (Overfitting Test), consistently converging to a **Unigram Model** (repeating a single token or predicting empty string) with **Vanishing Gradients** (Grad Norm 0.0000). Despite exhaustive debugging of data, precision, architecture, and hyperparameters, the model refuses to align the `Visual Features` with `Text Tokens`.

## Key Findings
1.  **Data is Valid:** Input images have valid statistics (Mean ~1.6, Std ~1.2, Non-flat), and targets are valid (`"변화와"`). Tokenizer produces valid IDs.
2.  **Model Collapses to Unigram:** The model quickly learns to predict the most frequent token (or empty string) and then gradient descent stops.
3.  **Signal Mismatch:**
    - `Visual Features` (ResNet) had magnitude ~1.5.
    - `Learned Positional Embeddings` had magnitude ~0.2 (too small).
    - `Scaled Text Embeddings` had magnitude ~20.0 (too large?).
    - **Fix Attempted:** Scaled Image PE to match Text PE. Scaling `Visual Features`. Did not resolve collapse.
4.  **Optimization Barrier:**
    - High LR (`1e-2`) caused gradient explosion/NaN.
    - Medium LR (`1e-3`) caused unigram convergence.
    - Low LR (`1e-4`) was too slow.
5.  **Architecture Simplification:**
    - Reducing `d_model` to 256 and removing `Input Projection` did not help.
    - Using `Sinusoidal Initialization` for both Text and Image PE (to force alignment) did not help.

## Root Cause Hypotheses
1.  **Decoder Masking/Attention Implementation:** There may be a subtle bug in [PARSeqDecoder](file:///workspaces/ocr/domains/recognition/models/decoder.py#7-139) layer (e.g., `batch_first=False` mismatch, or `tgt_mask` direction) that prevents the model from attending to the image, forcing it to behave as a Blind Language Model.
2.  **Optimizability:** The loss landscape for `Alignment` is a narrow valley. Random initialization places the model in a broad `Unigram Basin`. Without Pretrained Weights, the optimizer cannot escape.
3.  **Complex Interaction:** The interplay between `ResNet` feature distribution and `Transformer` expected input distribution might be off (e.g. `LayerNorm` missing before `Transformer`?).

## Recommendations
1.  **Use Pretrained Weights:** Initialize `ResNet` and [PARSeq](file:///workspaces/ocr/domains/recognition/models/architecture.py#9-308) with weights trained on Synthetic Data (MJ/ST) to provide a good starting point for alignment.
2.  **Audit [PARSeqDecoder](file:///workspaces/ocr/domains/recognition/models/decoder.py#7-139):** careful line-by-line review of the Transformer implementation, specifically `generate_square_subsequent_mask` and [forward](file:///workspaces/ocr/domains/recognition/models/decoder.py#81-139) tensor shapes.
3.  **Simplify Problem:** Try training a simple `CRNN` (CNN + RNN + CTC) on the same data. If CRNN learns, the Data is fine, and PARSeq is the problem.
