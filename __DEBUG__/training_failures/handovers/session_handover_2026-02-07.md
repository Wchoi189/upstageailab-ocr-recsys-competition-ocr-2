# Session Handover: Recognition Training Failure Debugging

**Date:** 2026-02-07
**Status:** 🔴 Active Issue (Model Collapse)
**Context:** `__DEBUG__/training_failures`

## Executive Summary
We spent the session debugging a persistent 0% accuracy / 100% CER failure in the PARSeq recognition model. Despite verifying data integrity, adjusting learning rates, scaling embeddings, and simplifying the architecture, the model consistently collapses to a **Unigram Prediction** (predicting the same character repeatedly or an empty string).

**Key Conclusion:** The issue is likely **not** data corruption (inputs are valid). It appears to be a fundamental optimization barrier (cold start problem) or a bug in the Transformer Decoder's masking/attention implementation that prevents it from "seeing" the visual features.

## Critical Findings
- **Data is Valid:** Input images and target tokens are correct.
- **Gradient Failure:** Gradients for `Pos Encoder` and `Embed Tokens` are near zero or vanish quickly.
- **Signal Mismatch:** Visual features (~1.5) were far larger than learned positional embeddings (~0.2), but even scaling PE to ~8.0 did not fix the collapse.
- **Synthetic Fail:** The model failed to learn even a fixed synthetic sequence `[1, 2, 3]` from random noise, suggesting a broken learning loop or decoder.

## Actionable Next Steps
1.  **Try CRNN:** Implement a simple CNN-RNN-CTC model (e.g., ResNet + BiLSTM + CTC).
    - **Goal:** Verify if *any* model can learn from this data pipeline. If CRNN works, the data is fine, and PARSeq is broken.
2.  **Pretrained Weights:** If sticking with PARSeq, load weights pretrained on MJSynth/SynthText.
    - **Goal:** A better initialization might bypass the "unigram basin" of attraction.
3.  **Audit Decoder:** meticulous review of `PARSeqDecoder` code, specifically `generate_square_subsequent_mask` and how `tgt_key_padding_mask` is applied (or not).

## Artifacts
- `findings.md`: Detailed log of hypotheses and evidence.
- `debug_report_2026-02-07_2228.md`: Comprehensive report of this session.
- `logs/`: Contains `run*.log` files from the overfitting tests.
