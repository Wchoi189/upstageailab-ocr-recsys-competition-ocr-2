# Research: KIE Model Selection Strategy

## 1. Introduction
This document evaluates potential models for the Key Information Extraction (KIE) task on the competition receipt/invoice dataset. The goal is to select a model that balances accuracy, inference speed, and ease of implementation.

## 2. Selection Criteria
- **Architecture**: Multimodal (Text + Layout + Image) vs Unimodal (Text + Layout)
- **Pre-training**: Availability of pre-trained weights (especially for Korean/multilingual if needed)
- **Performance**: SOTA on SROIE, FUNSD, CORD benchmarks
- **Efficiency**: Inference latency and resource requirements
- **Implementation**: Ease of use with Hugging Face Transformers or other libraries

## 3. Candidate Models

### 3.1 LayoutLMv3
- **Description**: Multimodal Transformer (Text + Layout + Image)
- **Pros**: SOTA on multiple benchmarks, unified architecture, strong visual understanding.
- **Cons**: Requires image inputs (larger input size), potentially slower than v2.
- **Suitability**: High. Strong candidate for receipts with complex layouts.

### 3.2 LayoutLMv2
- **Description**: Multimodal Transformer
- **Pros**: Good performance, handles image features.
- **Cons**: Older than v3, complex pre-training objectives.
- **Suitability**: Medium. v3 is generally preferred.

### 3.3 LiLT (Language-Independent Layout Transformer)
- **Description**: Decouples text and layout modeling. Can be combined with any pre-trained text model (e.g., XLM-RoBERTa).
- **Pros**: Excellent for multi-lingual tasks, flexible text backbone.
- **Cons**: Might lack deep visual feature alignment of LayoutLMv3.
- **Suitability**: High (if Korean language support is improved by swapping text encoder).

### 3.4 DONUT (Document Understanding Transformer)
- **Description**: OCR-free, end-to-end image-to-text generation.
- **Pros**: No OCR dependency, handles complex layouts well.
- **Cons**: Autoregressive generation can be slow, might hallucinate.
- **Suitability**: Medium-High. Good alternative if OCR quality is the bottleneck (though we have Upstage OCR).

### 3.5 PICK
- **Description**: Graph-based approach.
- **Pros**: Explicitly models relationships between text boxes.
- **Cons**: Older architecture, less support in modern libraries.
- **Suitability**: Low.

## 4. Evaluation Matrix

| Model | Multimodal? | Pre-trained? | Complexity | Inference Speed | Recommendation |
|---|---|---|---|---|---|
| LayoutLMv3 | Yes | Yes (Microsoft) | High | Medium | ⭐ Primary Candidate |
| LiLT | Yes (Layout) | Yes | Medium | Fast | Secondary Candidate |
| Donut | Yes (Vision) | Yes (Naver) | High | Slow | Backup |

## 5. Implementation Roadmap (Draft)

1.  **Baseline**: Train LayoutLMv3-base with pseudo-labels.
2.  **Comparison**: Train LiLT (with RoBERTa-base) on the same split.
3.  **Optimization**: Fine-tune hyperparameters for the winner.

## 6. References
- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [LiLT Paper](https://arxiv.org/abs/2202.13669)
- [Hugging Face Docs](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)
