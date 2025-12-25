---
title: "Phase 2-3: Hybrid VLM Pipeline & Optimization"
date: "2025-12-25"
author: "AgentQMS"
type: "feature"
---

# Phase 2-3: Hybrid VLM Pipeline & Optimization

Integrated the receipt extraction pipeline with a hybrid approach using **Qwen2.5-VL** for complex cases and a rule-based extractor for high-confidence/simple layouts. This release also includes backend optimizations and API enhancements.

## 🚀 Features

-   **Hybrid VLM Gating**: Automatically routes low-confidence or complex receipts to Qwen2.5-VL via vLLM.
-   **New Endpoint**: Added `POST /api/inference/extract` for structured receipt data extraction.
-   **VLM Extractor**: Robust `VLMExtractor` class with markdown parsing and metadata tagging.
-   **Optimization**: Orchestrator now supports extraction pipeline enabling/disabling.

## 🛠 Fixes

-   **Robustness**: Improved JSON parsing from VLM outputs to handle markdown code blocks.
-   **Observability**: Fixed `vlm_used` metric tracking in API responses.
-   **Code Quality**: Improved encapsulation in `InferenceService` and `InferenceEngine`.

## 📦 usage

```bash
# Start backend
uv run apps/ocr_inference_console/backend/main.py

# Send extraction request
curl -X POST http://localhost:8002/api/inference/extract \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "...", "enable_vlm": true}'
```
