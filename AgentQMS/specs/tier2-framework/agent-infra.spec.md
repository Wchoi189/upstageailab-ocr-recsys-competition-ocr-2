---
ads_version: '2.0'
id: 'AG-006'
type: 'tool_catalog'
tier: 3
priority: 'high'
updated: '2026-02-19'
spec_version: '1.0.0'
description: 'Ollama model catalog for Qwen and hybrid OCR agent families. Covers local inference, DashScope API, and hybrid pipeline configurations.'
dependencies:
  - SC-007
  - FW-034
---

# Ollama Models Configuration

> Ollama model catalog with specifications for Qwen model family and OCR-capable variants

## Specification

```yaml
agents:

  qwen:
    description: General coding and utility agents (local Ollama)
    endpoint: http://host.docker.internal:11434
    dependencies:
      - SC-007
      - FW-034
    inventory:
      - name: qwen3-coder:30b
        role: Architect
        context_window: 1048576
        vram_requirement: 18 GB
        strengths:
          - Repo-scale refactoring
          - Complex logic
          - Multi-file impact analysis

      - name: qwen3:4b-instruct
        role: Validator
        context_window: 262144
        vram_requirement: 2.5 GB
        strengths:
          - Thinking mode
          - Logical verification
          - Quality scoring

      - name: qwen3:1.7b
        role: Utility / Janitor
        context_window: 32768
        vram_requirement: 1.4 GB
        strengths:
          - Log parsing
          - Metadata extraction
          - Schema validation

  qwen-local-ocr:
    description: Local Ollama vision-language models for Korean OCR on 32x128 patches
    endpoint: http://host.docker.internal:11434
    dependencies:
      - SC-007  # Patch preprocessor (32x128 grayscale/contrast)
      - FW-034  # Ollama batch API wrapper
    inventory:
      - name: qwen2.5vl:7b
        role: Primary OCR Extractor (Vision-Language)
        context_window: 131072
        vram_requirement: 6.0 GB
        strengths:
          - Multilingual OCR (Korean incl.); handles blur/tilt on 32x128 patches
          - DocVQA-level accuracy for cropped text; outputs clean transcriptions
          - 'Prompt: "Extract exact Korean text. JSON: {text: ..., conf: 0.95}"'

      - name: qwen3:4b-instruct
        role: Text Validator / Confidence Scorer
        context_window: 262144
        vram_requirement: 2.5 GB
        strengths:
          - 'Post-OCR cleanup: verify extracted text against Korean admin codes'
          - Logical checks, hallucination filtering, quality scoring (1-10)
          - Lightweight for batch validation of golden dataset labels

      - name: qwen3:1.7b
        role: Fast Text Utility / Fallback Validator
        context_window: 32768
        vram_requirement: 1.4 GB
        strengths:
          - Ultra-quick checks on high-volume patches (char freq, validity)
          - Embed-like similarity to real gt_text corpus
          - Chain after VL OCR for scale

    usage_notes:
      prompt_template_vl: |
        <image>From this 32x128 Korean text patch, extract ONLY the exact text as JSON:
        {"text": "가나다...", "confidence": 0.95}
      prompt_template_text: |
        Extracted: "{{vl_text}}". Rate accuracy (0-1), suggest fixes for OCR errors. Korean admin doc context.
      optimizations:
        - 'VL first: single-image mode for speed on patches'
        - 'Chain: VL OCR → qwen3:4b validate → qwen3:1.7b score'
        - 'Batch: 1-4 patches/inference; temp=0.1 for determinism'
        - 'Korean boost: mention "Korean Hangul OCR" in VL prompts'
        - 'Monitor: if hallucinations >10%, fine-tune prompt or add deskew'

  dashscope-qwen-ocr:
    description: DashScope API (Alibaba Cloud) Qwen VL models for cloud OCR
    endpoint: https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
    auth: Bearer $DASHSCOPE_API_KEY  # Get from Model Studio console
    dependencies:
      - SC-007  # 32x128 patch prep (base64 encode for API)
      - FW-034  # Async batch caller for DashScope
    inventory:
      - name: qwen-vl-ocr-2025-11-20
        role: Specialized OCR Extractor
        context_window: dynamic
        cost: ~$0.001-0.005/image
        strengths:
          - Text extraction from docs/forms; multilingual incl. Korean
          - 'Built-in tasks: general_ocr, table_parsing, doc_parsing → JSON/LaTeX/HTML'
          - '32x128 optimized: min_pixels=3072 (scales up small images), max_pixels=8M'
          - High accuracy on patches; no downscale loss

      - name: qwen-vl-max
        role: Advanced VL OCR + Reasoning
        context_window: dynamic
        cost: Higher (~2x OCR model)
        strengths:
          - 'Full Qwen-VL: OCR + layout understanding, VQA on patches'
          - Handles blur/tilt/handwriting; Korean Hangul robust
          - Dynamic resolution for tiny crops; agentic for golden QA

    usage_notes:
      prompt_template_ocr: |
        {"messages": [{"role": "user", "content": [{"image": "base64://...", "min_pixels": 3072, "max_pixels": 8388608, "enable_rotate": true}]}],
         "ocr_options": {"task": "general_ocr"}}
      sdk_example_python: |
        import dashscope
        dashscope.api_key = 'sk-xxx'
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-ocr-2025-11-20',
            messages=[...],
            ocr_options={"task": "general_ocr"}
        )
      optimizations:
        - 'Tune pixels: 32x128 (~4k px) → set min_pixels=4096 for upscale if needed'
        - Batch via async; JSON mode for golden dataset
        - 'Fallback: qwen2.5-vl-plus if qwen-vl-ocr insufficient'
        - 'Cost control: test on sample set; OCR model cheaper than max'

  hybrid-ocr:
    description: Hybrid pipeline combining local LLM (olmOCR) and Python toolkit (Surya)
    dependencies:
      - SC-007  # Patch loader (PIL, 32x128 grayscale)
      - FW-034  # Orchestrator (batch dispatch)
    inventory:
      - name: richardyoung/olmocr2:7b-q8
        type: ollama_vl_llm
        endpoint: http://host.docker.internal:11434
        role: Reasoning OCR (docs/tables/handwriting)
        context_window: 131072
        vram_requirement: 9-12 GB
        install: 'ollama pull richardyoung/olmocr2:7b-q8  # 8.85 GB'
        strengths:
          - 82.4 olmBench; structured MD/JSON from patches
          - Korean via Qwen2.5-VL base; hallucination-resistant
        prompt_template: |
          <image>Extract Korean text from 32x128 patch as JSON: {"text": "...", "conf": 0.95}

      - name: surya-ocr
        type: python_toolkit
        endpoint: local_pytorch
        role: High-Throughput Line OCR (printed text)
        vram_requirement: 2-4 GB
        install: 'pip install surya-ocr torch torchvision  # Auto-dl models ~2GB'
        strengths:
          - 90+ langs incl. Korean; bbox+conf per line
          - Batch 32+ on 1 GPU; CPU fallback
        usage_snippet: |
          from surya.ocr import run_ocr
          res = run_ocr(images=[pil_img], langs=['ko'])
          golden_text = res[0]['text']
        params:
          detector_text_threshold: 0.7
          detector_blank_threshold: 0.3

    usage_notes:
      workflow:
        - 'Primary: Surya for speed/scale → golden bbox+text'
        - 'Fallback: olmOCR if conf<0.8 or handwriting detected'
        - 'Post: qwen3:4b-instruct validate extraction'
      output_schema:  # Unified golden format
        text: str
        bbox: '[int, int, int, int]  # [x1,y1,x2,y2] norm to 32x128'
        confidence: 'float  # 0-1'
      optimizations:
        - "Surya: GPU batch_size=32; langs=['ko'] only"
        - 'olmOCR: temp=0.1; single-image mode'
        - 'Hybrid thresh: Surya conf>0.9 → accept; else olmOCR'
```
