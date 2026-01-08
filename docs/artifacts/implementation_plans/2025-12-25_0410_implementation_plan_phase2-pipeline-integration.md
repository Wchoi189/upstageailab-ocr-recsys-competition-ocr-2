---
ads_version: "1.0"
title: "Phase 2-3: Pipeline Integration + VLM + Optimization"
date: "2025-12-25 04:10 (KST)"
type: "implementation_plan"
category: "development"
status: "pending"
version: "1.0"
tags: ['implementation', 'pipeline', 'vlm', 'extraction', 'phase2', 'phase3']
depends_on: ["2025-12-25_0410_implementation_plan_phase1-core-recognition.md"]
successor: null
---

# Phase 2-3: Pipeline Integration + VLM + Optimization

## Goal Description

Integrate the recognition, layout, and extraction modules into a unified end-to-end pipeline with hybrid VLM gating. This plan combines Phase 2 (Pipeline Integration + VLM) and Phase 3 (Optimization) into a single cohesive implementation due to their tight coupling.

**Scope**:
- Wire `LineGrouper` into `InferenceOrchestrator`
- Connect `ReceiptFieldExtractor` to recognition output
- Integrate Qwen2.5-VL-7B with confidence gating via vLLM
- Add `/api/inference/extract` endpoint
- Fine-tune PP-OCRv5 on receipt dataset
- Optimize throughput to 100+ pages/min

**Timeline**: 10-14 working days (after Phase 1 completion)

**Success Criteria**:
- End-to-end pipeline returns valid `ReceiptData` JSON
- 95% character accuracy on validation set
- 100+ pages/min throughput with hybrid gating
- VLM called for ≤20% of receipts

---

## User Review Required

> [!IMPORTANT]
> **VLM Deployment Decision**: Qwen2.5-VL will be served via vLLM as a separate microservice. This requires:
> - vLLM installation (~500MB)
> - AWQ quantized model download (~7GB)
> - Dedicated port (default: 8001)
>
> Confirm this architecture is acceptable.

> [!CAUTION]
> **Dataset Licensing**: The `mychen76/invoices-and-receipts_ocr_v1` dataset is for research use. Verify licensing compliance before production deployment.

---

## Proposed Changes

### Phase 2A: Pipeline Integration (Days 1-4)

#### [MODIFY] `ocr/inference/orchestrator.py`

Integrate layout and extraction into the prediction flow.

```diff
class InferenceOrchestrator:
    def __init__(self, device: str | None = None, enable_recognition: bool = False):
        # ... existing init ...
+       self._enable_layout = False
+       self._enable_extraction = False
+       self._layout_grouper = None
+       self._field_extractor = None

+   def enable_extraction_pipeline(self) -> None:
+       """Enable layout + extraction modules."""
+       from .layout.grouper import LineGrouper, LineGrouperConfig
+       from .extraction.field_extractor import ReceiptFieldExtractor, ExtractorConfig
+
+       self._enable_layout = True
+       self._enable_extraction = True
+       self._layout_grouper = LineGrouper(config=LineGrouperConfig())
+       self._field_extractor = ReceiptFieldExtractor(config=ExtractorConfig())

    def predict(
        self,
        image: np.ndarray,
        return_preview: bool = True,
+       enable_extraction: bool = False,
        # ... other args ...
    ) -> dict[str, Any] | None:
        # ... existing detection + recognition ...

+       # Stage 5: Layout grouping
+       if self._enable_layout and self._enable_recognition:
+           layout_result = self._run_layout_grouping(result)
+           result["layout"] = layout_result.model_dump()

+       # Stage 6: Field extraction with hybrid gating
+       if enable_extraction and self._enable_extraction:
+           receipt_data = self._run_extraction_with_gating(
+               layout_result, image
+           )
+           result["receipt_data"] = receipt_data.model_dump()

        return result

+   def _run_layout_grouping(self, result: dict) -> LayoutResult:
+       """Group recognized text into lines and blocks."""
+       from .layout.contracts import TextElement, BoundingBox
+
+       elements = []
+       for i, (poly, text, conf) in enumerate(zip(
+           result.get("polygons", []),
+           result.get("recognized_texts", []),
+           result.get("recognition_confidences", [])
+       )):
+           # Convert polygon string to coordinates
+           coords = self._parse_polygon(poly)
+           bbox = BoundingBox(
+               x_min=min(c[0] for c in coords),
+               y_min=min(c[1] for c in coords),
+               x_max=max(c[0] for c in coords),
+               y_max=max(c[1] for c in coords),
+           )
+           elements.append(TextElement(
+               polygon=coords,
+               bbox=bbox,
+               text=text,
+               confidence=conf,
+           ))
+
+       return self._layout_grouper.group_elements(elements)
```

---

### Phase 2B: VLM Integration with Confidence Gating (Days 5-7)

#### [NEW] `ocr/inference/extraction/vlm_extractor.py`

VLM-based semantic extraction via vLLM server.

```python
"""VLM-based receipt extraction using Qwen2.5-VL via vLLM."""

from __future__ import annotations

import logging
import httpx
from dataclasses import dataclass
from PIL import Image
import base64
import io

from .receipt_schema import ReceiptData

LOGGER = logging.getLogger(__name__)


@dataclass
class VLMExtractorConfig:
    """Configuration for VLM extraction."""
    server_url: str = "http://localhost:8001"
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    timeout: float = 5.0
    max_tokens: int = 2048


class VLMExtractor:
    """Extract receipt data using Qwen2.5-VL vision-language model."""

    EXTRACTION_PROMPT = '''Extract the following fields from this receipt image as JSON:
{
    "store_name": "...",
    "store_address": "...",
    "transaction_date": "YYYY-MM-DD",
    "transaction_time": "HH:MM",
    "items": [{"name": "...", "quantity": 1, "total_price": 0.00}],
    "subtotal": 0.00,
    "tax": 0.00,
    "total": 0.00,
    "payment_method": "card|cash",
    "card_last_four": "1234"
}
Return ONLY valid JSON, no explanation.'''

    def __init__(self, config: VLMExtractorConfig | None = None):
        self.config = config or VLMExtractorConfig()
        self._client = httpx.Client(timeout=self.config.timeout)
        LOGGER.info("VLMExtractor initialized | server=%s", self.config.server_url)

    def extract(self, image: Image.Image, ocr_context: str = "") -> ReceiptData:
        """Extract receipt data from image using VLM."""
        # Encode image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Build prompt with OCR context
        prompt = self.EXTRACTION_PROMPT
        if ocr_context:
            prompt = f"OCR Text:\n{ocr_context}\n\n{prompt}"

        # Call vLLM server
        response = self._client.post(
            f"{self.config.server_url}/v1/chat/completions",
            json={
                "model": self.config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": self.config.max_tokens,
            },
        )
        response.raise_for_status()

        # Parse response
        result = response.json()
        json_str = result["choices"][0]["message"]["content"]

        return ReceiptData.model_validate_json(json_str)

    def is_server_healthy(self) -> bool:
        """Check if vLLM server is responding."""
        try:
            response = self._client.get(f"{self.config.server_url}/health")
            return response.status_code == 200
        except Exception:
            return False
```

---

#### [MODIFY] `ocr/inference/orchestrator.py`

Add hybrid extraction with confidence gating.

```diff
+   def _run_extraction_with_gating(
+       self,
+       layout_result: LayoutResult,
+       original_image: np.ndarray,
+   ) -> ReceiptData:
+       """Extract receipt data with hybrid rule/VLM gating."""
+       # Try rule-based first (fast path: 80% of receipts)
+       receipt = self._field_extractor.extract(layout=layout_result)
+
+       # Gate to VLM if confidence too low or complex layout
+       if self._should_use_vlm(receipt, layout_result):
+           LOGGER.debug("Gating to VLM extraction (confidence=%.2f)", receipt.extraction_confidence)
+           receipt = self._run_vlm_extraction(original_image, layout_result)
+
+       return receipt
+
+   def _should_use_vlm(self, receipt: ReceiptData, layout: LayoutResult) -> bool:
+       """Determine if VLM extraction should be used."""
+       # Low confidence from rule-based extraction
+       if receipt.extraction_confidence < 0.7:
+           return True
+
+       # Complex layout indicators
+       if len(layout.blocks) > 5:  # Many separate text blocks
+           return True
+       if layout.tables:  # Table structures detected
+           return True
+
+       return False
+
+   def _run_vlm_extraction(
+       self,
+       image: np.ndarray,
+       layout: LayoutResult,
+   ) -> ReceiptData:
+       """Run VLM extraction with fallback to rule-based."""
+       try:
+           from .extraction.vlm_extractor import VLMExtractor
+           from PIL import Image
+
+           vlm = VLMExtractor()
+           if not vlm.is_server_healthy():
+               LOGGER.warning("VLM server unavailable, using rule-based")
+               return self._field_extractor.extract(layout=layout)
+
+           pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
+           return vlm.extract(pil_image, ocr_context=layout.text)
+       except Exception as e:
+           LOGGER.warning("VLM extraction failed: %s", e)
+           return self._field_extractor.extract(layout=layout)
```

---

### Phase 2C: API Endpoint (Day 8)

#### [MODIFY] `apps/ocr-inference-console/backend/main.py`

Add `/extract` endpoint.

```diff
+ from pydantic import BaseModel

+ class ExtractionRequest(BaseModel):
+     image_base64: str
+     enable_vlm: bool = True

+ class ExtractionResponse(BaseModel):
+     detection_result: dict
+     receipt_data: dict
+     processing_time_ms: float
+     vlm_used: bool

+ @app.post("/api/inference/extract", response_model=ExtractionResponse)
+ async def extract_receipt(request: ExtractionRequest):
+     """Extract structured data from receipt image."""
+     import base64
+     import time
+
+     start = time.perf_counter()
+
+     # Decode image
+     image_bytes = base64.b64decode(request.image_base64)
+     image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
+
+     # Run extraction pipeline
+     result = orchestrator.predict(
+         image,
+         return_preview=False,
+         enable_extraction=True,
+     )
+
+     elapsed = (time.perf_counter() - start) * 1000
+
+     return ExtractionResponse(
+         detection_result={
+             "polygons": result.get("polygons"),
+             "texts": result.get("recognized_texts"),
+         },
+         receipt_data=result.get("receipt_data", {}),
+         processing_time_ms=elapsed,
+         vlm_used="vlm" in result.get("receipt_data", {}).get("metadata", {}).get("extractor", ""),
+     )
```

---

### Phase 3A: Fine-tuning Pipeline (Days 9-11)

#### [NEW] `scripts/finetune_ppocr.py`

Fine-tuning script for PP-OCRv5 on receipt dataset.

```python
"""Fine-tune PP-OCRv5 on receipt dataset."""

import argparse
from pathlib import Path

from datasets import load_dataset
from paddleocr import PaddleOCR


def prepare_dataset(output_dir: Path):
    """Download and prepare mychen76/invoices-and-receipts dataset."""
    dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v1")

    # Convert to PaddleOCR format
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    label_file = train_dir / "label.txt"
    with open(label_file, "w") as f:
        for i, sample in enumerate(dataset["train"]):
            # Extract text regions and annotations
            image_path = train_dir / f"{i:06d}.jpg"
            sample["image"].save(image_path)

            # Write annotations
            for annotation in sample.get("annotations", []):
                text = annotation.get("text", "")
                f.write(f"{image_path}\t{text}\n")

    return train_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/receipts"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("Preparing dataset...")
    train_dir = prepare_dataset(args.output_dir)

    print("Fine-tuning PP-OCRv5...")
    # PaddleOCR fine-tuning command
    import subprocess
    subprocess.run([
        "python", "-m", "paddleocr.tools.train",
        "-c", "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml",
        "-o", f"Train.dataset.data_dir={train_dir}",
        "-o", f"Global.epoch_num={args.epochs}",
        "-o", f"Train.loader.batch_size_per_card={args.batch_size}",
    ])


if __name__ == "__main__":
    main()
```

---

### Phase 3B: Optimization (Days 12-14)

#### [NEW] `scripts/performance/benchmark_pipeline.py`

End-to-end throughput benchmark.

```python
"""Benchmark full extraction pipeline throughput."""

import time
import argparse
from pathlib import Path
import numpy as np
import cv2

from ocr.inference.orchestrator import InferenceOrchestrator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--enable-vlm", action="store_true")
    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = InferenceOrchestrator(enable_recognition=True)
    orchestrator.load_model("checkpoints/best_model.pth")
    orchestrator.enable_extraction_pipeline()

    # Collect test images
    images = list(args.images_dir.glob("*.jpg"))[:args.num_iterations]

    # Warmup
    for img_path in images[:5]:
        image = cv2.imread(str(img_path))
        orchestrator.predict(image, enable_extraction=True)

    # Benchmark
    start = time.perf_counter()
    vlm_calls = 0

    for img_path in images:
        image = cv2.imread(str(img_path))
        result = orchestrator.predict(image, enable_extraction=True)
        if result.get("receipt_data", {}).get("metadata", {}).get("vlm_used"):
            vlm_calls += 1

    elapsed = time.perf_counter() - start
    pages_per_min = len(images) / (elapsed / 60)

    print(f"Processed {len(images)} images in {elapsed:.2f}s")
    print(f"Throughput: {pages_per_min:.1f} pages/min")
    print(f"VLM calls: {vlm_calls}/{len(images)} ({100*vlm_calls/len(images):.1f}%)")


if __name__ == "__main__":
    main()
```

---

### Configuration

#### [NEW] `configs/extraction/hybrid.yaml`

Hybrid extraction configuration.

```yaml
defaults:
  - default
  - _self_

extraction:
  enabled: true

  # Hybrid gating settings
  gating:
    enabled: true
    min_confidence_for_rules: 0.7
    max_blocks_for_rules: 5

  # VLM settings
  vlm:
    enabled: true
    server_url: "http://localhost:8001"
    model: "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    timeout: 5.0

  # Rule-based extractor settings
  extractor:
    min_item_confidence: 0.5
    use_position_heuristics: true
    extract_items: true
    language: "ko"
```

---

### VLM Server Deployment

#### [NEW] `docker/docker-compose.vlm.yaml`

vLLM server deployment.

```yaml
version: "3.8"

services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ports:
      - "8001:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ
      --quantization awq
      --gpu-memory-utilization 0.3
      --max-model-len 4096
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Verification Plan

### Automated Tests

```bash
# 1. Pipeline integration tests
uv run pytest tests/integration/test_extraction_pipeline.py -v

# 2. VLM gating tests
uv run pytest tests/unit/test_vlm_extractor.py -v

# 3. API endpoint tests
uv run pytest tests/integration/test_extract_endpoint.py -v

# 4. Accuracy measurement on validation set
uv run python scripts/evaluate_accuracy.py \
    --dataset mychen76/invoices-and-receipts_ocr_v1 \
    --split test

# 5. Throughput benchmark (target: 100+ pages/min)
uv run python scripts/performance/benchmark_pipeline.py \
    --images-dir data/test_receipts \
    --num-iterations 100
```

### Manual Verification

1. **VLM Server Health**: `curl http://localhost:8001/health`
2. **Sample Extraction**: Run on 20 diverse receipts, verify JSON output
3. **VRAM Monitoring**: Verify total ≤14GB during operation
4. **Confidence Gating**: Verify VLM is called ≤20% of the time

---

## Task Checklist

### Phase 2A: Pipeline Integration (Days 1-4)
- [ ] Wire `LineGrouper` into orchestrator
- [ ] Implement `_run_layout_grouping()` method
- [ ] Implement `_parse_polygon()` helper
- [ ] Write integration tests for layout flow
- [ ] Verify 45 existing layout tests pass

### Phase 2B: VLM Integration (Days 5-7)
- [ ] Create `vlm_extractor.py` module
- [ ] Implement `VLMExtractor` class
- [ ] Implement confidence gating logic
- [ ] Deploy vLLM server with AWQ model
- [ ] Write VLM unit tests
- [ ] Verify VRAM stays under 8GB

### Phase 2C: API Endpoint (Day 8)
- [ ] Add `/api/inference/extract` endpoint
- [ ] Create request/response models
- [ ] Add OpenAPI documentation
- [ ] Write endpoint integration tests

### Phase 3A: Fine-tuning (Days 9-11)
- [ ] Download mychen76 dataset
- [ ] Write dataset preparation script
- [ ] Configure PP-OCRv5 fine-tuning
- [ ] Run training (~4-8 hours)
- [ ] Evaluate on validation set

### Phase 3B: Optimization (Days 12-14)
- [ ] Create throughput benchmark script
- [ ] Profile bottlenecks
- [ ] Tune batch sizes
- [ ] Tune confidence thresholds
- [ ] Document final metrics

---

## Rollback Plan

If integration fails:

1. **VLM Issues**: Disable VLM gating, use rule-based only
2. **Pipeline Issues**: Revert orchestrator changes, keep Phase 1 working
3. **Performance Issues**: Reduce batch sizes, disable VLM, optimize later

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| E2E pipeline functional | Yes | `/extract` returns valid JSON |
| Character accuracy | ≥95% | mychen76 test set |
| Throughput | ≥100 pages/min | Benchmark script |
| VLM call rate | ≤20% | Benchmark script |
| VRAM (concurrent) | ≤14GB | `nvidia-smi` |
| API latency (p99) | ≤2s | Benchmark script |

---

## Dependencies & Blockers

| Dependency | Status | Mitigation |
|------------|--------|------------|
| Phase 1 completion | Pending | Wait for Phase 1 PR merge |
| vLLM installation | Unverified | Docker image available |
| AWQ model download | ~7GB | Pre-download in setup |
| mychen76 dataset | ~2GB | Auto-download via HuggingFace |
| GPU runner (CI) | User-managed | Document manual testing |

---

## Next Steps After Completion

Upon successful completion of Phase 2-3:

1. Create PR with all changes
2. Deploy to staging environment
3. Run production load tests
4. Create deployment documentation
5. Archive implementation plans
