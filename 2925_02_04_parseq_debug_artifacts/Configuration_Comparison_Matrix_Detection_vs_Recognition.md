# Configuration Comparison Matrix: Detection vs Recognition

**Date:** 2026-02-04  
**Phase:** 1 (Configuration Archaeology)  
**Status:** Findings Documented  

---

## 1. High-Level Comparison

| Feature | Detection Config (`det_resnet50_v1`) | Recognition Config (`rec_baseline_v1`) | Status |
| :--- | :--- | :--- | :--- |
| **Model Entry Point** | [model](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/__init__.py#4-35) (direct) | `model.architectures` (nested) | ❌ **Major Deviation** |
| **Target Class** | `ocr.core.models.architecture.OCRModel` | `ocr.domains.recognition.models.architecture.PARSeq` | ⚠️ Different Patterns |
| **Atomic Components** | Encoder, Decoder, Head, Loss | Encoder, Decoder ONLY | ❌ **Missing Head/Loss** |
| **Domain Config** | `domain.task: detection` | `domain: recognition` (legacy string?) | ⚠️ Inconsistent Structure |
| **Data Transforms** | `data.transforms.train_transform` etc. | `data.transforms` (single entry) | ⚠️ Different Augmentation Strategy |

---

## 2. Model Configuration Structure

### Detection (Reference - V5 Standard)
**Pattern:** Generic Container + Injected Components
```yaml
model:
  _target_: ocr.core.models.architecture.OCRModel
  encoder:
    _target_: ocr.core.models.encoder.timm_backbone.TimmBackbone
    model_name: resnet50
  decoder:
    _target_: ocr.domains.detection.models.decoders.fpn_decoder.FPNDecoder
  head:  # ✅ PRESENT
    _target_: ocr.domains.detection.models.heads.db_head.DBHead
  loss:  # ✅ PRESENT
    _target_: ocr.domains.detection.models.loss.db_loss.DBLoss
```

### Recognition (Audit Target - Broken)
**Pattern:** Specific Architecture Class
```yaml
model:
  architectures:  # ⚠️ Key Difference: Nested under 'architectures'
    _target_: ocr.domains.recognition.models.architecture.PARSeq
    encoder:
      _target_: ocr.core.models.encoder.timm_backbone.TimmBackbone
      model_name: resnet18
    decoder:
      _target_: ocr.domains.recognition.models.decoder.PARSeqDecoder
    # ❌ MISSING: head
    # ❌ MISSING: loss
    vocab_size: 1000  # ⚠️ Injected here, but also in decoder?
  vocab_size: 1000
```

---

## 3. Key Findings

### Finding 1: Two Different Instantiation Paths
- **Detection** relies on `OCRModel` (generic) which likely takes `head` and `loss` in [__init__](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#15-50).
- **Recognition** relies on [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) module directly. The [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) class *must* support atomic injection of `head` and `loss` if we want to follow V5, but they are absent in config.

### Finding 2: Missing Critical Components
- Recognition config has **NO definition** for `head` or `loss`.
- Even if [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) has internal defaults (legacy behavior), the V5 atomic goal is to make them explicit.
- This explains why `self.head` and `self.loss` are `None` at runtime.

### Finding 3: Config Structure Mismatch
- Detection: `domain.task` = `detection`.
- Recognition: `domain` = `recognition` (string value, not struct?).
  - *Correction based on dump:* `domain: recognition` is at root of dumped config? No, it's `domain: recognition` line 155 in dump.
  - In Detection dump line 255: `domain:` then indented `task: detection`.
  - **Result:** Recognition domain config is likely flat or malformed compared to Detection.

---

## 4. Interpolation Trace

### Recognition Vocab Size
- `model.architectures.decoder.vocab_size` -> `${model.vocab_size}`
- `model.vocab_size` -> `1000` (Literal)
- **Chain:** Verified ✅

### Recognition Paths
- `data.train_dataset.lmdb_path` -> `${global.paths.root_dir}/data/processed/...`
- **Chain:** Verified ✅

---

## 5. Next Steps (Phase 2)

1. **Verify Instantiation Logic:** Check [ocr/core/models/__init__.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/__init__.py) to confirm how it handles `model.architectures` vs [model](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/__init__.py#4-35) target.
2. **Inspect PARSeq Signature:** Does `PARSeq.__init__` accept `head` and `loss`?
3. **Trace `OCRModel`:** How does Detection's `OCRModel` use its components vs Recognition's [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257)?
