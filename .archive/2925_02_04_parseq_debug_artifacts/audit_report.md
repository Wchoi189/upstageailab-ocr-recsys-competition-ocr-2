# Audit Report: Recognition Pipeline Post-Refactor

**Date:** 2026-02-04  
**Audit ID:** recognition-audit-post-refactor  
**Status:** Complete - Root Cause Identified  

---

## 1. Executive Summary

The recognition pipeline is currently **broken** due to an incomplete V5 atomic migration. While the [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) architecture class is updated to support atomic instantiation (injection of components), the Hydra configuration fails to provide the necessary `head` and `loss` components.

The reported "optimizer got an empty parameter list" error was a **false lead**. The model successfully instantiates with **55.8M parameters**. The actual runtime failure is a `TypeError` during the forward pass because `self.head` is `None`.

---

## 2. Key Findings

### Finding 1: False "Empty Parameter" Signal
- **Evidence:** Dry-run script confirmed `model.parameters()` yields **55,853,632 params**.
- **Explanation:** The optimizer error likely occurred in a `fast_dev_run` scenario where `Lightning` logic interacted with the incomplete model state, or was an artifact of a previous uncommited state.
- **Status:** Debunked ✅

### Finding 2: Missing Atomic Components (The Root Cause)
- **Evidence:** [rec_config_dump.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/rec_config_dump.yaml) shows `head` and `loss` are completely absent.
- **Evidence:** Dry-run confirmed `model.head` is `None` and `model.loss` is `None`.
- **Impact:** `PARSeq.forward()` calls `self.head(...)` resulting in `TypeError: 'NoneType' object is not callable`.
- **Status:** Critical Fix Required ❌

### Finding 3: Structural Deviation from Detection
- **Detection Pipeline:** Uses [OCRModel](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py#16-309) (generic) with flat [model](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/__init__.py#4-35) config. atomic components explicit.
- **Recognition Pipeline:** Uses [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) (specific) with nested `model.architectures` config.
- **Impact:** Inconsistent configuration patterns across domains.

---

## 3. Component Status

| Component | Class | Status | Configuration |
| :--- | :--- | :--- | :--- |
| **Model** | [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) | ✅ **Ready** | Atomic instantiation logic implemented |
| **Encoder** | `TimmBackbone` | ✅ **Ready** | `resnet18` correctly configured |
| **Decoder** | `PARSeqDecoder` | ✅ **Ready** | Transformer decoder configured |
| **Head** | *Unknown* | ❌ **Missing** | No config, defaults to None |
| **Loss** | *Unknown* | ❌ **Missing** | No config, defaults to None |

---

## 4. Recommendations

### Immediate Fixes (V5 Compliance)
1.  **Define Head:** Add `head` to [configs/model/architectures/parseq.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/model/architectures/parseq.yaml). Likely a Linear layer or dedicated `PARSeqHead` if one exists (need to verify `ocr/domains/recognition/models/head`?).
2.  **Define Loss:** Add `loss` to [configs/model/architectures/parseq.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/model/architectures/parseq.yaml). Likely `CrossEntropyLoss` since [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) handles simple token classification.
3.  **Fix Domain Config:** Ensure [configs/domain/recognition.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/domain/recognition.yaml) is correctly structured and not overridden by string "recognition".

### Long-Term
1.  **Standardize Model Config:** Refactor Recognition to use the same [OCRModel](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py#16-309) flat structure as Detection (`model._target_ = OCRModel` with `architecture_name=parseq` passed as arg, or migrating [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) to fully generic structure).

---

## 5. Conclusion

The recognition pipeline is **recoverable** without major code changes. The "deferred" status was likely due to these missing configuration pieces rather than deep architectural flaws. By applying standard V5 atomic pattern (injecting Head/Loss), we can restore functionality.
