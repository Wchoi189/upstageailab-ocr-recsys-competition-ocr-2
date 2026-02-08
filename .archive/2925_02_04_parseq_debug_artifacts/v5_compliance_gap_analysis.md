# V5 Compliance Gap Analysis: Recognition Pipeline

**Date:** 2026-02-04  
**Status:** Audit Findings  
**Reference:** Detection Pipeline (V5 Gold Standard)

---

## Executive Summary
The Recognition pipeline currently fails V5 compliance in three critical areas. While the underlying code ([PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) class) is capable of V5-style atomic instantiation, the configuration layer is using a mixture of legacy nested patterns and missing component definitions. This results in a broken runtime state where `head` and `loss` are `None`.

---

## Gap 1: Model Configuration Pattern (Critical)

| Feature | Detection (V5 Standard) | Recognition (Current State) |
| :--- | :--- | :--- |
| **Root Key** | [model](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/__init__.py#4-35) | `model.architectures` |
| **Instantiator** | [OCRModel](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py#16-309) (Generic) | [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) (Specific) |
| **Structure** | Flat (Model owns components) | Nested (Architecture owns components) |

**Analysis:**
The V5 standard expects [model](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/__init__.py#4-35) to be the entry point for instantiation. Recognition currently nests its definition under `architectures`, likely a holdover from the legacy registry system where `architecture_name` was looked up.

**Remediation:**
- Move [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) definition to [model](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/__init__.py#4-35) root in config.
- OR ensure [OCRModel](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/architecture.py#16-309) wrapper is used and [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) is passed as the `architecture` argument (less preferred for atomic).
- **Target State:** `model: { _target_: ocr.domains.recognition.models.architecture.PARSeq, ... }`

---

## Gap 2: Missing Atomic Components (Blocker)

| Component | Detection Config | Recognition Config |
| :--- | :--- | :--- |
| **Encoder** | ✅ Defined | ✅ Defined |
| **Decoder** | ✅ Defined | ✅ Defined |
| **Head** | ✅ Defined (`DBHead`) | ❌ **MISSING** |
| **Loss** | ✅ Defined (`DBLoss`) | ❌ **MISSING** |

**Analysis:**
The [PARSeq](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/models/architecture.py#8-257) class correctly implements atomic instantiation (`if encoder: self.head = head`), but the configuration fails to provide these keys. This leads to `self.head` being `None`, which will cause a `TypeError` during the forward pass. This is the likely root cause of runtime failures.

**Remediation:**
- Define `head` in [configs/model/architectures/parseq.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/model/architectures/parseq.yaml).
- Define `loss` in [configs/model/architectures/parseq.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/model/architectures/parseq.yaml).
- Ensure appropriate classes exist (e.g., `PARSeqHead` or `Linear`, `CrossEntropyLoss`).

---

## Gap 3: Domain Configuration Structure (Major)

| Feature | Detection Config | Recognition Config |
| :--- | :--- | :--- |
| **Output Type** | Struct (Dict) | String (`"recognition"`)? |
| **Content** | `task`, `batch_size`, etc. | `task: recognition` |

**Analysis:**
The dumped config shows `domain: recognition` (string) for recognition, while detection has a full dictionary. This suggests the [configs/domain/recognition.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/domain/recognition.yaml) is not being composed correctly or is being overridden by a string value (likely in `experiment` config). This breaks the [OCRProjectOrchestrator](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/pipelines/orchestrator.py#22-213) which expects to read `domain.task`.

**Remediation:**
- Fix [configs/experiment/rec_baseline_v1.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/experiment/rec_baseline_v1.yaml) if it overrides domain.
- Ensure [configs/domain/recognition.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/domain/recognition.yaml) maps correctly to the `domain` namespace.

---

## Action Plan
1.  **Fix Domain Config:** trace why it is a string.
2.  **Align Model Config:** Flatten `model.architectures` to [model](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/models/__init__.py#4-35).
3.  **Add Missing Components:** Add `head` and `loss` to PARSeq config.
