# **OCR Text Recognition & Layout Analysis System**

<div> align="center">

[![CI](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E.svg)](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18)

**AI-optimized text-recognition system with layout analysis for accurate information extraction**

[English](README.md) • [한국어](README.ko.md)

## 📚 Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Research & Development](#research--development)
6. [Project Progress](#project-progress)
7. [Technical Stack](#technical-stack)
8. [Models & Performance](#models--performance)
9. [Documentation](#documentation)
10. [Contributing](#contributing)
11. [License](#license)

</div>

---

## **Overview**
This project originated from the **Upstage AI Bootcamp OCR competition** and has evolved into an end-to-end text recognition system with advanced layout analysis. It is currently undergoing final preparation for major architectural upgrades to enhance scalability and performance.

**Primary Repositories:**
- **Personal Continuation:** [Wchoi189/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2)
- **Original Bootcamp:** [AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)

---

## **Key Features**
- **Layout-Aware Text Recognition**: Combines detection, recognition, and spatial analysis for structured documents.
- **High-Performance ETL**: Converts 616k+ raw images to LMDB format in minutes with resumable multiprocessing.
- **Cloud-Native Scaling**: AWS Fargate integration bypasses API rate limits for large-scale document processing.
- **Robust Quality Assurance**: Structured bug reporting and incident resolution workflows.

---

## **System Architecture**
The system is designed with three core components:

1. **Project Compass**:
   - Centralized state management for AI agents via **Model Context Protocol (MCP)**.
   - Tools: `env_check`, `session_init`, `reconcile`, `ocr_convert`, `ocr_inspect`.
   - *State files*: [`current_session.yml`](project_compass/active_context/current_session.yml), [`blockers.yml`](project_compass/active_context/blockers.yml), [`dataset_registry.yml`](project_compass/environments/dataset_registry.yml).

2. **OCR ETL Pipeline**:
   - Converts raw datasets to optimized LMDB format for training.
   - Supports resumable processing to avoid data loss.

3. **Cloud Batch Processor**:
   - Serverless AWS Fargate architecture for asynchronous document parsing.
   - Uses S3 for storage and Parquet for efficient annotation querying.

---

## **Data Processing Pipeline**
### **Local Processing**
- **Tool**: `ocr-etl-pipeline`
- **Features**:
  - Zero-clutter LMDB conversion.
  - Multithreaded image decoding optimized for RTX 3090 workstations.
- **Performance**:
  - Processed 616,366 AI Hub samples in ~1 minute.
  - Reduced filesystem overhead by 99.9% (1 file vs. 600k files).

### **Cloud Processing**
- **Tool**: `aws-batch-processor`
- **Architecture**:
  - Serverless batch jobs on AWS Fargate.
  - Intermediate results stored in S3.
- **Data Catalog**: [`data_catalog.yaml`](aws-batch-processor/data/export/data_catalog.yaml)

---

## **Research & Development**
### **Key Insights**
- **KIE + Document Parse Abandonment**:
  - Receipts are linear/semi-structured, not tabular.
  - Document Parse API introduced HTML contamination in embeddings.
  - LayoutLM models underperformed compared to **PARSeq/CRNN** for receipt OCR.
- **Solution**: Shifted focus to text recognition pipelines using AI Hub datasets.

### **Experiment Tracking**
- **System**: `experiment_manager/`
- **Artifacts**:
  - Baseline metrics: [20251218_1415_report_baseline-metrics-summary.md](...)
  - Incident reports: [20251220_0130_incident_report_perspective_correction_data_loss.md](...)

---

## **Project Progress**
<div> align="center">

| Phase | Status | Progress |
|-------|--------|----------|
| **Core Development** | ✅ Complete | 100% |
| **Pre-Upgrade Preparation** | 🔄 In Progress | 90% |
| **Architectural Upgrades** | ⏳ Planned | 0% |

**Overall Progress: 85%**

</div>

**Current Focus**: Training Text Recognition Model (PARSeq/CRNN) on AI Hub LMDB dataset.

---

## **Technical Stack**
| Category | Technologies |
|----------|-------------|
| **ML/DL** | PyTorch, PyTorch Lightning, Hydra |
| **Backend** | FastAPI, ONNX Runtime |
| **Tools** | `uv` (Required), npm, W&B, Playwright, Vitest |
| **QMS** | AgentQMS (Artifacts, Standards, Compliance) |

---

## **Models & Performance**
| Model Name | Architecture | H-Mean | Hugging Face |
|------------|--------------|--------|--------------|
| **Receipt Detection KR** | DBNet + PAN (ResNet18) | 95.37% | [Model Card](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18) |

---

## **Documentation**
- **System Architecture**: [system-architecture.yaml](AgentQMS/standards/tier1-sst/system-architecture.yaml)
- **API Contracts**: [api-contracts.yaml](AgentQMS/standards/tier2-framework/api-contracts.yaml)
- **File Placement Rules**: [file-placement-rules.yaml](AgentQMS/standards/tier1-sst/file-placement-rules.yaml)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

## **Contributing**
Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## **License**
MIT License - see [LICENSE](LICENSE) for details.

<div> align="center">
[⬆ Back to Top](#ocr-text-recognition--layout-analysis-system)
</div>

