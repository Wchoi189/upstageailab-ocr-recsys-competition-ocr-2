# OCR Text Recognition & Layout Analysis System

| [![CI](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions) [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org) [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E.svg)](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18) |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|

**AI-optimized text-recognition system with layout analysis for accurate information extraction**
[English](README.md) • [한국어](README.ko.md)

---

## 📚 Table of Contents
[Features](#features) • [Project Compass](#project-compass-ai-navigation) • [Documentation](#documentation)

---

## About
This project originated from the **Upstage AI Bootcamp OCR competition** and has evolved into an end-to-end text recognition system with advanced layout analysis. It is currently undergoing final preparation and safety checks before major architectural upgrades.

**Repositories:**
- **Personal (Continuation):** [Wchoi189/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2)
- **Original (Bootcamp):** [AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)

---

## Features
- **Layout-Aware OCR**: Combines detection, recognition, and spatial analysis.
- **High-Performance ETL**: Converts 616k+ images to LMDB format in minutes.
- **Cloud-Native Scaling**: AWS Fargate integration for API rate-limiting bypass.

---

## Project Compass: AI Navigation
Centralized state management via **Model Context Protocol (MCP)**. Tools:
- `env_check` (validates environment)
- `session_init` (focuses session on goals)
- `reconcile` (syncs metadata with disk)
- `ocr_convert` (ETL pipeline to LMDB)
- `ocr_inspect` (validates LMDB integrity)

<details>
<summary>📂 Project Compass State</summary>

| Category | File | Description |
|----------|------|-------------|
| **🧠 Active Context** | [`current_session.yml`](project_compass/active_context/current_session.yml) | Current high-level objective and lock state. |
| **🗺️ Roadmap** | [`02_recognition.yml`](project_compass/roadmap/02_recognition.yml) | Plan for the Text Recognition phase. |
| **🤖 Agents** | [`AGENTS.yaml`](project_compass/AGENTS.yaml) | Registry of available tools and MCP commands. |

</details>

---

## OCR ETL Pipeline
- **Zero Clutter**: Converts images to a single LMDB file.
- **Resumable**: JSON state tracking for pause/resume.
- **Multiprocessing**: Optimized for RTX 3090 workstations.

**Performance**:
Processed 616,366 AI Hub samples in ~1 minute.

---

## AWS Batch Processor
Serverless AWS Fargate architecture for:
- Overcoming API rate limits
- Parallel document parsing
- S3 storage for intermediate results

---

## Quality Assurance
Structured bug reports for:
- Pipeline crashes
- Data integrity issues
- HTML contamination in receipt parsing

[📂 View Bug Reports](docs/artifacts/bug_reports/)

---

## Project Progress
| Phase | Status | Progress |
|-------|--------|----------|
| **Phase 1-4: Core Development** | ✅ Complete | 100% |
| **Phase 5: Pre-Upgrade Preparation** | 🔄 In Progress | 90% |
| **Phase 6: Architectural Upgrades** | ⏳ Planned | 0% |

**Overall: 85% Complete**
**Current Focus**: Training PARSeq/CRNN model on AI Hub LMDB dataset.

---

## Technical Stack
| Category | Technologies |
|----------|-------------|
| **ML/DL** | PyTorch, PyTorch Lightning, Hydra |
| **Backend** | FastAPI, ONNX Runtime |
| **Tools** | `uv` (Required), npm, W&B, Playwright, Vitest |

---

## Models & Performance
| Model Name | Architecture | H-Mean | Hugging Face |
|------------|--------------|--------|--------------|
| **Receipt Detection KR** | DBNet + PAN (ResNet18) | 95.37% | [🤗 Model Card](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18) |

---

## Documentation
- [System Architecture](AgentQMS/standards/tier1-sst/system-architecture.yaml)
- [API Contracts](AgentQMS/standards/tier2-framework/api-contracts.yaml)
- [File Placement Rules](AgentQMS/standards/tier1-sst/file-placement-rules.yaml)
- [Changelog](CHANGELOG.md)

---

## Contributing
Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License
MIT License - see [LICENSE](LICENSE) for details.

| [⬆ Back to Top](#ocr-text-recognition--layout-analysis-system) |
|:-------------------------------------------------------------:|
