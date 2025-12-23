<div align="center">

[![CI](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E.svg)](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18)

# OCR Text Recognition & Layout Analysis System

**AI-optimized text-recognition system with layout analysis for accurate information extraction**

[English](README.md) • [한국어](README.ko.md)

[Features](#features) • [Progress](#project-progress) • [Documentation](#documentation)

</div>

---

## About

This project originated from the Upstage AI Bootcamp OCR competition and has evolved into a personal continuation focused on building an end-to-end text recognition system with advanced layout analysis. Currently undergoing final preparation and safety checks before major architectural upgrades.

**Repositories:**
- **Personal (Continuation):** [Wchoi189/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2)
- **Original (Bootcamp):** [AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)

---

## Features

- **Perspective Correction**: High-reliability edge detection using binary mask outputs from Rembg.
- **Perspective Warping**: Geometric transformations to optimize the visibility of target regions.
- **Background Normalization**: Resolves detection failures caused by lighting variance and color casts in high-quality images.
- **Image Analysis**: Specialized VLM tools for automated image assessment and technical defect reporting.

---
## OCR Inference Console

The OCR Inference Console is a proof-of-concept frontend for the OCR web service. It provides a streamlined interface for document preview and structured output analysis.

<div align="center">
  <a href="docs/assets/images/demo/my-app.webp">
    <img src="docs/assets/images/demo/my-app.webp" alt="OCR Inference Console" width="800px" />
  </a>
  <p><em>OCR Inference Console: Three-panel layout featuring document preview, layout analysis, and structured JSON output. (Click to enlarge)</em></p>
</div>

### UX Attribution
The user interface design is inspired by the **Upstage Document OCR Console**. The layout patterns, including the three-panel console with document preview and structured output, follow the interaction models established by Upstage's product suite.

All code and implementations in this repository are based on the Upstage OCR RecSys competition baseline. Key contributions include modernizing configurations, improve performance, and enhancing the development workflow.

Original: https://console.upstage.ai/playground/document-ocr

---
## Experiment Tracker: Organized AI-Driven Research

**Problem Solved**: Rapid AI-driven experimentation often produces a high volume of artifacts, scripts, and documentation that require systematic organization to remain manageable. Traditional project structures fail when experiments iterate daily and debugging requires instant access to reliable documentation.

**Solution**: `experiment-tracker/` - A structured system for organizing experimental artifacts optimized for both human readability and AI consumption. Provides standardized protocols for common workflows and output format for artifacts.

### Example of Standardized Technical Reports & Documentation

**Baseline Analysis**
- [Baseline Metrics Summary](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/artifacts/20251218_1415_report_baseline-metrics-summary.md) - Comprehensive baseline metrics establishing performance benchmarks when comparing subtle improvements in quality

**Incident Resolution**
- [Data Loss Incident Report](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/artifacts/20251220_0130_incident_report_perspective_correction_data_loss.md) - Critical data loss incident analysis and resolution strategy

**Comparative Analysis**
- [Background Normalization Comparison](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/.metadata/reports/20251218_1458_report_background-normalization-comparison.md) - Background normalization strategy comparison with quantitative results

### Visual Results & Demos

<div align="center">

| Fitted Corners | Corrected Output |
| :---: | :---: |
| [<img src="docs/assets/images/demo/original-with-fitted-corners.webp" width="700px" />](docs/assets/images/demo/original-with-fitted-corners.webp) | [<img src="experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/drp.en_ko.in_house.selectstar_000712_step2_corrected.jpg" width="250px" />](experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/outputs/full_pipeline_correct/drp.en_ko.in_house.selectstar_000712_step2_corrected.jpg) |
| *Corner detection and geometric fitting* | *Final perspective-corrected output* |

*(Click images to enlarge)*

</div>

### Key Benefits

- **AI-Optimized**: Documentation structure designed for efficient AI consumption.
- **Standardized Protocols**: Reduces manual prompting and produces high-quality results.
- **Traceability**: Full reproduction path for all experimental results.
- **Scalable Organization**: Isolated experiment artifacts to prevent context chaos.

---
## Low Prediction Resolution

<div align="center">

| Before: Persistent Low Predictions | Internal Process | After: Successful Detection |
| :---: | :---: | :---: |
| [<img src="docs/assets/images/demo/inference-persistent-empties-before.webp" width="250px" />](docs/assets/images/demo/inference-persistent-empties-before.webp) | [<img src="docs/assets/images/demo/inference-persistent-empties-after.webp" width="250px" />](docs/assets/images/demo/inference-persistent-empties-after.webp) | [<img src="docs/assets/images/demo/inference-persistent-empties-after2.webp" width="250px" />](docs/assets/images/demo/inference-persistent-empties-after2.webp) |
| *Empty patches* | *Filter application* | *Normalized geometry* |

*(Click images to enlarge)*

</div>

---
## Project Progress

<div align="center">

| Phase | Status | Progress |
|-------|--------|----------|
| **Phase 1-4: Core Development** | Complete | 100% |
| **Phase 5: Pre-Upgrade Preparation** | In Progress | 80% |
| **Phase 6: Architectural Upgrades** | Planned | 0% |

**Overall: 80% Complete**

</div>

**Current Focus:** Final safety checks, system validation, and preparation for major architectural enhancements.

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/DL** | PyTorch, PyTorch Lightning, Hydra |
| **Backend** | FastAPI, ONNX Runtime |
| **Frontend** | React 19, Next.js 16, Chakra UI, Streamlit |
| **Tools** | UV (Python), npm, W&B, Playwright, Vitest |

---

## Model Zoo

| Model Name | Architecture | H-Mean | Hugging Face |
|------------|--------------|--------|--------------|
| **Receipt Detection KR** | DBNet + PAN (ResNet18) | 95.37% | [🤗 Model Card](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18) |

---

## Documentation

**AI-Facing Resources (.ai-instructions)**
- [System Architecture](.ai-instructions/tier1-sst/system-architecture.yaml)
- [API Contracts](.ai-instructions/tier2-framework/api-contracts.yaml)
- [AgentQMS Workflows](AgentQMS/knowledge/agent/system.md)

**Reference**
- [File Placement Rules](.ai-instructions/tier1-sst/file-placement-rules.yaml)
- [Changelog](CHANGELOG.md)

---

## Project Structure

```
├── AgentQMS/          # AI documentation and quality management
├── apps/              # Frontend & backend applications
├── configs/           # Hydra configuration (89 YAML files)
├── docs/              # AI-optimized documentation & artifacts
├── ocr/               # Core OCR Python package
├── runners/           # Training/testing/prediction scripts
├── scripts/           # Utility scripts
├── tests/             # Unit & integration tests
```

Detailed structure: [.ai-instructions/tier1-sst/file-placement-rules.yaml](.ai-instructions/tier1-sst/file-placement-rules.yaml)

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

[⬆ Back to Top](#ocr-text-recognition--layout-analysis-system)

</div>
