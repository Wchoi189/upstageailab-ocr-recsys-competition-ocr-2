<div align="center">

[![CI](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E.svg)](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18)

# OCR Text Recognition & Layout Analysis System

**AI-optimized text-recognition system with layout analysis for accurate information extraction**

[English](README.md) • [한국어](README.ko.md)

[Features](#features) • [Project Compass](#project-compass-ai-navigation) • [Documentation](#documentation)

</div>

---

## About

This project originated from the Upstage AI Bootcamp OCR competition and has evolved into a personal continuation focused on building an end-to-end text recognition system with advanced layout analysis. Currently undergoing final preparation and safety checks before major architectural upgrades.

**Repositories:**
- **Personal (Continuation):** [Wchoi189/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2)
- **Original (Bootcamp):** [AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)

---

## Project Compass: AI Navigation

**Project Compass** is the central nervous system of this project, designed to help AI agents navigate, understand, and maintain the codebase without needing to perform archaeological digs through git history.

### MCP Integration
We have exposed Project Compass internals via **Model Context Protocol (MCP)**, allowing AI agents to directly interact with the project state.

**Available Tools:**
- `env_check`: Validates the `uv` environment, Python version, and CUDA status against the lock file.
- `session_init --objective [GOAL]`: Atomically updates the current session context to focus on a specific objective.
- `reconcile`: Performs a deep scan of experiment metadata and synchronizes the state (`manifest.json`) with the actual disk content.
- `ocr_convert`: **(New)** Launch the multi-threaded ETL pipeline to convert datasets to LMDB.
- `ocr_inspect`: **(New)** Verify the integrity of LMDB datasets.

<details>
<summary><strong>📂 Explore Project Compass State (Click to Expand)</strong></summary>

The **Project Compass** maintains the live state of the project. AI agents read these files to understand context, blockers, and goals.

| Category | File | Description |
|----------|------|-------------|
| **🧠 Active Context** | [`current_session.yml`](project_compass/active_context/current_session.yml) | Current high-level objective and lock state. |
| | [`blockers.yml`](project_compass/active_context/blockers.yml) | List of active impediments and dependency issues. |
| **🗺️ Roadmap** | [`02_recognition.yml`](project_compass/roadmap/02_recognition.yml) | Plan for the Text Recognition phase. |
| **🤖 Agents** | [`AGENTS.yaml`](project_compass/AGENTS.yaml) | Registry of available tools and MCP commands. |
| **💾 Data** | [`dataset_registry.yml`](project_compass/environments/dataset_registry.yml) | Single source of truth for dataset paths and formats. |
| **📜 History** | [`session_handover.md`](project_compass/session_handover.md) | The "Landing Page" for the next agent. |

</details>

---

## OCR ETL Pipeline & Data Processing

We have developed a standalone, high-performance data processing package `ocr-etl-pipeline` to handle massive datasets efficiently.

### Key Features
- **Zero Clutter**: Converts millions of raw image files into a single **LMDB (Lightning Memory-Mapped Database)** file.
- **Resumable**: Uses a JSON state file to track progress, allowing huge jobs to be paused and resumed without data loss.
- **Multiprocessing**: Optimized for RTX 3090 workstations, utilizing all available CPU cores for image decoding and cropping.

**Performance**:
- Processed 616,366 samples from AI Hub in ~1 minute.
- Reduced filesystem overhead significantly (1 file vs 600k files).

---

## Research Insights & Pivots

### Why We Abandoned KIE + Document Parse for Receipts

Our initial strategy involved using a Key Information Extraction (KIE) model combined with the Document Parse API. However, extensive testing revealed critical misalignment:

1.  **Fundamental Mismatch**: Receipts are primarily linear or semi-structured text streams. The Document Parse API treats documents as highly structured tables/forms.
2.  **HTML Contamination**: The API output for receipts often contained excessive HTML table tags that did not represent the visual layout, poisoning the embeddings.
3.  **LayoutLM Inefficiency**: LayoutLM models thrive on complex 2D spatial relationships (like forms). For receipt OCR, a robust **Text Recognition (PARSeq/CRNN)** model proved to be far more effective and less brittle.

**Result**: We pivoted to building a dedicated Text Recognition pipeline on the AI Hub dataset, abandoning the KIE approach for this specific domain.

---

## AWS Batch Processor: Cloud-Native Data Engineering

To overcome local resource constraints and API rate limits, we built `aws-batch-processor`, a standalone module for serverless batch processing.

**Problem**: The Document Parse API free tier enforces strict rate limits, and processing 5,000+ documents would stall the main workflow if run synchronously locally.
**Solution**: Offloaded processing to **AWS Fargate** using a serverless batch architecture. This allowed overnight processing without keeping the local machine online.

- **Architecture**: [View Diagram & Implementation Details](aws-batch-processor/README.md)
- **Data Catalog**: [`aws-batch-processor/data/export/data_catalog.yaml`](aws-batch-processor/data/export/data_catalog.yaml)
- **Script Catalog**: [`aws-batch-processor/script_catalog.yaml`](aws-batch-processor/script_catalog.yaml)

### Key Technologies
- **AWS Fargate**: Serverless compute for batch jobs.
- **S3**: Durable storage for intermediate and final results.
- **Parquet**: Columnar storage for efficient annotation querying.

---

## Quality Assurance & Bug Tracking

We maintain a rigorous quality standard by generating detailed, structured bug reports for major incidents. These artifacts serve as persistent learning resources.

[📂 **View Bug Report Collection**](docs/artifacts/bug_reports/)

**Example Artifacts:**
- **Critical Failures**: Documenting root causes of pipeline crashes.
- **Data Integrity**: tracking issues like "HTML contamination in receipt parsing".
- **Resolution**: Every report includes a verified fix strategy, preventing regression.

---

## Features

- **Perspective Correction**: High-reliability edge detection using binary mask outputs from Rembg.
- **Perspective Warping**: Geometric transformations to optimize the visibility of target regions.
- **Background Normalization**: Resolves detection failures caused by lighting variance and color casts in high-quality images.
- **Image Analysis**: Specialized VLM tools for automated image assessment and technical defect reporting.

---

## Experiment Tracker: Organized AI-Driven Research

**Problem Solved**: Rapid AI-driven experimentation often produces a high volume of artifacts, scripts, and documentation that require systematic organization to remain manageable.

**Solution**: `experiment_manager/` - A structured system for organizing experimental artifacts optimized for both human readability and AI consumption.

### Example of Standardized Technical Reports & Documentation

**Baseline Analysis**
- [Baseline Metrics Summary](experiment_manager/experiments/20251217_024343_image_enhancements_implementation/artifacts/20251218_1415_report_baseline-metrics-summary.md) - Comprehensive baseline metrics establishing performance benchmarks when comparing subtle improvements in quality

**Incident Resolution**
- [Data Loss Incident Report](experiment_manager/experiments/20251217_024343_image_enhancements_implementation/artifacts/20251220_0130_incident_report_perspective_correction_data_loss.md) - Critical data loss incident analysis and resolution strategy

---

## Project Progress

<div align="center">

| Phase | Status | Progress |
|-------|--------|----------|
| **Phase 1-4: Core Development** | Complete | 100% |
| **Phase 5: Pre-Upgrade Preparation** | In Progress | 90% |
| **Phase 6: Architectural Upgrades** | Planned | 0% |

**Overall: 85% Complete**

</div>

**Current Focus:** Training the Text Recognition Model (PARSeq/CRNN) using the newly generated AI Hub LMDB dataset.

---

## Tech Stack & Environment

**Strict Policy**: This project uses `uv` for all Python package management and execution.

| Category | Technologies |
|----------|-------------|
| **ML/DL** | PyTorch, PyTorch Lightning, Hydra |
| **Backend** | FastAPI, ONNX Runtime |
| **Tools** | **UV (Required)**, npm, W&B, Playwright, Vitest |
| **QMS** | AgentQMS (Artifacts, Standards, Compliance) |

---

## Model Zoo

| Model Name | Architecture | H-Mean | Hugging Face |
|------------|--------------|--------|--------------|
| **Receipt Detection KR** | DBNet + PAN (ResNet18) | 95.37% | [🤗 Model Card](https://huggingface.co/wchoi189/receipt-text-detection_kr-pan_resnet18) |

---

## Documentation

**AI-Facing Resources (AgentQMS Standards)**
- [System Architecture](AgentQMS/standards/tier1-sst/system-architecture.yaml)
- [API Contracts](AgentQMS/standards/tier2-framework/api-contracts.yaml)
- [AgentQMS Workflows](AgentQMS/knowledge/agent/system.md)

**Reference**
- [File Placement Rules](AgentQMS/standards/tier1-sst/file-placement-rules.yaml)
- [Changelog](CHANGELOG.md)

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

<div align="center">

[⬆ Back to Top](#ocr-text-recognition--layout-analysis-system)

</div>
