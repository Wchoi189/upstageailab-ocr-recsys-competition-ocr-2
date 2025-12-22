<div align="center">

[![CI](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/Wchoi189/upstageailab-ocr-recsys-competition-ocr-2/actions)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# OCR Text Detection & Recognition System

**Modular, production-ready OCR for receipt text detection and recognition**

[Quick Start](#-quick-start) • [Features](#-features) • [Documentation](#-documentation) • [Progress](#-project-progress)

</div>

---

## 📖 About

A comprehensive OCR system for detecting and recognizing text in receipt images. Built with PyTorch Lightning and Hydra for modularity and production readiness.

**Key Features:**
- 🎯 DBNet-based text detection with 97.8% H-Mean
- ⚡ 5-8x faster training with offline preprocessing
- 🧩 Modular architecture (plug-and-play components)
- 🎨 Interactive UI tools (Streamlit + React + Next.js)
- 📊 W&B integration for experiment tracking

---

## 🚀 Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd upstageailab-ocr-recsys-competition-ocr-2
./scripts/setup/00_setup-environment.sh

# Train a model
uv run python runners/train.py model/presets=model_example trainer.max_epochs=10

# Run inference UI
python run_ui.py inference
```

**Prerequisites:** Python 3.11+, UV package manager, CUDA GPU (recommended)

📘 **Detailed guides:** [Installation](docs/guides/installation.md) • [Training](docs/guides/training.md) • [Configuration](docs/architecture/CONFIG_ARCHITECTURE.md)

---

## ✨ Features

<div align="center">

| **Command Builder** | **Real-time Inference** | **Evaluation Viewer** |
|:---:|:---:|:---:|
| ![Command Builder](docs/assets/images/demo/command-builder-predict-command-generate.png) | ![Inference](docs/assets/images/demo/real-time-ocr-inference-select-img.png.jpg) | ![Evaluation](docs/assets/images/demo/ocr-eval-results-viewer-gallery.png) |
| Build training commands | Test models interactively | Analyze results visually |

</div>

### Current Capabilities

✅ **Text Detection** - DBNet architecture with polygon outputs
✅ **Offline Preprocessing** - Pre-computed maps for 5-8x speedup
✅ **Modular Components** - Registry-based encoders, decoders, heads, losses
✅ **Modular Inference Engine** - 8-component orchestrator pattern with 67% code reduction
✅ **Modern UIs** - Streamlit tools + React SPA + Next.js console
✅ **FastAPI Backend** - Inference API with job tracking

### Planned Features

🔜 **Text Recognition** - End-to-end OCR pipeline
🔜 **Layout Analysis** - Document structure understanding
🔜 **Multi-language Support** - Beyond English receipts

---

## 📊 Project Progress

<div align="center">

| Phase | Status | Progress |
|-------|--------|----------|
| **Phase 1-3: Core Features** | ✅ Complete | 100% |
| **Phase 4: Testing & QA** | 🟡 In Progress | 40% |
| **Phase 5: Next.js Migration** | 🟡 In Progress | 75% |
| **Phase 6-7: Future Work** | ⚪ Planned | 0% |

**Overall: 55% Complete**

</div>

### Recent Highlights

- ✅ Config architecture consolidation (43% cognitive load reduction)
- ✅ Client-side background removal with ONNX.js
- ✅ FastAPI backend with real inference API
- ✅ Next.js console with Chakra UI

**Current Focus:** E2E testing, Next.js API routes, analytics integration

📋 **Detailed roadmap:** [docs/roadmap.md](docs/roadmap.md)

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/DL** | PyTorch, PyTorch Lightning, Hydra |
| **Backend** | FastAPI, ONNX Runtime |
| **Frontend** | React 19, Next.js 16, Chakra UI, Streamlit |
| **Tools** | UV (Python), npm, W&B, Playwright, Vitest |

---

## 📚 Documentation

**Getting Started**
- [Installation Guide](docs/guides/installation.md)
- [Training Guide](docs/guides/training.md)
- [Configuration Guide](docs/architecture/CONFIG_ARCHITECTURE.md)

**Development**
- [Architecture Overview](docs/architecture/architecture.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [AgentQMS Workflows](AgentQMS/knowledge/agent/system.md)

**Reference**
- [API Documentation](docs/api-reference.md)
- [Changelog](CHANGELOG.md)
- [Troubleshooting](docs/guides/troubleshooting.md)

---

## 🏗️ Project Structure

```
├── apps/              # Frontend & backend applications
├── configs/           # Hydra configuration (89 YAML files)
├── docs/              # Documentation & artifacts
├── ocr/               # Core OCR Python package
├── runners/           # Training/testing/prediction scripts
├── scripts/           # Utility scripts
├── tests/             # Unit & integration tests
└── ui/                # Streamlit UI applications
```

📖 **Detailed structure:** [docs/architecture/project-structure.md](docs/architecture/project-structure.md)

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick checklist:**
- Fork & create feature branch
- Add tests for new features
- Update documentation
- Submit pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [DBNet](https://github.com/MhLiao/DB) - Text detection architecture
- [CLEval](https://github.com/clovaai/CLEval) - Evaluation metrics
- [PyTorch Lightning](https://lightning.ai) - Training framework
- [Hydra](https://hydra.cc) - Configuration management

---

<div align="center">

**Built with ❤️ for OCR research and development**

[⬆ Back to Top](#ocr-text-detection--recognition-system)

</div>
