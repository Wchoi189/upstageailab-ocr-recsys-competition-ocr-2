<!-- Badges -->
<div align="center">

<!-- TODO: Update CI badge URL when repository is migrated -->
[![CI](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/actions)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org)
[![UV](https://img.shields.io/badge/UV-0.8+-purple.svg)](https://github.com/astral-sh/uv)
[![Hydra](https://img.shields.io/badge/Hydra-1.3+-green.svg)](https://hydra.cc)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.1+-orange.svg)](https://lightning.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen.svg)](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)
[![Maintenance](https://img.shields.io/badge/Maintained-yes-green.svg)](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2)

</div>

<div align="center">

<!-- TODO: Add project logo/banner image here -->
<!--
![Project Logo](docs/assets/images/logo.png)
-->

# OCR Text Detection & Recognition System

**A modular, production-ready OCR system for receipt text detection and recognition**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Roadmap](#-roadmap) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

This project is a comprehensive OCR (Optical Character Recognition) system designed for detecting and recognizing text in receipt images. Built with a modular architecture using PyTorch Lightning and Hydra, it provides a flexible framework for text detection with plans to extend to text recognition and layout analysis.

### Key Capabilities

- **Text Detection**: Accurate polygon-based text region detection using DBNet architecture
- **Modular Architecture**: Plug-and-play components (encoders, decoders, heads, loss functions)
- **Production-Ready**: Offline preprocessing pipeline for 5-8x faster training/validation
- **Interactive UI**: Streamlit-based tools for model testing, preprocessing demos, and evaluation
- **Comprehensive Tooling**: Command builder, evaluation viewer, and real-time inference interface

---

## 🚀 Quick Links

**Getting Started**
- [Installation Guide](#-quick-start)
- [First Steps Tutorial](#-getting-started-tutorial)
- [Configuration Guide](docs/architecture-overview.md)

**Development**
- [Contributing Guidelines](CONTRIBUTING.md)
- [Development Setup](#-development)
- [Code Standards](docs/development/coding-standards.md)

**Resources**
- [API Reference](docs/api-reference.md)
- [Changelog](docs/CHANGELOG.md)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)

---

## 📊 Project Status

<div align="center">

| Phase | Status | Progress |
|-------|--------|----------|
| **Phase 1: Performance & Maintainability** | 🟡 In Progress | 40% |
| **Phase 2: Preprocessing Enhancement** | ⚪ Planned | 0% |
| **Phase 3: Text Recognition** | ⚪ Planned | 0% |
| **Phase 4: Layout Recognition** | ⚪ Planned | 0% |
| **Phase 5: Frontend & CI/CD** | ⚪ Planned | 0% |

**Overall Progress: 8%**

</div>

---

## ✨ Features

### Current Features

- ✅ **DBNet-based Text Detection**: Real-time scene text detection with differentiable binarization
- ✅ **Modular Component System**: Registry-based architecture for easy experimentation
- ✅ **Offline Preprocessing**: Pre-computed probability and threshold maps for faster training
- ✅ **Streamlit UI Tools**: Command builder, evaluation viewer, and inference interface
- ✅ **Hydra Configuration**: Centralized configuration management
- ✅ **W&B Integration**: Experiment tracking and monitoring
- ✅ **Performance Profiling**: Advanced performance analysis tools

### Planned Improvements

This project is actively being improved with focus on:

1. **🚀 System Performance Optimization**
   - Identify and resolve performance bottlenecks
   - Optimize training and inference pipelines
   - Memory optimization and efficient resource utilization

2. **🔧 Enhanced Maintainability**
   - Code refactoring and documentation improvements
   - Better error handling and logging
   - Comprehensive test coverage

3. **🔄 Enhanced Preprocessing Pipeline**
   - Advanced image enhancement techniques
   - More robust augmentation strategies
   - Preprocessing pipeline optimization

4. **📝 Text Recognition Capabilities**
   - Integration of text recognition models
   - End-to-end OCR pipeline (detection + recognition)
   - Support for multiple languages

5. **📐 Layout Recognition Capabilities**
   - Document structure analysis
   - Region classification (header, body, footer, etc.)
   - Table and form detection

6. **🎨 Frontend UI for Testing**
   - Interactive model testing interface
   - Preprocessing pipeline demos
   - Real-time visualization tools
   - Model comparison dashboard

7. **🔄 CI/CD Integration**
   - Automated testing pipelines
   - Continuous integration workflows
   - Automated deployment processes
   - Code quality checks

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology |
|----------|-----------|
| **Deep Learning** | PyTorch, PyTorch Lightning |
| **Configuration** | Hydra |
| **Package Manager** | UV |
| **UI Framework** | Streamlit |
| **Experiment Tracking** | Weights & Biases |
| **Testing** | pytest |
| **Documentation** | Markdown, Sphinx (planned) |

</div>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- UV package manager
- CUDA-compatible GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd upstageailab-ocr-recsys-competition-ocr-2

# Setup environment (automated)
./scripts/setup/00_setup-environment.sh
```

### Basic Usage

```bash
# Run unit tests
uv run pytest tests/ -v

# Train model
uv run python runners/train.py preset=example trainer.max_epochs=10

# Test model
uv run python runners/test.py preset=example checkpoint_path="outputs/ocr_training/checkpoints/best.ckpt"

# Generate predictions
uv run python runners/predict.py preset=example checkpoint_path="outputs/ocr_training/checkpoints/best.ckpt"
```

### UI Tools

```bash
# Command Builder - Build and execute training commands
uv run streamlit run ui/command_builder.py

# Real-time Inference - Test models interactively
uv run streamlit run ui/inference_ui.py

# Evaluation Viewer - Visualize and analyze results
uv run streamlit run ui/evaluation_viewer.py
```

---

## 📚 Getting Started Tutorial

### Step 1: Environment Setup

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo-url>
cd upstageailab-ocr-recsys-competition-ocr-2
./scripts/setup/00_setup-environment.sh
```

### Step 2: Verify Installation

```bash
# Run tests to verify everything works
uv run pytest tests/ -v

# Check if GPU is available
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Prepare Data

```bash
# Place your dataset in the data/datasets/ directory
# Structure should be:
# data/datasets/
#   ├── images/
#   │   ├── train/
#   │   ├── val/
#   │   └── test/
#   └── jsons/
#       ├── train.json
#       ├── val.json
#       └── test.json
```

### Step 4: Preprocess Data (Optional but Recommended)

```bash
# Preprocess maps for faster training
uv run python scripts/preprocess_maps.py
```

### Step 5: Train Your First Model

```bash
# Start with a small experiment
uv run python runners/train.py \
    preset=example \
    trainer.max_epochs=5 \
    data.train_num_samples=100 \
    data.val_num_samples=20
```

### Step 6: Test and Evaluate

```bash
# Test the trained model
uv run python runners/test.py \
    preset=example \
    checkpoint_path="outputs/ocr_training/checkpoints/latest.ckpt"

# Generate predictions
uv run python runners/predict.py \
    preset=example \
    checkpoint_path="outputs/ocr_training/checkpoints/latest.ckpt"
```

### Step 7: Explore with UI

```bash
# Launch the inference UI to test interactively
uv run streamlit run ui/inference_ui.py
```

**Next Steps**: Check out the [Documentation](#-documentation) section for more advanced usage.

---

## 📁 Project Structure

```
├── configs/              # Hydra configuration files
│   ├── train.yaml
│   ├── test.yaml
│   ├── predict.yaml
│   └── preset/          # Model and dataset presets
├── data/                # Dataset directory
│   ├── datasets/
│   └── jsons/
├── docs/                # Documentation
│   ├── agents/          # AI agent instructions
│   ├── maintainers/     # Maintainer documentation
│   └── ...
├── ocr/                 # Core OCR modules
│   ├── datasets/
│   ├── lightning_modules/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── runners/             # Training, testing, prediction scripts
├── scripts/             # Utility scripts
│   ├── agent_tools/
│   └── setup/
├── ui/                  # Streamlit UI applications
│   ├── command_builder.py
│   ├── inference_ui.py
│   └── evaluation_viewer.py
└── tests/               # Unit and integration tests
```

---

## 🏗️ Architecture

### Modular Design

The system uses a registry-based architecture that allows plug-and-play component replacement:

- **Encoders**: MobileNetV3, ResNet, EfficientNet (planned)
- **Decoders**: PAN, DBNet++
- **Heads**: Custom detection heads
- **Loss Functions**: Various loss implementations

### DBNet Architecture

The core detection model is based on DBNet (Differentiable Binarization Network), which provides:

- Real-time text detection
- Accurate polygon-based text region localization
- Differentiable binarization for end-to-end training

![DBNet Architecture](docs/assets/images/banner/flow-chart-of-the-dbnet.png)

---

## 📊 Performance

### Current Model Performance

- **H-Mean**: 0.9787
- **Precision**: 0.9771
- **Recall**: 0.9809
- **Training Time**: ~22 minutes (10 epochs on V100 GPU)

### Evaluation Metrics

The project uses **CLEval** (Character-Level Evaluation) for text detection assessment:

![CLEval](https://github.com/clovaai/CLEval/raw/master/resources/screenshots/explanation.gif)

### Example Results

<!-- TODO: Add example output images/results here -->
<!--
<div align="center">

**Example Detection Results**

| Input Image | Detection Result | Recognition Result |
|------------|------------------|-------------------|
| ![Input](docs/assets/images/examples/input_1.jpg) | ![Detection](docs/assets/images/examples/detection_1.jpg) | Text: "Receipt content..." |
| ![Input](docs/assets/images/examples/input_2.jpg) | ![Detection](docs/assets/images/examples/detection_2.jpg) | Text: "Receipt content..." |

</div>
-->

---

## 🔄 Preprocessing Pipeline

### Offline Preprocessing

The project includes an offline preprocessing system that pre-computes probability and threshold maps, resulting in **5-8x faster validation speed**.

```bash
# Preprocess entire dataset
uv run python scripts/preprocess_maps.py

# Test with limited samples
uv run python scripts/preprocess_maps.py data.train_num_samples=100 data.val_num_samples=20
```

### Features

- Pre-computed probability and threshold maps
- Automatic fallback to real-time generation if maps are missing
- Compressed `.npz` format for efficient storage
- Image enhancement using Doctr library
- CamScanner-style preprocessing

---

## 🎨 Features Showcase

### UI Tools

<div align="center">

<table style="border-collapse: collapse; border: none;">
<tr>
<td style="border: none; padding: 20px; text-align: center; vertical-align: top;">
<strong>Command Builder</strong><br><br>
<img src="docs/assets/images/demo/command-builder-predict-command-generate.png" alt="Command Builder Interface" style="width: 800px; height: 500px; object-fit: contain; border: 1px solid #ddd; border-radius: 8px;"><br><br>
<small>Build and execute training commands with an intuitive interface</small>
</td>
</tr>
<tr>
<td style="border: none; padding: 20px; text-align: center; vertical-align: top;">
<strong>Real-time OCR Inference</strong><br><br>
<img src="docs/assets/images/demo/real-time-ocr-inference-select-img.png.jpg" alt="Real-time OCR Inference Interface" style="width: 800px; height: 500px; object-fit: contain; border: 1px solid #ddd; border-radius: 8px;"><br><br>
<small>Interactive interface for real-time text detection on receipt images</small>
</td>
</tr>
<tr>
<td style="border: none; padding: 20px; text-align: center; vertical-align: top;">
<strong>Evaluation Viewer</strong><br><br>
<img src="docs/assets/images/demo/ocr-eval-results-viewer-gallery.png" alt="Evaluation Viewer Interface" style="width: 800px; height: 500px; object-fit: contain; border: 1px solid #ddd; border-radius: 8px;"><br><br>
<small>Visualize and analyze model evaluation results in detail</small>
</td>
</tr>
</table>

</div>

<!-- TODO: Add animated GIFs showcasing key features -->
<!--
### Animated Demos

<div align="center">

**Interactive Model Testing**

![Model Testing Demo](docs/assets/images/demos/model-testing-demo.gif)

**Preprocessing Pipeline Demo**

![Preprocessing Demo](docs/assets/images/demos/preprocessing-demo.gif)

</div>
-->

---

## 🛠️ Development

### Environment Setup

**Important**: This project uses **UV** package manager. Do not use pip, conda, or poetry.

```bash
# All commands must use 'uv run' prefix
uv run python runners/train.py
uv run pytest tests/
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_metrics.py

# Run with coverage
uv run pytest tests/ --cov=ocr
```

### Code Quality

- Follow PEP 8 style guidelines
- Use type hints where applicable
- Write comprehensive docstrings
- Maintain test coverage above 80%

### Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `uv run pytest tests/`
4. Check code quality: `uv run ruff check .`
5. Commit with clear messages
6. Push and create a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 🗺️ Roadmap

### Phase 1: Performance & Maintainability (Current) 🟡
- [ ] System performance profiling and optimization
- [ ] Code refactoring and documentation
- [ ] Enhanced error handling and logging
- [ ] Comprehensive test coverage

### Phase 2: Preprocessing Enhancement ⚪
- [ ] Advanced image enhancement techniques
- [ ] Robust augmentation strategies
- [ ] Preprocessing pipeline optimization

### Phase 3: Text Recognition ⚪
- [ ] Text recognition model integration
- [ ] End-to-end OCR pipeline
- [ ] Multi-language support

### Phase 4: Layout Recognition ⚪
- [ ] Document structure analysis
- [ ] Region classification
- [ ] Table and form detection

### Phase 5: Frontend & CI/CD ⚪
- [ ] Enhanced UI for model testing
- [ ] Preprocessing pipeline demos
- [ ] CI/CD pipeline implementation
- [ ] Automated deployment

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: CUDA out of memory
**Solution**: Reduce batch size in config or use gradient accumulation
```bash
uv run python runners/train.py preset=example data.batch_size=4 trainer.accumulate_grad_batches=2
```

#### Issue: Preprocessing maps not found
**Solution**: Either run preprocessing or the system will automatically fallback to real-time generation (slower)
```bash
uv run python scripts/preprocess_maps.py
```

#### Issue: UV command not found
**Solution**: Install UV package manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Issue: Import errors after setup
**Solution**: Ensure you're using `uv run` prefix for all Python commands
```bash
# ❌ Wrong
python runners/train.py

# ✅ Correct
uv run python runners/train.py
```

### Getting Help

- Check [Documentation](#-documentation) for detailed guides
- Search [GitHub Issues](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/issues)
- Create a new issue with:
  - Error messages
  - Steps to reproduce
  - Environment details (OS, Python version, GPU)

---

## ❓ FAQ

<!-- TODO: Add frequently asked questions -->
<!--
### General Questions

**Q: What is the minimum GPU memory required?**
A: A GPU with at least 8GB VRAM is recommended for training. Inference can run on smaller GPUs or CPU.

**Q: Can I use this for other document types besides receipts?**
A: Yes, the model can be fine-tuned for other document types, though it's currently optimized for receipts.

**Q: How do I add support for a new language?**
A: [Instructions for adding language support]

### Technical Questions

**Q: Why use UV instead of pip?**
A: UV is significantly faster for dependency resolution and installation, improving development workflow.

**Q: How do I customize the model architecture?**
A: [Instructions for customizing architecture]
-->

*More FAQs coming soon. Feel free to [open an issue](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/issues) with your question!*

---

## 📚 Documentation

- [Architecture Overview](docs/architecture-overview.md) - System architecture and design decisions
- [Process Management Guide](docs/process-management-guide.md) - Training process management
- [Component Diagrams](docs/component-diagrams.md) - Visual component documentation
- [API Reference](docs/api-reference.md) - API documentation
- [Development Guide](docs/development/) - Coding standards and conventions
- [Data Contracts](docs/pipeline/data_contracts.md) - Data pipeline specifications
- [Changelog](docs/CHANGELOG.md) - Project changelog and version history

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code of conduct
- How to submit pull requests
- Development setup
- Coding standards
- Issue reporting

**Quick Contribution Checklist:**
- [ ] Fork the repository
- [ ] Create a feature branch
- [ ] Make your changes
- [ ] Add tests
- [ ] Update documentation
- [ ] Submit a pull request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [DBNet](https://github.com/MhLiao/DB) - Differentiable Binarization Network
- [CLEval](https://github.com/clovaai/CLEval) - Character-Level Evaluation
- [PyTorch Lightning](https://lightning.ai) - Deep learning framework
- [Hydra](https://hydra.cc) - Configuration management
- [UV](https://github.com/astral-sh/uv) - Fast Python package manager

---

## 📖 References

### Papers
- **CLEval**: Character-Level Evaluation for Text Detection and Recognition Tasks
  [arXiv:2006.06244](https://arxiv.org/pdf/2006.06244.pdf)
- **DBNet**: Real-time Scene Text Detection with Differentiable Binarization
  [AAAI 2020](https://arxiv.org/abs/1911.08947)

### Resources
- [DBNet Repository](https://github.com/MhLiao/DB)
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [PyTorch Lightning Docs](https://pytorch-lightning.readthedocs.io/)
- [UV Documentation](https://github.com/astral-sh/uv)

---

## 📧 Contact

For questions, suggestions, or contributions:

- **Issues**: [GitHub Issues](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/discussions) *(if enabled)*

---

<div align="center">

**Built with ❤️ for OCR research and development**

[⬆ Back to Top](#-overview)

</div>
