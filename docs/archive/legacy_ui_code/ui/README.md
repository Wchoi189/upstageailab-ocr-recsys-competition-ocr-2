# ⚠️ DEPRECATED

This directory (`ui/`) is deprecated and will be removed in a future update.
The functionality is being migrated to `apps/frontend` and `apps/playground-console`.

Please do not add new features here.

# OCR Project Streamlit UI

This directory contains Streamlit applications for managing OCR training workflows and real-time inference.

## ⚠️ Legacy Files Warning

**IMPORTANT**: Some files in this directory are **deprecated wrappers** that will be removed in future versions. These files exist only for backward compatibility and should **NOT** be updated.

### Deprecated Wrapper Files (DO NOT UPDATE)

- ⚠️ **`ui/command_builder.py`** - Deprecated wrapper for `ui/apps/command_builder/app.py`
- ⚠️ **`ui/inference_ui.py`** - Deprecated wrapper for `ui/apps/inference/app.py`
- ⚠️ **`ui/evaluation_viewer.py`** - Deprecated wrapper for `ui/evaluation/app.py`

**For new code**: Always import directly from the actual implementations:
- Use `ui.apps.command_builder` instead of `ui.command_builder`
- Use `ui.apps.inference` instead of `ui.inference_ui`
- Use `ui.evaluation` instead of `ui.evaluation_viewer`

**Migration Guide**: See [Architecture](#architecture) section below for details.

## Table of Contents

- [Applications](#applications)
    - [Command Builder (`command_builder.py`)](#command-builder-command_builderpy)
        - [Features](#features)
        - [Process Safety](#process-safety)
        - [Usage](#usage)
    - [Inference UI (`inference_ui.py`)](#inference-ui-inference_uipy)
        - [Features](#features-1)
        - [Setup Requirements](#setup-requirements)
        - [Usage](#usage-1)
        - [Inference Workflow](#inference-workflow)
        - [Demo Mode](#demo-mode)
    - [Evaluation Viewer (`evaluation_viewer.py`)](#evaluation-viewer-evaluation_viewerpy)
        - [주요 기능](#주요-기능)
        - [사용법](#사용법)
        - [분석 기능](#분석-기능)
- [Applications](#applications-1)
    - [Command Builder (`command_builder.py`)](#command-builder-command_builderpy-1)
    - [Evaluation Viewer (`evaluation_viewer.py`)](#evaluation-viewer-evaluation_viewerpy-1)
    - [Resource Monitor (`resource_monitor.py`)](#resource-monitor-resource_monitorpy)
        - [Features](#features-2)
        - [Process Management](#process-management)
        - [Usage](#usage-2)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Development](#development)
- [Future Enhancements](#future-enhancements)

## Applications

### Command Builder ⚠️ DEPRECATED WRAPPER

> **⚠️ WARNING**: `ui/command_builder.py` is a deprecated wrapper. Use `ui/apps/command_builder/app.py` instead.

A user-friendly interface for building and executing training, testing, and prediction commands.

**Features:**
- Interactive model architecture selection (encoders, decoders, heads, losses)
- Training parameter adjustment (learning rate, batch size, epochs)
- Experiment configuration (W&B integration, checkpoint resuming)
- Real-time command validation and preview
- One-click command execution with progress monitoring
- **Improved process management** - Safe process group handling prevents orphaned training processes

**Process Safety:**
- Uses process groups for complete cleanup on interruption
- Automatic termination of DataLoader worker processes
- Graceful shutdown handling for interrupted training sessions
- Integration with process monitoring utilities

**Usage:**
```bash
# Run the command builder UI (recommended)
python run_ui.py command_builder

# Or directly with streamlit (deprecated - use run_ui.py instead)
uv run streamlit run ui/apps/command_builder/app.py
```

### Inference UI ⚠️ DEPRECATED WRAPPER

> **⚠️ WARNING**: `ui/inference_ui.py` is a deprecated wrapper. Use `ui/apps/inference/app.py` instead.

Real-time OCR inference interface for instant predictions on uploaded images.

**Features:**
- Drag-and-drop image upload (supports JPG, PNG, BMP)
- **Automatic checkpoint catalog discovery** - Scans all trained model checkpoints
- Model checkpoint selection from trained models with metadata display
- Real-time inference with progress tracking
- Interactive visualization of OCR predictions
- Batch processing for multiple images
- Demo mode with mock predictions when models aren't available

**Checkpoint Catalog:**
The inference UI automatically discovers and catalogs all trained model checkpoints from the `outputs/` directory. Each checkpoint entry includes:
- Experiment name and training configuration
- Model architecture (encoder/decoder/head)
- Training metrics (epoch, step, validation loss)
- Timestamp and file size information
- Compatibility validation for inference

**Setup Requirements:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # or use your preferred method

# Install dependencies
uv sync

# For real inference (optional - demo mode works without this)
# Train a model first using the command builder UI
```

**Usage:**
```bash
# Run the inference UI (recommended)
python run_ui.py inference

# Or directly with streamlit (deprecated - use run_ui.py instead)
uv run streamlit run ui/apps/inference/app.py
```

**Inference Workflow:**
1. Upload one or more images via drag-and-drop
2. **Browse available checkpoints** in the catalog with metadata preview
3. Select a trained model checkpoint (or use demo mode)
4. Click "Run Inference" for instant results
5. View predictions overlaid on images
6. Extract and copy recognized text

**Demo Mode:**
If no trained models are available, the UI automatically switches to demo mode with mock predictions, allowing you to test the interface and workflow before training models.

### Evaluation Viewer

> **⚠️ WARNING**: `ui/evaluation_viewer.py` is a deprecated wrapper. Use `ui/evaluation/app.py` instead.

A modular interface for viewing and analyzing OCR evaluation results.

**Architecture:**
- `ui/evaluation/app.py` - Main application entry point
- `ui/evaluation/single_run.py` - Single model analysis view
- `ui/evaluation/comparison.py` - Model comparison view
- `ui/evaluation/gallery.py` - Image gallery with filtering
- `ui/evaluation/__init__.py` - Package initialization

**주요 기능:**
- 예측 결과 CSV 파일 로드 및 분석
- 데이터셋 통계 및 분포 차트 표시
- 예측 분석 (바운딩 박스 면적, 종횡비 등)
- 이미지별 예측 결과 시각화 (바운딩 박스 오버레이)
- 모델 간 비교 및 차이 분석
- 이미지 갤러리 with 필터링 (높은 신뢰도, 낮은 신뢰도 등)
- 대화형 차트 및 통계 테이블

**사용법:**
```bash
# 평가 결과 뷰어 실행 (recommended)
python run_ui.py evaluation_viewer

# Or directly with streamlit (deprecated - use run_ui.py instead)
uv run streamlit run ui/evaluation/app.py

# 데모 실행
python demo_evaluation_viewer.py
```

**분석 기능:**
- 전체 데이터셋 통계 (이미지 수, 예측 수, 평균 예측/이미지)
- 예측 분포 히스토그램
- 바운딩 박스 면적 및 종횡비 분석
- 개별 이미지 예측 결과 시각화

## Applications (Continued)

### Command Builder ⚠️ DEPRECATED WRAPPER

> **⚠️ WARNING**: `ui/command_builder.py` is a deprecated wrapper. Use `ui/apps/command_builder/app.py` instead.

See [Command Builder section above](#command-builder--deprecated-wrapper) for details.

### Evaluation Viewer ⚠️ DEPRECATED WRAPPER

> **⚠️ WARNING**: `ui/evaluation_viewer.py` is a deprecated wrapper. Use `ui/evaluation/app.py` instead.

See [Evaluation Viewer section above](#evaluation-viewer) for details.

### Resource Monitor (`resource_monitor.py`) - ✅ New!
A comprehensive monitoring interface for system resources, training processes, and GPU utilization.

**Features:**
- Real-time CPU, memory, and GPU resource monitoring
- Training process and worker process status display
- Process management with safe termination and force kill options
- GPU memory usage visualization with progress bars
- Auto-refresh capability (5-second intervals)
- Quick action buttons for process cleanup and emergency stops

**Process Management:**
- View all training processes and their worker processes
- Terminate processes gracefully (SIGTERM) or forcefully (SIGKILL)
- Confirmation dialogs for dangerous operations
- Integration with the process monitor utility script

**Usage:**
```bash
# Run the resource monitor UI
python run_ui.py resource_monitor

# Or directly with streamlit
uv run streamlit run ui/resource_monitor.py
```

## Architecture

The UI is built with a modular design. **Important**: Some files are deprecated wrappers that will be removed.

### Architecture Diagram

```
ui/
├── ⚠️ command_builder.py          # DEPRECATED: Wrapper for apps/command_builder/app.py
├── ⚠️ inference_ui.py             # DEPRECATED: Wrapper for apps/inference/app.py
├── ⚠️ evaluation_viewer.py        # DEPRECATED: Wrapper for evaluation/app.py
│
├── apps/                          # Actual implementations (USE THESE)
│   ├── command_builder/
│   │   └── app.py                # ✅ Main command builder app
│   └── inference/
│       └── app.py                # ✅ Real-time inference interface
│
├── evaluation/                    # ✅ Evaluation results viewer (modular)
│   ├── __init__.py
│   ├── app.py                    # ✅ Main application
│   ├── single_run.py             # Single model analysis
│   ├── comparison.py             # Model comparison
│   └── gallery.py                # Image gallery
│
├── resource_monitor.py            # ✅ System resource and process monitor
├── components/                    # Reusable UI components
├── utils/                         # Utility modules
│   ├── config_parser.py          # Parses Hydra configurations
│   └── ⚠️ command_builder.py     # DEPRECATED: Use utils/command instead
│   └── command/                  # ✅ Command building utilities
└── __init__.py
```

### Migration Guide

**For Developers:**
- **DO NOT** update deprecated wrapper files (`command_builder.py`, `inference_ui.py`, `evaluation_viewer.py`)
- **DO** update the actual implementations in `apps/` and `evaluation/` directories
- **DO** import directly from actual implementations in new code

**For Users:**
- Use `python run_ui.py <command>` to launch UI apps (recommended)
- Avoid using `streamlit run ui/<wrapper_file>.py` directly
- Wrapper files will be removed in future versions (target: Month 3-6)

**Import Examples:**
```python
# ❌ DON'T (deprecated)
from ui.command_builder import main
from ui.inference_ui import main
from ui.evaluation_viewer import main

# ✅ DO (correct)
from ui.apps.command_builder import main
from ui.apps.inference.app import main
from ui.evaluation import main
```

## Dependencies

- `streamlit >= 1.28.0` - Web UI framework
- Project dependencies (PyTorch, Lightning, etc.)

## Development

The UI applications are designed to be:
- **Modular**: Separate concerns with clear interfaces
- **Extensible**: Easy to add new features and components
- **Integrated**: Works seamlessly with existing CLI tools
- **User-friendly**: Intuitive interface for complex configurations

## Future Enhancements

- Advanced ablation study configuration
- Real-time training progress monitoring
- Model comparison tools
- Automated hyperparameter optimization
- Custom dataset upload and validation
