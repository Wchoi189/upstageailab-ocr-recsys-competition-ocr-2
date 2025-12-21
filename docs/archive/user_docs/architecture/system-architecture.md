---
type: architecture
component: null
status: current
version: "2.0"
last_updated: "2025-12-15"
---

# System Architecture

**Purpose**: OCR system architecture with modular ML framework, multi-application deployment, and shared inference engine.

---

## Architecture Overview

| Layer | Components | Registry | Configuration |
|-------|------------|----------|---------------|
| **ML Framework** | Encoders, Decoders, Heads, Losses | Central component catalog | Hydra-based declarative configs |
| **Applications** | Legacy Streamlit (deprecated), Playground Console (Next.js), OCR Console (Vite), Backend API (FastAPI) | N/A | YAML-driven model/data configs |
| **Shared Logic** | InferenceEngine, OCR modules | N/A | 106 YAML config files |

---

## Application Landscape

| Application | Type | Location | Status | Purpose |
|-------------|------|----------|--------|---------|
| **Legacy Streamlit** | Streamlit | `ui/` | ⚠️ Deprecated | Inference, Command Builder, Visualization |
| **Playground Console** | Next.js | `apps/playground-console/` | 🟡 75% Complete | Command builder, inference, comparison |
| **OCR Inference Console** | Vite+React | `apps/ocr-inference-console/` | 🟡 70% Complete | Lightweight inference UI |
| **Backend API** | FastAPI | `apps/backend/` | ✅ Active | Serves Next.js consoles |

---

## Component Registry

| Base Class | Registry | Factory | Config Location |
|------------|----------|---------|-----------------|
| `BaseEncoder` | Encoder registry | ModelFactory | `configs/model/encoder/` |
| `BaseDecoder` | Decoder registry | ModelFactory | `configs/model/decoder/` |
| `BaseHead` | Head registry | ModelFactory | `configs/model/head/` |
| `BaseLoss` | Loss registry | ModelFactory | `configs/model/loss/` |

**ModelFactory**: Assembles models from registered components using Hydra instantiation.

---

## Directory Structure

```
ocr/
├── architectures/     # DBNet, EAST implementations
├── core/              # Abstract base classes
├── models/            # Model factory, composite model
├── datasets/          # Data loading
├── training/          # Training logic
└── evaluation/        # Metrics

apps/
├── backend/           # FastAPI (ocr_bridge, playground_api)
├── playground-console/   # Next.js full console
└── ocr-inference-console/  # Vite inference-only console

ui/                    # DEPRECATED Streamlit apps

configs/               # 106 YAML files
├── _base/             # Base templates
├── model/             # encoder/, decoder/, head/, loss/
├── data/              # Dataset configs
└── trainer/           # Training configs
```

---

## Data Flow

### Training Pipeline
1. Input Image → OCRTransforms
2. ValidatedOCRDataset → DataLoader
3. OCRLightningModule → OCRModel
4. Encoder → Decoder → Head → Loss
5. Backward pass → Optimizer update

### Inference Pipeline
1. Frontend (Playground/OCR Console) → Backend API
2. Backend → InferenceEngine (`ui/utils/inference/engine.py`)
3. InferenceEngine → OCR modules (model loading, preprocessing)
4. Image preprocessing (LongestMaxSize + PadIfNeeded)
5. Coordinate transformation + polygon extraction
6. Response → Frontend

---

## Backend API Endpoints

| Route | Consumer | Purpose |
|-------|----------|---------|
| `/ocr/*` | OCR Inference Console | Inference, checkpoint list |
| `/api/*` | Playground Console | Command builder, inference, comparison |
| `/docs` | Development | Swagger documentation |

**Key Components**:
- `services/ocr_bridge.py` - Wraps InferenceEngine for OCR Console
- `services/playground_api/` - Full playground API

---

## Shared Logic: InferenceEngine

**Location**: `ui/utils/inference/engine.py`

**Consumers**: OCR Bridge, Playground API, Legacy Streamlit

**Capabilities**:
- Model loading with caching (lazy load for fast startup)
- Image preprocessing (coordinate transformation)
- Polygon extraction from model outputs

**Why Shared**: Ensures consistent behavior, eliminates duplication.

---

## Configuration System (Hydra)

**Usage**:
```bash
# Basic training
uv run python runners/train.py preset=<name>

# Override parameters
uv run python runners/train.py model.optimizer.lr=0.0005 data.batch_size=16

# Switch architectures
uv run python runners/train.py model.architecture=east
```

**Instantiation**:
```python
from hydra.utils import instantiate

config = {
    '_target_': 'ocr_framework.architectures.dbnet.encoder.TimmBackbone',
    'backbone': 'resnet50',
    'pretrained': True
}
encoder = instantiate(config)
```

---

## Dependencies

| Component | Imports | Internal Dependencies |
|-----------|---------|----------------------|
| **Backend** | FastAPI, PyTorch | InferenceEngine, OCR modules |
| **InferenceEngine** | PyTorch, Albumentations | OCR models, configs |
| **ML Framework** | PyTorch, Timm, Hydra | Registered components |
| **Frontend Apps** | React/Next.js, TypeScript | Backend API |

---

## Constraints

- **Encoder-Decoder Compatibility**: Encoder output channels must match decoder input channels
- **Model-Specific Data**: CRAFT requires character-level annotations; DBNet accepts word/line level
- **Lazy Loading**: InferenceEngine defers model loading until first request for fast startup
- **Legacy Deprecation**: Streamlit apps receive bug fixes only; no new features

---

## Backward Compatibility

**Status**: Maintained across backend API, InferenceEngine interface

**Breaking Changes**: None in current version

**Migration Path**: Legacy Streamlit → Playground Console (command builder) or OCR Console (inference)

**Compatibility Matrix**:

| Interface | v1.x (Legacy) | v2.0 (Current) | Notes |
|-----------|---------------|----------------|-------|
| InferenceEngine API | ✅ Compatible | ✅ Compatible | Signature unchanged |
| Backend `/ocr/*` | N/A (new) | ✅ Stable | New API, versioned |
| Backend `/api/*` | N/A (new) | 🟡 In progress | Playground API stabilizing |
| Hydra Configs | ✅ Compatible | ✅ Compatible | YAML structure stable |

**Development Policy**:

| Application | Status | Policy |
|------------|--------|--------|
| Backend API, Playground, OCR Console | ✅ Active | Full development, new features |
| Legacy Streamlit | ⚠️ Maintenance | Bug fixes only, no features |
| Archived docs | ⛔ Deprecated | Unmaintained |

---

## References

- [Config Architecture](config-architecture.md)
- [Backward Compatibility](backward-compatibility.md)
- [API Decoupling](api-decoupling.md)
- [Inference Overview](inference-overview.md)
