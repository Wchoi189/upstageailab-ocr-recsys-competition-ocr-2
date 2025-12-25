# OCR Inference Console — AI Agent Documentation

**Purpose**: Ultra-concise, AI agent-oriented documentation for integrating the OCR pipeline with the React-based inference console. Instructions only, no tutorials.

---

## Project State

**Current**:
- OCR pipeline exists in `ocr/` (PyTorch Lightning + DBNet architecture)
- React app scaffold in `apps/ocr-inference-console/` (Vite + TypeScript + React 19)
- Integration layer needed to bridge Python backend with TypeScript frontend

**Next Dev Step**:
- Create Python-to-TypeScript API bridge (FastAPI)
- Expose OCR inference endpoints
- Integrate React UI with API
- Validate on dataset (`data/datasets/images_val_canonical/`, `data/datasets/jsons/val.json`)

---

## Tech Stack

**Frontend** (`apps/ocr-inference-console/`):
- React 19.0.0 + TypeScript
- Vite 6.0.3 (build tool)
- Tailwind CSS 3.4.17 (styling)

**Backend** (OCR Pipeline in `ocr/`):
- Python 3.11+
- PyTorch 2.8+ / PyTorch Lightning 2.1+
- Hydra 1.3+ (config management)
- DBNet architecture for text detection

**Integration**:
- REST API (FastAPI recommended)
- JSON request/response format

---

## Key Paths

- **App Root**: `apps/ocr-inference-console/`
- **Docs (target)**: `apps/ocr-inference-console/docs/`
- **Reference Docs**: `apps/agentqms-dashboard/frontend/docs/` (structural template)
- **OCR Pipeline**: `ocr/` (47+ Python modules)
- **Data (local, gitignored)**:
  - Images: `data/datasets/images_val_canonical/*.jpg`
  - Annotations: `data/datasets/jsons/val.json`
- **Model Checkpoints**: `outputs/experiments/train/ocr/ocr_training_b/{run_id}/checkpoints/`

---

## Data Schema

### Input: Annotation JSON (`val.json`)

```json
{
  "images": {
    "drp.en_ko.in_house.selectstar_000007.jpg": {
      "words": {
        "0001": {
          "points": [[273.31, 164.16], [585.2, 162.57], [585.2, 200.61], [271.72, 202.2]],
          "transcription": "텍스트 내용",
          "illegibility": false,
          "language": "ko"
        }
      },
      "img_w": 1280,
      "img_h": 960
    }
  }
}
```

**Fields**:
- `words`: Object mapping word IDs to word data
- `points`: Array of [x, y] coordinate pairs (polygon vertices)
- `transcription`: Text content
- `illegibility`: Boolean flag
- `language`: Language code (e.g., "ko", "en")
- `img_w`, `img_h`: Image dimensions (pixels)

### Output: Polygon CSV (Submission Format)

```csv
filename,polygons
drp.en_ko.in_house.selectstar_003883.jpg,10 50 100 50 100 150 10 150|110 150 200 150 200 250 110 250
```

**Format**:
- Header: `filename,polygons`
- Polygon format: Space-separated "x y" pairs
- Multiple polygons: Pipe-separated (`|`)
- Coordinates: Integers, pixel-space, top-left origin (0,0)

---

## Existing Infrastructure

### OCR Pipeline Entry Points

**CLI Prediction**:
```bash
uv run python runners/predict.py \
    preset=example \
    checkpoint_path="outputs/experiments/train/ocr/ocr_training_b/{run_id}/checkpoints/best.ckpt"
```

**CLI Training**:
```bash
uv run python runners/train.py preset=example trainer.max_epochs=10
```

**Streamlit UI** (Legacy):
```bash
uv run streamlit run ui/apps/inference/app.py
```

### Key Modules

- **Lightning Module**: `ocr/lightning_modules/ocr_pl.py` (OCRPLModule)
- **Architecture**: `ocr/models/architecture.py` (DBNet)
- **Datasets**: `ocr/datasets/` (data loaders)
- **Metrics**: `ocr/metrics/cleval_metric.py` (CLEval evaluation)
- **Validation**: `ocr/validation/models.py` (Pydantic v2 models)
- **Utils**: `ocr/utils/` (21 utility modules)

### Loading Model Checkpoint

```python
from ocr.lightning_modules.ocr_pl import OCRPLModule

model = OCRPLModule.load_from_checkpoint("path/to/best.ckpt")
predictions = model.predict(image_tensor)
```

---

## Recommended Docs Structure

```
apps/ocr-inference-console/docs/
├── README.md                    # Index + quick links
├── integration/                 # Integration guides
│   ├── ocr-pipeline-bridge.md  # Python → TypeScript API bridge
│   ├── api-endpoints.md        # REST API specification
│   └── data-flow.md            # Data flow diagrams
├── data/                        # Data schemas
│   ├── input-format.md         # Annotation JSON format
│   └── output-format.md        # Polygon CSV format
├── development/                 # Development guides
│   ├── setup.md                # Local setup instructions
│   └── testing.md              # Test commands
└── meta/                        # AI agent instructions
    └── agent-instructions.md   # AI agent system prompts
```

**Reference**: Use `apps/agentqms-dashboard/frontend/docs/` as structural template.

---

## Priority Tasks (Ordered)

### 1. Create Docs Structure (1h)
- Set up `apps/ocr-inference-console/docs/` with recommended structure
- Write README.md with navigation links
- Add frontmatter to all docs (YAML metadata)

### 2. Document OCR Pipeline (2h)
- Map `ocr/` modules to functionality
- Document inference entry points (CLI, API, Streamlit)
- Create architecture diagram (OCR pipeline → API → React UI)

### 3. Design API Bridge (3h)
- Define REST API endpoints (`/predict`, `/health`, `/models`)
- Specify request/response schemas (JSON)
- Plan FastAPI integration with `runners/predict.py`

### 4. Implement Data Validator (2h)
- Create `apps/ocr-inference-console/scripts/validate_data.py`
- Reuse `ocr.validation.models` (Pydantic)
- Add CLI: `python scripts/validate_data.py --path data/datasets --sample 10`

### 5. Create Integration Stub (4h)
- Implement `apps/ocr-inference-console/adapters/ocr_bridge.py`
- Connect to `runners/predict.py` or `ocr.lightning_modules.ocr_pl`
- Add smoke test (process 3 sample images)

### 6. Build React UI (6h)
- Image upload component
- Polygon overlay display
- API client with error handling
- Loading states and feedback

### 7. End-to-End Testing (2h)
- Test upload → inference → display flow
- Validate against sample dataset
- Add CI job for ocr-inference-console

---

## Validation

### Existing Validation

**Runtime Validation**: `ocr/validation/models.py` (Pydantic v2)
**Integration Tests**: `tests/integration/test_ocr_pipeline_integration.py`
**Data Contracts**: `docs/pipeline/data_contracts.md`

### New Validator (for ocr-inference-console)

**Purpose**: Validate dataset before inference
**Location**: `apps/ocr-inference-console/scripts/validate_data.py`
**Reuse**: Import from `ocr.validation.models`

**CLI**:
```bash
python scripts/validate_data.py --path data/datasets --sample 10 --out validation.json
```

**Output** (`validation.json`):
```json
{
  "images_checked": 10,
  "missing_images": 0,
  "invalid_records": 0,
  "status": "ok"
}
```

**Exit Codes**:
- `0`: All validations passed
- `1`: Missing images or invalid records

---

## CI Integration

**Existing CI**: `.github/workflows/ci.yml`
**Test Framework**: pytest (configured in `pytest.ini`)

**Add to CI** (for ocr-inference-console):
```yaml
- name: Test OCR Inference Console
  run: |
    cd apps/ocr-inference-console
    npm install
    npm run test
    npm run build
```

---

## AgentQMS Integration

**Framework**: AgentQMS (see `AgentQMS/knowledge/agent/system.md`)
**Artifacts Location**: `docs/artifacts/`
**Naming Convention**: `YYYY-MM-DD_HHMM_[type]_descriptor.md`

**Artifact Types**:
- `implementation_plan` → `docs/artifacts/implementation_plans/`
- `bug` → `docs/artifacts/bug_reports/`
- `assessment` → `docs/artifacts/assessments/`
- `session` → `docs/artifacts/completed_plans/completion_summaries/session_notes/`

**For ocr-inference-console**:
- Create implementation plan for API bridge
- Document session handovers
- Track bugs and assessments

---

## Development Commands

### Frontend (React + Vite)

```bash
cd apps/ocr-inference-console
npm install
npm run dev          # Start dev server (http://localhost:5173)
npm run build        # Production build
npm run preview      # Preview production build
npm run test         # Run tests
```

### Backend (OCR Pipeline)

```bash
# Predict with checkpoint
uv run python runners/predict.py \
    preset=example \
    checkpoint_path="outputs/experiments/train/ocr/ocr_training_b/{run_id}/checkpoints/best.ckpt"

# Validate data
uv run python scripts/validate_data.py --path data/datasets --sample 10

# Run tests
uv run pytest tests/ -v
```

### Full Stack (if FastAPI backend exists)

```bash
# Backend
uv run uvicorn apps.backend.app:app --reload --port 8000

# Frontend
cd apps/ocr-inference-console && npm run dev
```

---

## Acceptance Criteria

### Data Validator
- Exit code 0 for valid dataset
- Outputs JSON: `{"images_checked": N, "missing_images": 0, "invalid_records": 0}`
- Validates image existence and JSON schema

### Integration Smoke Test
- `build/output/predictions.csv` created
- First row filename matches existing image in `data/datasets/images_val_canonical/`
- Polygon format matches submission spec

### CI Pipeline
- Runs tests and validator automatically on PR
- Frontend builds successfully
- Backend tests pass

---

## Notes

**Data Availability**:
- Dataset is gitignored (may be missing in CI/dev clones)
- Provide sample manifest (`data/sample_manifest.json`) with filenames and checksums for CI

**Conventions**:
- Keep docs machine-parseable (short headers, single-line descriptions)
- Use JSON for API contracts
- Follow AgentQMS naming conventions for artifacts
- Add YAML frontmatter to all docs

**Performance**:
- Offline preprocessing available: `uv run python scripts/preprocess_maps.py`
- Provides 5-8x faster validation speed

---

## Quick Reference

**Project Root**: `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/`
**App**: `apps/ocr-inference-console/`
**OCR Pipeline**: `ocr/`
**Main README**: `README.md` (project overview)
**Data Contracts**: `docs/pipeline/data_contracts.md`
**AgentQMS**: `AgentQMS/knowledge/agent/system.md`
