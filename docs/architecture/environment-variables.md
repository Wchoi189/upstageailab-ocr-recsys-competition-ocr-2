---
type: architecture
component: environment
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Environment Variables

**Purpose**: `OCR_*` prefixed environment variables for flexible path configuration; supports Docker, CI/CD, custom directory structures.

---

## Environment Variables

| Variable | Default | Scope |
|----------|---------|-------|
| `OCR_PROJECT_ROOT` | Auto-detected | Project root directory |
| `OCR_CONFIG_DIR` | `{root}/configs` | Config directory |
| `OCR_OUTPUT_DIR` | `{root}/outputs` | Output directory |
| `OCR_DATA_DIR` | `{root}/data` | Data directory |
| `OCR_IMAGES_DIR` | `{root}/data/datasets/images` | Images directory |
| `OCR_ANNOTATIONS_DIR` | `{root}/data/datasets/jsons` | Annotations directory |
| `OCR_LOGS_DIR` | `{root}/outputs/logs` | Logs directory |
| `OCR_CHECKPOINTS_DIR` | `{root}/outputs/experiments/train/ocr` | Checkpoints directory |
| `OCR_CHECKPOINT_PATH` | Auto-detected (latest) | Specific checkpoint file (optional) |
| `OCR_SUBMISSIONS_DIR` | `{root}/outputs/submissions` | Submissions directory |

---

## OCR_CHECKPOINT_PATH (Auto-Detection)

| Behavior | Implementation |
|----------|----------------|
| **Auto-Detection** (default) | If not set, system detects latest checkpoint from `outputs/experiments/train/ocr/` (sorted by modification time) |
| **Manual Override** | Set to specific `.ckpt` file (relative or absolute path) |

**Examples**:
```bash
# Auto-detection (recommended)
unset OCR_CHECKPOINT_PATH
make serve-ocr-console  # Uses latest checkpoint

# Manual override
export OCR_CHECKPOINT_PATH=outputs/experiments/train/ocr/pan_resnet18/checkpoints/epoch-14.ckpt
make serve-ocr-console  # Uses specified checkpoint
```

---

## Usage Patterns

### FastAPI Application
**Initialization**: Environment variables loaded on FastAPI startup (`@app.on_event("startup")`)

**Logging** (startup):
```
INFO: === Path Configuration ===
INFO: Project root: /workspaces/upstageailab-ocr-recsys-competition-ocr-2
INFO: Config directory: /custom/configs
INFO: Output directory: /custom/outputs
INFO: Using environment variables: OCR_CONFIG_DIR, OCR_OUTPUT_DIR
```

### Streamlit Applications
**Initialization**: Environment variables loaded on app start

### Docker Containers
**Example**:
```bash
docker run -e OCR_CONFIG_DIR=/app/configs -e OCR_OUTPUT_DIR=/app/outputs ocr-app
```

---

## Dependencies

| Component | Environment Variable Usage |
|-----------|----------------------------|
| **FastAPI** | Startup event reads `OCR_*` vars |
| **Streamlit** | App init reads `OCR_*` vars |
| **InferenceEngine** | Uses `OCR_CHECKPOINT_PATH` for model loading |

---

## Constraints

- **Naming Convention**: All variables prefixed with `OCR_`
- **Path Validation**: Invalid paths logged as warnings; fallback to defaults
- **Auto-Detection**: `OCR_CHECKPOINT_PATH` requires valid `OCR_CHECKPOINTS_DIR` for auto-detection

---

## Backward Compatibility

**Status**: Maintained for all deployment scenarios

**Breaking Changes**: None

**Compatibility Matrix**:

| Deployment | Environment Variable Support | Status |
|------------|------------------------------|--------|
| Local Dev | ✅ Full | ✅ Supported |
| Docker | ✅ Full | ✅ Supported |
| CI/CD | ✅ Full | ✅ Supported |
| Legacy Streamlit | ✅ Full | ✅ Supported |

---

## References

- [System Architecture](system-architecture.md)
- [Config Architecture](config-architecture.md)
- [Backend Pipeline Contract](../backend/api/backend-pipeline-contract.md)
