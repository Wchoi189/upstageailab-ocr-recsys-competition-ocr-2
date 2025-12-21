---
type: architecture
component: checkpoint_catalog
status: current
version: "2.0"
last_updated: "2025-12-15"
---

# Checkpoint Catalog V2 Design

**Purpose**: YAML-based checkpoint metadata system with lazy loading; 40-100x performance improvement over V1.

---

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| **YAML-First** | Metadata stored in `.metadata.yaml`; eliminates checkpoint loading |
| **Lazy Loading** | Load checkpoints only as last-resort fallback |
| **Modular** | Focused modules with clear separation of concerns |
| **Type Safety** | Pydantic V2 for validation and serialization |
| **Backward Compatible** | Support legacy checkpoints without metadata |
| **Cacheable** | LRU cache for repeated catalog builds |

---

## Module Structure

```
ui/apps/inference/services/checkpoint/
├── types.py                       # Pydantic models (YAML schema)
├── metadata_loader.py             # YAML metadata loading
├── config_resolver.py             # Config file resolution & loading
├── validator.py                   # Schema validation & compatibility
├── inference_engine.py            # Checkpoint fallback analysis
├── cache.py                       # Caching layer
├── catalog.py                     # Main catalog builder (orchestration)
└── wandb_client.py                # Wandb API integration (future)
```

---

## Core Metadata Schema (Pydantic V2)

| Field | Type | Purpose |
|-------|------|---------|
| `schema_version` | Literal["1.0"] | Schema versioning for migrations |
| `checkpoint_path` | str | Relative path to checkpoint file |
| `exp_name` | str | Experiment name (directory name) |
| `created_at` | str | ISO 8601 timestamp |
| `training` | TrainingInfo | Epoch, global_step, status |
| `model` | ModelInfo | Architecture, encoder, decoder, head, loss |
| `metrics` | MetricsInfo | Precision, recall, hmean, loss |
| `checkpointing` | CheckpointingConfig | Checkpoint callback config |
| `hydra_config_path` | Optional[str] | Path to Hydra config.yaml |
| `wandb_run_id` | Optional[str] | Wandb run ID for artifact retrieval |

**Nested Models**:
- `TrainingInfo`: epoch, global_step, status
- `ModelInfo`: architecture, encoder, decoder, head, loss
- `MetricsInfo`: val_precision, val_recall, val_hmean, val_loss
- `CheckpointingConfig`: monitor, mode, save_top_k, save_last

---

## YAML Metadata Format

**File**: `.metadata.yaml` (stored alongside checkpoints)

**Example**:
```yaml
schema_version: "1.0"
checkpoint_path: "outputs/experiments/train/ocr/2025-10-18_14-30-00/epoch-14.ckpt"
exp_name: "2025-10-18_14-30-00"
created_at: "2025-10-18T14:45:23+00:00"

training:
  epoch: 14
  global_step: 5040
  status: "completed"

model:
  architecture: "dbnet"
  encoder: "resnet50"
  decoder: "unet"
  head: "db_head"
  loss: "db_loss"

metrics:
  val_precision: 0.8523
  val_recall: 0.8412
  val_hmean: 0.8467
  val_loss: 0.0234

checkpointing:
  monitor: "val_hmean"
  mode: "max"
  save_top_k: 3
  save_last: true

hydra_config_path: "outputs/experiments/train/ocr/2025-10-18_14-30-00/.hydra/config.yaml"
wandb_run_id: "abc123def456"
```

**Size**: ~1-2KB per checkpoint (vs ~500MB-2GB for full checkpoint)

---

## Module Responsibilities

| Module | Responsibility | Key Functions |
|--------|----------------|---------------|
| `metadata_loader` | Load/parse `.metadata.yaml` | `load_metadata()`, `scan_checkpoint_dir()` |
| `config_resolver` | Resolve/load Hydra configs | `resolve_config()`, `load_config_yaml()` |
| `validator` | Validate schema & compatibility | `validate_metadata()`, `check_compatibility()` |
| `inference_engine` | Fallback checkpoint analysis | `extract_metadata_from_ckpt()` |
| `cache` | LRU caching | `cached_catalog()`, `cache_key()` |
| `catalog` | Orchestrate catalog building | `build_catalog()`, `get_checkpoint_list()` |

---

## Data Flow

### Fast Path (YAML metadata exists)
1. Scan checkpoint directory
2. Load `.metadata.yaml` files (Pydantic validation)
3. Build catalog from YAML metadata
4. Cache result
5. Return catalog

**Performance**: 50-300ms for 100 checkpoints

### Slow Path (Legacy fallback)
1. Scan checkpoint directory
2. Load checkpoint state dict
3. Extract metadata from checkpoint
4. Build catalog from extracted metadata
5. Cache result
6. Return catalog

**Performance**: 5-30s for 100 checkpoints (40-100x slower)

---

## API Surface

### Public Exports
```python
# Primary API
from ui.apps.inference.services.checkpoint import build_catalog, get_checkpoint_list

# Data models
from ui.apps.inference.services.checkpoint import CheckpointMetadataV1, CheckpointEntry

# Utilities
from ui.apps.inference.services.checkpoint import load_metadata, validate_metadata
```

### Usage
```python
# Build catalog
catalog = build_catalog(outputs_dir="outputs/experiments/train/ocr/")

# Get sorted checkpoint list
checkpoints = get_checkpoint_list(catalog, sort_by="val_hmean", descending=True)
```

---

## Performance Targets

| Scenario | Checkpoints | Metadata Coverage | Target Time | Speedup |
|----------|-------------|-------------------|-------------|---------|
| **Fast Path** | 100 | 100% | 50-300ms | 100-600x |
| **Slow Path** | 100 | 0% | 5-30s | 1x (baseline) |
| **Mixed** | 100 | 50% | 2.5-15s | 2x |

---

## Dependencies

| Module | Imports | Internal Dependencies |
|--------|---------|----------------------|
| `types` | Pydantic V2 | None |
| `metadata_loader` | PyYAML, Pydantic | `types` |
| `config_resolver` | Hydra, PyYAML | `types` |
| `validator` | Pydantic | `types` |
| `inference_engine` | PyTorch | `types`, `metadata_loader` |
| `cache` | functools.lru_cache | `types` |
| `catalog` | Pathlib | All modules |

---

## Constraints

- **YAML Format**: Must conform to CheckpointMetadataV1 schema
- **Backward Compatibility**: Must support legacy checkpoints without metadata
- **Performance**: Fast path must be <300ms for 100 checkpoints
- **Type Safety**: All data validated via Pydantic V2

---

## Backward Compatibility

**Status**: Maintained for legacy checkpoints

**Breaking Changes**: None (V1 catalog API preserved)

**Migration Path**:
1. V1 checkpoints without `.metadata.yaml` use slow path (fallback)
2. New checkpoints write `.metadata.yaml` during training
3. Gradual migration as new checkpoints replace old ones

**Compatibility Matrix**:

| Checkpoint Type | Metadata | Performance | Support |
|----------------|----------|-------------|---------|
| V2 (with YAML) | ✅ | Fast (50-300ms) | ✅ Full |
| V1 (legacy) | ❌ | Slow (5-30s) | ✅ Fallback |
| Mixed | 🟡 | Medium (2.5-15s) | ✅ Full |

---

## References

- [System Architecture](system-architecture.md)
- [Backend Pipeline Contract](../backend/api/backend-pipeline-contract.md)
- [Backward Compatibility](backward-compatibility.md)
