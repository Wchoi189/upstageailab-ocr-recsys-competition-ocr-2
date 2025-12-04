# Implementation Plan: Checkpoint Loading Performance Optimization

## Problem Statement

The Next.js Inference Studio currently takes ~5 minutes to load 16 checkpoints, causes 500 errors, and displays streaming GET request errors. The root cause is missing metadata files that force the system into a slow fallback path (loading full PyTorch checkpoints).

## Goals

1. Reduce checkpoint loading time from ~5 minutes to <1 second for 16 checkpoints
2. Eliminate 500 errors during initial page load
3. Enable progressive/streaming loading for better UX
4. Implement efficient `outputs/` folder exposure without file system overload
5. Validate checkpoint health to prevent inference failures

## Root Causes Identified

### 1. Missing Metadata Files (Primary)
- **Impact**: 40-100x slower loading (2.5s vs 10ms per checkpoint)
- **Cause**: `MetadataCallback` not integrated into training pipeline
- **Evidence**: Only `ocr_training_b/` has `.metadata.yaml` files

### 2. No Progressive Loading (Secondary)
- **Impact**: All-or-nothing blocking requests with 30s timeout
- **Cause**: Single synchronous API endpoint, no streaming support
- **Evidence**: Frontend waits for complete response before rendering

### 3. File System Traversal Overhead (Tertiary)
- **Impact**: 2-3 seconds additional overhead per request
- **Cause**: Recursive `outputs/*.ckpt` glob on every catalog build
- **Evidence**: No indexing or caching of directory structure

### 4. No Checkpoint Validation
- **Impact**: Some checkpoints fail during inference
- **Cause**: No health checks after training or before catalog inclusion

## Architecture Review

### Current System (Checkpoint Catalog V2)
- **Design**: `docs/architecture/checkpoint_catalog_v2_design.md`
- **Implementation**: `ocr/utils/checkpoints/catalog.py`
- **Metadata Schema**: `ocr/utils/checkpoints/schemas.py` (CheckpointMetadataV1)
- **Callback**: `ocr/lightning_modules/callbacks/metadata_callback.py`
- **Generation Script**: `scripts/checkpoints/generate_metadata.py`

### Performance Targets
- **With Metadata**: <10ms per checkpoint → 160ms for 16 checkpoints
- **Without Metadata**: 2-5s per checkpoint → 40s+ for 16 checkpoints
- **Speedup**: 40-100x improvement with metadata

## Implementation Plan

### Phase 1: Generate Missing Metadata (Immediate - Day 1)

**Priority**: P0 (Blocks everything else)

#### 1.1 Audit Current Metadata Coverage
**Task**: Identify which checkpoints lack metadata files

```bash
# Count checkpoints vs metadata files
find outputs/ -name "*.ckpt" -type f | wc -l
find outputs/ -name "*.metadata.yaml" -type f | wc -l

# List checkpoints without metadata
make checkpoint-metadata-dry-run
```

**Acceptance Criteria**:
- Complete list of checkpoints needing metadata
- Estimated generation time calculated

#### 1.2 Generate Metadata for Existing Checkpoints
**Task**: Run metadata generation script for all existing checkpoints

```bash
# Full generation (may take 5-10 minutes for 60 checkpoints)
make checkpoint-metadata

# Or parallel generation with progress
python scripts/checkpoints/generate_metadata.py \
  --outputs-dir outputs/ \
  --workers 4 \
  --progress
```

**Files Modified**: None (generates new `.metadata.yaml` files)

**Acceptance Criteria**:
- All valid checkpoints have adjacent `.metadata.yaml` files
- Metadata validates against CheckpointMetadataV1 schema
- Catalog build time drops to <1s

#### 1.3 Verify Performance Improvement
**Task**: Test catalog build speed with metadata

```bash
# Restart backend to clear cache
make backend-restart

# Check logs for catalog build time
grep "Catalog built" logs/backend/*.log

# Expected: "Catalog built: 16 entries (16 YAML, 0 Wandb, 0 legacy) in 0.15s"
```

**Acceptance Criteria**:
- Catalog build completes in <1 second
- Fast path count (YAML) matches checkpoint count
- No 500 errors during checkpoint loading

---

### Phase 2: Integrate Metadata Generation into Training (Day 1-2)

**Priority**: P0 (Prevents future metadata gaps)

#### 2.1 Create Metadata Callback Config
**Task**: Add callback configuration for training pipeline

**File**: `configs/callbacks/metadata.yaml` (NEW)

```yaml
# @package _global_

# Metadata generation callback for Checkpoint Catalog V2
# Generates .metadata.yaml files during training for 40-100x faster catalog builds

defaults:
  - default
  - _self_

callbacks:
  metadata:
    _target_: ocr.lightning_modules.callbacks.metadata_callback.MetadataCallback
    exp_name: ${exp_name}
    outputs_dir: ${paths.output_dir}
    training_phase: training
    generate_on_save: true
    validate_schema: true
```

**Acceptance Criteria**:
- Config follows Hydra conventions
- Resolves `exp_name` and `outputs_dir` from global config
- Can be toggled via `callbacks.metadata=null`

#### 2.2 Add Metadata Callback to Default Training
**Task**: Include metadata callback in default training configuration

**File**: `configs/train.yaml`

```yaml
defaults:
  - _self_
  - data: default
  - model: default
  - callbacks: default
  - callbacks/metadata  # ← Add this line
  - logger: wandb
  - trainer: default
  # ... rest of defaults
```

**Alternative**: Modify `configs/callbacks/default.yaml` to include metadata

**Acceptance Criteria**:
- Metadata callback active by default for all training runs
- Can be disabled with `callbacks.metadata=null` override
- No performance impact on training (<1ms overhead per checkpoint)

#### 2.3 Update Training Documentation
**Task**: Document metadata callback in training docs

**Files**:
- `CONTRIBUTING.md` - Add section on checkpoint metadata
- `docs/pipeline/training.md` - Explain metadata generation
- `README.md` - Update training instructions if applicable

**Content**:
- Purpose of metadata files
- Performance benefits (40-100x speedup)
- How to disable if needed
- Manual generation command for legacy checkpoints

**Acceptance Criteria**:
- Clear documentation for developers
- Troubleshooting guide included
- Examples of metadata file structure

#### 2.4 Validate Metadata Generation During Training
**Task**: Run a test training job and verify metadata is generated

```bash
# Small test run
python run_train.py \
  experiment=test \
  trainer.max_epochs=1 \
  callbacks.metadata.generate_on_save=true

# Check for metadata file
ls -la outputs/test/checkpoints/*.metadata.yaml
```

**Acceptance Criteria**:
- `.metadata.yaml` files created alongside `.ckpt` files
- Metadata validates against schema
- No training errors or warnings

---

### Phase 3: Implement Progressive Loading (Day 2-3)

**Priority**: P1 (Improves UX, prevents timeouts)

#### 3.1 Add Streaming Checkpoint Endpoint (Backend)
**Task**: Create SSE endpoint for progressive checkpoint loading

**File**: `apps/backend/routers/inference.py`

```python
from fastapi.responses import StreamingResponse
import json

@router.get("/checkpoints/stream")
async def stream_checkpoints(
    limit: int = 50,
    batch_size: int = 10
) -> StreamingResponse:
    """
    Stream checkpoints progressively using Server-Sent Events (SSE).

    Returns checkpoints in batches to enable progressive UI updates.
    """
    async def generate():
        options = _catalog_options()

        # Build catalog with generator pattern
        entry_count = 0
        batch = []

        for entry in _catalog_generator(options):
            batch.append(entry)
            entry_count += 1

            # Send batch when full or at limit
            if len(batch) >= batch_size or entry_count >= limit:
                data = {
                    "type": "batch",
                    "checkpoints": [e.dict() for e in batch],
                    "count": entry_count,
                    "has_more": entry_count < limit
                }
                yield f"data: {json.dumps(data)}\n\n"
                batch = []

            if entry_count >= limit:
                break

        # Send final batch if any remain
        if batch:
            data = {
                "type": "batch",
                "checkpoints": [e.dict() for e in batch],
                "count": entry_count,
                "has_more": False
            }
            yield f"data: {json.dumps(data)}\n\n"

        # Send completion event
        yield f"data: {json.dumps({'type': 'complete', 'total': entry_count})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
```

**Additional Changes**:
- Refactor `build_lightweight_catalog` to support generator pattern
- Add `_catalog_generator` helper function
- Maintain backward compatibility with `/checkpoints` endpoint

**Acceptance Criteria**:
- SSE endpoint streams checkpoints in batches
- Compatible with EventSource API
- No memory accumulation (generator pattern)
- Handles client disconnection gracefully

#### 3.2 Implement Streaming Client (Frontend)
**Task**: Add SSE client for progressive checkpoint loading

**File**: `apps/frontend/src/api/inference.ts`

```typescript
export interface CheckpointStreamCallback {
  onBatch: (checkpoints: CheckpointWithMetadata[], count: number, hasMore: boolean) => void;
  onComplete: (total: number) => void;
  onError: (error: Error) => void;
}

export function streamCheckpoints(
  limit: number = 50,
  callbacks: CheckpointStreamCallback
): EventSource {
  const url = `${API_BASE_URL}/inference/checkpoints/stream?limit=${limit}`;
  const eventSource = new EventSource(url);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      if (data.type === 'batch') {
        const checkpoints = data.checkpoints.map(enrichCheckpointData);
        callbacks.onBatch(checkpoints, data.count, data.has_more);
      } else if (data.type === 'complete') {
        callbacks.onComplete(data.total);
        eventSource.close();
      }
    } catch (err) {
      callbacks.onError(err instanceof Error ? err : new Error(String(err)));
      eventSource.close();
    }
  };

  eventSource.onerror = (err) => {
    callbacks.onError(new Error('Stream connection error'));
    eventSource.close();
  };

  return eventSource;
}
```

**Acceptance Criteria**:
- EventSource properly handles SSE protocol
- Callbacks fired for each batch
- Error handling and reconnection logic
- Cleanup on unmount

#### 3.3 Update CheckpointPicker Component
**Task**: Refactor to use streaming API for progressive loading

**File**: `apps/frontend/src/components/inference/CheckpointPicker.tsx`

```typescript
useEffect(() => {
  setLoading(true);
  setAllCheckpoints([]);

  const eventSource = streamCheckpoints(100, {
    onBatch: (newCheckpoints, count, hasMore) => {
      setAllCheckpoints(prev => [...prev, ...newCheckpoints]);
      setLoadingProgress({ loaded: count, hasMore });

      // Show first batch immediately
      if (count <= 10) {
        setLoading(false);
      }
    },
    onComplete: (total) => {
      setLoading(false);
      setLoadingProgress({ loaded: total, hasMore: false });
    },
    onError: (err) => {
      setError(err.message);
      setLoading(false);
    }
  });

  return () => {
    eventSource.close();
  };
}, []);
```

**UI Enhancements**:
- Show loading progress: "Loaded 10/60 checkpoints..."
- Render checkpoints as they arrive (no blocking)
- Graceful degradation if SSE not supported

**Acceptance Criteria**:
- First batch renders within 1 second
- Progressive updates every 10 checkpoints
- No UI freezing during load
- Smooth user experience

#### 3.4 Add Feature Toggle
**Task**: Make streaming optional with fallback to standard endpoint

**File**: `apps/frontend/src/config.ts`

```typescript
export const FEATURES = {
  streamingCheckpoints: true,  // Toggle via env var
  // ... other features
};
```

**Environment Variable**: `VITE_ENABLE_STREAMING_CHECKPOINTS=true`

**Acceptance Criteria**:
- Can toggle between streaming and standard loading
- Defaults to streaming when available
- Falls back gracefully if SSE fails

---

### Phase 4: Optimize Outputs Folder Exposure (Day 3-4)

**Priority**: P1 (Reduces file system overhead)

#### 4.1 Create Checkpoint Index System
**Task**: Build indexing system to avoid repeated file system scans

**File**: `ocr/utils/checkpoints/index.py` (NEW)

```python
"""
Checkpoint index management for fast lookup without file system scans.

The index is a JSON file that maps experiment names to checkpoint metadata,
updated incrementally as new checkpoints are saved.
"""

from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

class CheckpointIndex:
    """
    Maintains an index of all checkpoints for O(1) lookup.

    Index structure:
    {
      "version": "1.0",
      "last_updated": "2025-12-04T10:30:00Z",
      "checkpoints": {
        "ocr_training_b": {
          "checkpoints": ["best.ckpt", "last.ckpt"],
          "metadata_files": ["best.metadata.yaml", "last.metadata.yaml"],
          "last_modified": "2025-12-03T15:20:00Z"
        }
      }
    }
    """

    def __init__(self, outputs_dir: Path):
        self.outputs_dir = outputs_dir
        self.index_file = outputs_dir / ".checkpoint_index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        """Load existing index or create new one."""
        if self.index_file.exists():
            return json.loads(self.index_file.read_text())
        return {"version": "1.0", "checkpoints": {}}

    def add_checkpoint(self, exp_name: str, checkpoint_name: str):
        """Add checkpoint to index."""
        if exp_name not in self.index["checkpoints"]:
            self.index["checkpoints"][exp_name] = {
                "checkpoints": [],
                "metadata_files": []
            }

        if checkpoint_name not in self.index["checkpoints"][exp_name]["checkpoints"]:
            self.index["checkpoints"][exp_name]["checkpoints"].append(checkpoint_name)

        # Check for metadata file
        metadata_name = f"{checkpoint_name}.metadata.yaml"
        metadata_path = self.outputs_dir / exp_name / "checkpoints" / metadata_name
        if metadata_path.exists():
            self.index["checkpoints"][exp_name]["metadata_files"].append(metadata_name)

        self.save()

    def get_checkpoint_paths(self) -> List[Path]:
        """Get all checkpoint paths from index without file system scan."""
        paths = []
        for exp_name, data in self.index["checkpoints"].items():
            exp_dir = self.outputs_dir / exp_name / "checkpoints"
            for ckpt in data["checkpoints"]:
                paths.append(exp_dir / ckpt)
        return paths

    def save(self):
        """Persist index to disk."""
        self.index["last_updated"] = datetime.utcnow().isoformat() + "Z"
        self.index_file.write_text(json.dumps(self.index, indent=2))

    def rebuild(self):
        """Full rebuild from file system (fallback/initialization)."""
        self.index = {"version": "1.0", "checkpoints": {}}

        for ckpt_path in self.outputs_dir.rglob("*.ckpt"):
            exp_name = ckpt_path.parent.parent.name
            ckpt_name = ckpt_path.name
            self.add_checkpoint(exp_name, ckpt_name)
```

**Acceptance Criteria**:
- Index loads in <10ms
- Supports incremental updates
- Rebuild command for recovery
- Thread-safe for concurrent access

#### 4.2 Integrate Index with Catalog Builder
**Task**: Use index instead of file system scans

**File**: `ocr/utils/checkpoints/catalog.py`

```python
from ocr.utils.checkpoints.index import CheckpointIndex

def build_lightweight_catalog(options: CatalogOptions) -> CatalogResult:
    """Build checkpoint catalog using index for fast lookup."""

    # Try to use index first
    index = CheckpointIndex(options.outputs_dir)

    if index.index_file.exists():
        LOGGER.info("Using checkpoint index for fast catalog build")
        checkpoint_paths = index.get_checkpoint_paths()
    else:
        LOGGER.warning("Checkpoint index not found, falling back to file system scan")
        checkpoint_paths = sorted(options.outputs_dir.rglob("*.ckpt"))

        # Rebuild index in background for next time
        index.rebuild()

    # Rest of catalog building logic...
```

**Acceptance Criteria**:
- Index used when available
- Falls back to file system scan if index missing
- Auto-rebuilds index on first use
- Logs which method was used

#### 4.3 Update Metadata Callback to Update Index
**Task**: Incrementally update index when checkpoints are saved

**File**: `ocr/lightning_modules/callbacks/metadata_callback.py`

```python
from ocr.utils.checkpoints.index import CheckpointIndex

class MetadataCallback(Callback):
    def __init__(self, exp_name: str, outputs_dir: Path, ...):
        super().__init__()
        self.index = CheckpointIndex(outputs_dir)
        # ... rest of init

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Generate metadata and update index after checkpoint save."""
        # ... existing metadata generation logic ...

        # Update index
        checkpoint_name = Path(checkpoint_path).name
        self.index.add_checkpoint(self.exp_name, checkpoint_name)

        LOGGER.debug(f"Updated checkpoint index for {self.exp_name}/{checkpoint_name}")
```

**Acceptance Criteria**:
- Index updated automatically on checkpoint save
- <1ms overhead per checkpoint
- No race conditions with concurrent saves

#### 4.4 Add Index Management Commands
**Task**: Add Makefile targets for index operations

**File**: `Makefile`

```makefile
.PHONY: checkpoint-index-rebuild
checkpoint-index-rebuild:  ## Rebuild checkpoint index from file system
	@echo "Rebuilding checkpoint index..."
	python -c "from ocr.utils.checkpoints.index import CheckpointIndex; \
	           from ocr.utils.path_utils import get_path_resolver; \
	           index = CheckpointIndex(get_path_resolver().config.output_dir); \
	           index.rebuild(); \
	           print(f'Indexed {len(index.get_checkpoint_paths())} checkpoints')"

.PHONY: checkpoint-index-verify
checkpoint-index-verify:  ## Verify checkpoint index accuracy
	@echo "Verifying checkpoint index..."
	python scripts/checkpoints/verify_index.py
```

**Acceptance Criteria**:
- Commands documented in `make help`
- Rebuild completes in <5 seconds for 60 checkpoints
- Verify detects missing/extra entries

---

### Phase 5: Add Checkpoint Health Validation (Day 4-5)

**Priority**: P2 (Prevents inference failures)

#### 5.1 Create Checkpoint Validator
**Task**: Implement checkpoint health check system

**File**: `ocr/utils/checkpoints/validator.py` (NEW)

```python
"""
Checkpoint validation to detect corrupted or incompatible checkpoints.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of checkpoint validation."""
    is_valid: bool
    checkpoint_path: Path
    error_message: Optional[str] = None
    warnings: list[str] = None

    # Validation checks
    can_load: bool = False
    has_state_dict: bool = False
    has_config: bool = False
    has_metadata: bool = False
    architecture_valid: bool = False

def validate_checkpoint(checkpoint_path: Path) -> ValidationResult:
    """
    Validate checkpoint health and compatibility.

    Checks:
    1. File is readable and not corrupted
    2. Contains valid state_dict
    3. Has associated config.yaml
    4. Has metadata.yaml (warning if missing)
    5. Architecture matches known models
    """
    result = ValidationResult(
        is_valid=False,
        checkpoint_path=checkpoint_path,
        warnings=[]
    )

    # Check 1: Can load checkpoint file
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        result.can_load = True
    except Exception as e:
        result.error_message = f"Failed to load checkpoint: {str(e)}"
        return result

    # Check 2: Has state_dict
    if "state_dict" in ckpt:
        result.has_state_dict = True
    else:
        result.error_message = "Missing 'state_dict' key"
        return result

    # Check 3: Has config file
    config_path = checkpoint_path.parent.parent / "config.yaml"
    result.has_config = config_path.exists()
    if not result.has_config:
        result.warnings.append(f"Missing config file: {config_path}")

    # Check 4: Has metadata file
    metadata_path = checkpoint_path.with_suffix(".ckpt.metadata.yaml")
    result.has_metadata = metadata_path.exists()
    if not result.has_metadata:
        result.warnings.append(f"Missing metadata file (performance impact): {metadata_path}")

    # Check 5: Architecture validation
    try:
        # Check for known model keys
        state_keys = set(ckpt["state_dict"].keys())
        required_keys = {"model.encoder", "model.decoder", "model.head"}

        # Simplified check - look for model prefix
        has_model_keys = any(k.startswith("model.") for k in state_keys)
        result.architecture_valid = has_model_keys

        if not result.architecture_valid:
            result.warnings.append("Unexpected state_dict structure")
    except Exception as e:
        result.warnings.append(f"Could not validate architecture: {str(e)}")

    # Final verdict
    result.is_valid = result.can_load and result.has_state_dict

    return result
```

**Acceptance Criteria**:
- Validates checkpoints without loading full model
- Detects corrupted files
- Identifies missing dependencies (config, metadata)
- Returns actionable error messages

#### 5.2 Add Validation to Catalog Builder
**Task**: Filter out invalid checkpoints during catalog build

**File**: `ocr/utils/checkpoints/catalog.py`

```python
from ocr.utils.checkpoints.validator import validate_checkpoint

def build_lightweight_catalog(options: CatalogOptions) -> CatalogResult:
    """Build checkpoint catalog with validation."""

    entries = []
    invalid_count = 0

    for checkpoint_path in checkpoint_paths:
        # Validate checkpoint
        validation = validate_checkpoint(checkpoint_path)

        if not validation.is_valid:
            LOGGER.warning(
                "Skipping invalid checkpoint %s: %s",
                checkpoint_path,
                validation.error_message
            )
            invalid_count += 1
            continue

        # Log warnings but include checkpoint
        for warning in validation.warnings:
            LOGGER.debug(warning)

        # Build catalog entry...

    LOGGER.info(
        "Catalog built: %d valid checkpoints, %d invalid skipped",
        len(entries),
        invalid_count
    )
```

**Acceptance Criteria**:
- Invalid checkpoints excluded from catalog
- Warnings logged but don't block inclusion
- Performance impact <100ms for 60 checkpoints

#### 5.3 Add Validation Command
**Task**: Provide CLI tool for checkpoint validation

**File**: `scripts/checkpoints/validate_checkpoints.py` (NEW)

```python
"""
Validate all checkpoints and generate health report.
"""

import sys
from pathlib import Path
from ocr.utils.checkpoints.validator import validate_checkpoint
from ocr.utils.path_utils import get_path_resolver

def main():
    outputs_dir = get_path_resolver().config.output_dir
    checkpoint_paths = sorted(outputs_dir.rglob("*.ckpt"))

    print(f"Validating {len(checkpoint_paths)} checkpoints...\n")

    valid = []
    invalid = []
    warnings_count = 0

    for ckpt_path in checkpoint_paths:
        result = validate_checkpoint(ckpt_path)

        if result.is_valid:
            valid.append(ckpt_path)
            warnings_count += len(result.warnings)

            if result.warnings:
                print(f"⚠️  {ckpt_path.relative_to(outputs_dir)}")
                for warning in result.warnings:
                    print(f"    {warning}")
        else:
            invalid.append(ckpt_path)
            print(f"❌ {ckpt_path.relative_to(outputs_dir)}")
            print(f"    {result.error_message}")
        print()

    print(f"\n{'='*60}")
    print(f"✅ Valid: {len(valid)}")
    print(f"⚠️  Warnings: {warnings_count}")
    print(f"❌ Invalid: {len(invalid)}")
    print(f"{'='*60}\n")

    if invalid:
        print("Invalid checkpoints should be removed or regenerated.")
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
```

**Makefile Target**:

```makefile
.PHONY: checkpoint-validate
checkpoint-validate:  ## Validate all checkpoints and generate health report
	python scripts/checkpoints/validate_checkpoints.py
```

**Acceptance Criteria**:
- Reports valid/invalid checkpoints
- Lists specific issues for each checkpoint
- Exit code indicates success/failure
- Can be used in CI/CD pipeline

#### 5.4 Add Validation to Metadata Generation
**Task**: Skip metadata generation for invalid checkpoints

**File**: `scripts/checkpoints/generate_metadata.py`

```python
from ocr.utils.checkpoints.validator import validate_checkpoint

def generate_metadata_for_checkpoint(checkpoint_path: Path, ...):
    """Generate metadata with validation."""

    # Validate first
    validation = validate_checkpoint(checkpoint_path)

    if not validation.is_valid:
        LOGGER.error(
            "Skipping metadata generation for invalid checkpoint %s: %s",
            checkpoint_path,
            validation.error_message
        )
        return False

    # Warn about issues but continue
    for warning in validation.warnings:
        LOGGER.warning(warning)

    # Generate metadata...
```

**Acceptance Criteria**:
- Invalid checkpoints skipped automatically
- Warnings logged but don't block generation
- Summary shows validation results

---

### Phase 6: Backend Optimizations (Day 5)

**Priority**: P2 (Additional performance gains)

#### 6.1 Add Catalog Preloading on Server Start
**Task**: Warm up catalog cache during backend initialization

**File**: `apps/backend/main.py` or `apps/backend/app.py`

```python
@app.on_event("startup")
async def preload_checkpoint_catalog():
    """
    Preload checkpoint catalog on server start to eliminate first-request latency.

    Runs in background to not block server startup.
    """
    import asyncio
    from apps.backend.services.checkpoint import _discover_checkpoints

    async def _preload():
        try:
            LOGGER.info("Preloading checkpoint catalog...")
            start = time()

            checkpoints = _discover_checkpoints(limit=100)

            LOGGER.info(
                "Checkpoint catalog preloaded: %d checkpoints in %.2fs",
                len(checkpoints),
                time() - start
            )
        except Exception as exc:
            LOGGER.error("Failed to preload checkpoint catalog: %s", exc)

    # Run in background
    asyncio.create_task(_preload())
```

**Acceptance Criteria**:
- Catalog preloaded on server start
- Doesn't block server startup
- First request gets cached result
- Logged for monitoring

#### 6.2 Implement Pagination
**Task**: Add true pagination support (not just limit)

**File**: `apps/backend/routers/inference.py`

```python
@router.get("/checkpoints", response_model=CheckpointListResponse)
def list_checkpoints(
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "modified_time",
    sort_order: str = "desc"
) -> CheckpointListResponse:
    """
    List checkpoints with pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Items per page
        sort_by: Sort field (modified_time, hmean, name)
        sort_order: Sort direction (asc, desc)
    """
    # Build full catalog once (cached)
    all_checkpoints = _discover_checkpoints(limit=1000)

    # Sort
    if sort_by == "hmean":
        all_checkpoints.sort(
            key=lambda c: c.metrics.get("hmean", 0),
            reverse=(sort_order == "desc")
        )
    elif sort_by == "modified_time":
        all_checkpoints.sort(
            key=lambda c: c.modified_time or "",
            reverse=(sort_order == "desc")
        )
    # ... other sort options

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    page_checkpoints = all_checkpoints[start:end]

    return CheckpointListResponse(
        checkpoints=page_checkpoints,
        total=len(all_checkpoints),
        page=page,
        page_size=page_size,
        total_pages=(len(all_checkpoints) + page_size - 1) // page_size
    )
```

**Response Model**:

```python
class CheckpointListResponse(BaseModel):
    checkpoints: list[CheckpointSummary]
    total: int
    page: int
    page_size: int
    total_pages: int
```

**Acceptance Criteria**:
- Supports pagination parameters
- Returns total count and pages
- Frontend can request specific pages
- Sorting works correctly

#### 6.3 Add Caching Headers
**Task**: Set appropriate cache headers for checkpoint endpoints

**File**: `apps/backend/routers/inference.py`

```python
from fastapi import Response

@router.get("/checkpoints")
def list_checkpoints(..., response: Response):
    """List checkpoints with caching."""

    checkpoints = _discover_checkpoints(...)

    # Set cache headers
    response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutes
    response.headers["ETag"] = _generate_catalog_etag(checkpoints)

    return checkpoints

def _generate_catalog_etag(checkpoints: list) -> str:
    """Generate ETag based on checkpoint list."""
    content = "".join(c.checkpoint_path for c in checkpoints)
    return hashlib.md5(content.encode()).hexdigest()
```

**Acceptance Criteria**:
- Cache-Control headers set appropriately
- ETags enable conditional requests
- Browser/client caching reduces server load

---

### Phase 7: Frontend Enhancements (Day 6)

**Priority**: P3 (Polish & UX improvements)

#### 7.1 Add Loading States and Progress Indicators
**Task**: Improve UX during checkpoint loading

**File**: `apps/frontend/src/components/inference/CheckpointPicker.tsx`

```typescript
interface LoadingState {
  phase: 'initializing' | 'loading' | 'complete' | 'error';
  loaded: number;
  total: number;
  message: string;
}

// In component
const [loadingState, setLoadingState] = useState<LoadingState>({
  phase: 'initializing',
  loaded: 0,
  total: 0,
  message: 'Initializing checkpoint catalog...'
});

// Render
{loadingState.phase === 'loading' && (
  <div className="checkpoint-loading">
    <ProgressBar
      value={loadingState.loaded}
      max={loadingState.total}
    />
    <p>{loadingState.message}</p>
    <p>{loadingState.loaded} / {loadingState.total} checkpoints loaded</p>
  </div>
)}
```

**Acceptance Criteria**:
- Clear loading states
- Progress bar for streaming loads
- Informative messages
- Smooth transitions

#### 7.2 Implement Virtual Scrolling
**Task**: Optimize rendering of large checkpoint lists

**File**: `apps/frontend/src/components/inference/CheckpointList.tsx`

```typescript
import { FixedSizeList as List } from 'react-window';

const CheckpointList: React.FC<Props> = ({ checkpoints }) => {
  const Row = ({ index, style }) => (
    <div style={style}>
      <CheckpointCard checkpoint={checkpoints[index]} />
    </div>
  );

  return (
    <List
      height={600}
      itemCount={checkpoints.length}
      itemSize={120}
      width="100%"
    >
      {Row}
    </List>
  );
};
```

**Acceptance Criteria**:
- Renders 100+ checkpoints smoothly
- No scroll lag
- Maintains performance with large lists

#### 7.3 Add Checkpoint Filtering and Search
**Task**: Enable users to find checkpoints quickly

**File**: `apps/frontend/src/components/inference/CheckpointPicker.tsx`

```typescript
const [searchQuery, setSearchQuery] = useState('');
const [filters, setFilters] = useState({
  minHmean: 0,
  architecture: 'all',
  dateRange: 'all'
});

const filteredCheckpoints = useMemo(() => {
  return allCheckpoints.filter(ckpt => {
    // Search filter
    if (searchQuery && !ckpt.exp_name.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }

    // Hmean filter
    if (ckpt.metrics?.hmean && ckpt.metrics.hmean < filters.minHmean) {
      return false;
    }

    // Architecture filter
    if (filters.architecture !== 'all' && ckpt.model?.architecture !== filters.architecture) {
      return false;
    }

    return true;
  });
}, [allCheckpoints, searchQuery, filters]);
```

**Acceptance Criteria**:
- Search by experiment name
- Filter by metric thresholds
- Filter by architecture type
- Instant filtering (no backend calls)

#### 7.4 Add Checkpoint Comparison View
**Task**: Allow side-by-side comparison of checkpoints

**File**: `apps/frontend/src/components/inference/CheckpointComparison.tsx` (NEW)

**Features**:
- Select multiple checkpoints
- Compare metrics side-by-side
- Highlight differences
- Export comparison table

**Acceptance Criteria**:
- Supports 2-4 checkpoint comparison
- Clear metric visualization
- Responsive layout

---

### Phase 8: Monitoring and Logging (Day 6-7)

**Priority**: P3 (Observability)

#### 8.1 Add Performance Metrics
**Task**: Track catalog build performance

**File**: `ocr/utils/checkpoints/catalog.py`

```python
import time
from contextlib import contextmanager

@contextmanager
def track_performance(operation: str):
    """Context manager for performance tracking."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        LOGGER.info(f"{operation} completed in {duration:.3f}s")

        # Send to monitoring (if configured)
        if wandb.run:
            wandb.log({f"perf/{operation}": duration})

def build_lightweight_catalog(options: CatalogOptions) -> CatalogResult:
    """Build catalog with performance tracking."""

    with track_performance("checkpoint_catalog_build"):
        # ... catalog building logic
        pass
```

**Metrics to Track**:
- Catalog build time
- Fast path vs slow path ratio
- Checkpoint validation time
- Index rebuild time
- API endpoint latency

**Acceptance Criteria**:
- Metrics logged consistently
- Sent to W&B or monitoring system
- Available for alerting

#### 8.2 Add Error Tracking
**Task**: Comprehensive error logging for debugging

**File**: `apps/backend/services/checkpoint.py`

```python
import traceback

def _discover_checkpoints(limit: int = 50) -> list[CheckpointSummary]:
    """Discover checkpoints with comprehensive error tracking."""

    try:
        catalog = build_lightweight_catalog(options)
    except Exception as exc:
        # Log full traceback
        LOGGER.error(
            "Failed to build checkpoint catalog:\n%s",
            traceback.format_exc()
        )

        # Track error in monitoring
        if wandb.run:
            wandb.log({"errors/catalog_build": 1})

        # Send to error tracking service (Sentry, etc.)
        # sentry_sdk.capture_exception(exc)

        raise HTTPException(
            status_code=500,
            detail=f"Failed to build checkpoint catalog: {str(exc)}. "
                   "Check server logs for details."
        )
```

**Acceptance Criteria**:
- Full stack traces logged
- Error context captured
- Integration points for Sentry/similar
- User-friendly error messages

#### 8.3 Add Health Check Endpoint
**Task**: Expose catalog health for monitoring

**File**: `apps/backend/routers/health.py`

```python
@router.get("/health/checkpoints")
def checkpoint_catalog_health() -> dict:
    """
    Health check for checkpoint catalog system.

    Returns:
        - catalog_status: healthy/degraded/unhealthy
        - checkpoint_count: number of available checkpoints
        - metadata_coverage: percentage of checkpoints with metadata
        - last_build_time: catalog build duration
        - errors: list of recent errors
    """
    try:
        start = time()
        checkpoints = _discover_checkpoints(limit=1000)
        build_time = time() - start

        # Calculate metadata coverage
        with_metadata = sum(1 for c in checkpoints if c.has_metadata)
        metadata_coverage = (with_metadata / len(checkpoints) * 100) if checkpoints else 0

        # Determine status
        if build_time > 5.0:
            status = "degraded"
        elif metadata_coverage < 80:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "catalog_status": status,
            "checkpoint_count": len(checkpoints),
            "metadata_coverage": f"{metadata_coverage:.1f}%",
            "last_build_time": f"{build_time:.2f}s",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as exc:
        return {
            "catalog_status": "unhealthy",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
```

**Acceptance Criteria**:
- Returns health status
- Includes key metrics
- Can be monitored by uptime services
- Fast response (<1s)

---

## Testing Plan

### Unit Tests

#### Test Metadata Generation
```python
def test_metadata_callback_generates_file(tmp_path):
    """Test metadata file generation during training."""
    callback = MetadataCallback(exp_name="test", outputs_dir=tmp_path)
    # ... simulate checkpoint save
    assert (tmp_path / "test/checkpoints/epoch=0.ckpt.metadata.yaml").exists()

def test_metadata_schema_validation():
    """Test metadata conforms to schema."""
    metadata = load_checkpoint_metadata(metadata_path)
    # Should not raise ValidationError
    CheckpointMetadataV1(**metadata)
```

#### Test Checkpoint Index
```python
def test_checkpoint_index_add_and_retrieve(tmp_path):
    """Test index operations."""
    index = CheckpointIndex(tmp_path)
    index.add_checkpoint("exp1", "best.ckpt")

    paths = index.get_checkpoint_paths()
    assert len(paths) == 1
    assert paths[0].name == "best.ckpt"

def test_checkpoint_index_rebuild(tmp_path):
    """Test index rebuild from file system."""
    # Create fake checkpoints
    (tmp_path / "exp1/checkpoints").mkdir(parents=True)
    (tmp_path / "exp1/checkpoints/best.ckpt").touch()

    index = CheckpointIndex(tmp_path)
    index.rebuild()

    assert len(index.get_checkpoint_paths()) == 1
```

#### Test Checkpoint Validation
```python
def test_validate_valid_checkpoint(valid_checkpoint_path):
    """Test validation of valid checkpoint."""
    result = validate_checkpoint(valid_checkpoint_path)
    assert result.is_valid
    assert result.can_load
    assert result.has_state_dict

def test_validate_corrupted_checkpoint(corrupted_checkpoint_path):
    """Test validation of corrupted checkpoint."""
    result = validate_checkpoint(corrupted_checkpoint_path)
    assert not result.is_valid
    assert result.error_message is not None
```

### Integration Tests

#### Test Catalog Builder
```python
def test_catalog_build_with_metadata(outputs_dir_with_metadata):
    """Test catalog build uses metadata files."""
    options = CatalogOptions(outputs_dir=outputs_dir_with_metadata)

    start = time()
    catalog = build_lightweight_catalog(options)
    build_time = time() - start

    assert build_time < 1.0  # Should be fast with metadata
    assert catalog.fast_path_count == len(catalog.entries)

def test_catalog_build_without_metadata(outputs_dir_without_metadata):
    """Test catalog build falls back without metadata."""
    options = CatalogOptions(outputs_dir=outputs_dir_without_metadata)

    catalog = build_lightweight_catalog(options)

    assert catalog.slow_path_count > 0
    assert len(catalog.entries) > 0  # Should still work
```

#### Test API Endpoints
```python
def test_list_checkpoints_endpoint(test_client):
    """Test /checkpoints endpoint."""
    response = test_client.get("/api/inference/checkpoints?limit=10")

    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 10
    assert all("checkpoint_path" in c for c in data)

def test_stream_checkpoints_endpoint(test_client):
    """Test /checkpoints/stream endpoint."""
    with test_client.get("/api/inference/checkpoints/stream") as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"

        # Read events
        events = []
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                events.append(json.loads(line[6:]))

        assert len(events) > 0
        assert events[-1]["type"] == "complete"
```

### Performance Tests

#### Test Catalog Build Speed
```python
def test_catalog_build_performance_with_metadata(benchmark, outputs_dir_with_metadata):
    """Benchmark catalog build with metadata."""
    options = CatalogOptions(outputs_dir=outputs_dir_with_metadata)

    result = benchmark(build_lightweight_catalog, options)

    # Should be <1s for 60 checkpoints
    assert result.catalog_build_time_seconds < 1.0

def test_catalog_build_performance_without_metadata(benchmark, outputs_dir_without_metadata):
    """Benchmark catalog build without metadata (slow path)."""
    options = CatalogOptions(outputs_dir=outputs_dir_without_metadata)

    result = benchmark(build_lightweight_catalog, options)

    # Document slow path performance for comparison
    print(f"Slow path build time: {result.catalog_build_time_seconds:.2f}s")
```

#### Test Frontend Loading
```typescript
describe('CheckpointPicker loading performance', () => {
  it('should render first batch within 1 second', async () => {
    const start = Date.now();

    render(<CheckpointPicker />);

    // Wait for first batch
    await waitFor(() => {
      expect(screen.getByText(/checkpoint/i)).toBeInTheDocument();
    });

    const loadTime = Date.now() - start;
    expect(loadTime).toBeLessThan(1000);
  });

  it('should handle 100 checkpoints without lag', async () => {
    const checkpoints = generateMockCheckpoints(100);

    const { container } = render(
      <CheckpointList checkpoints={checkpoints} />
    );

    // Measure scroll performance
    const list = container.querySelector('.checkpoint-list');
    const start = performance.now();

    list.scrollTop = list.scrollHeight;
    await new Promise(resolve => setTimeout(resolve, 100));

    const scrollTime = performance.now() - start;
    expect(scrollTime).toBeLessThan(100);
  });
});
```

### End-to-End Tests

#### Test Full User Flow
```python
def test_checkpoint_loading_user_flow(browser):
    """Test complete checkpoint loading flow."""
    # Navigate to Inference Studio
    browser.get("http://localhost:5173/inference")

    # Wait for checkpoint picker to load
    wait = WebDriverWait(browser, 30)
    checkpoint_picker = wait.until(
        EC.presence_of_element_located((By.CLASS_NAME, "checkpoint-picker"))
    )

    # Should see checkpoints within 1 second
    checkpoints = wait.until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "checkpoint-card"))
    )

    assert len(checkpoints) > 0

    # Should not see 500 errors
    logs = browser.get_log('browser')
    errors = [log for log in logs if log['level'] == 'SEVERE']
    assert len(errors) == 0
```

---

## Rollout Plan

### Phase 1: Immediate (Day 1)
1. Generate metadata for all existing checkpoints
2. Verify performance improvement
3. Deploy to development environment

### Phase 2: Training Integration (Day 1-2)
1. Add metadata callback to training configs
2. Test with new training runs
3. Validate metadata generation

### Phase 3: Progressive Loading (Day 2-3)
1. Implement streaming backend
2. Update frontend for progressive loading
3. Test with multiple checkpoint counts

### Phase 4: Optimization (Day 3-5)
1. Implement checkpoint index
2. Add validation system
3. Optimize catalog builder

### Phase 5: Polish (Day 5-7)
1. Add monitoring and logging
2. Improve frontend UX
3. Write documentation

### Phase 6: Production (Day 7+)
1. Staging environment deployment
2. Performance validation
3. Production rollout with monitoring

---

## Success Metrics

### Performance Targets
- **Catalog Build Time**: <1 second (from ~5 minutes)
- **First Render Time**: <1 second (from page load)
- **Metadata Coverage**: >95% of checkpoints
- **Error Rate**: <1% of requests
- **Index Rebuild Time**: <5 seconds for 60 checkpoints

### User Experience Targets
- **Zero 500 errors** during normal operation
- **Progressive loading** shows first results within 1 second
- **Smooth scrolling** with 100+ checkpoints
- **Clear error messages** for invalid checkpoints

### System Health Targets
- **Metadata generation** automated for all training runs
- **Index accuracy** >99% (no stale entries)
- **Checkpoint validation** catches 100% of corrupted files
- **Cache hit rate** >90% for repeated requests

---

## Rollback Plan

### If Performance Issues Occur
1. Revert to non-streaming endpoint
2. Increase timeout values
3. Reduce batch sizes

### If Metadata Generation Fails
1. Disable metadata callback
2. Fall back to slow path (still functional)
3. Fix issues in development
4. Regenerate metadata

### If Index Becomes Stale
1. Delete `.checkpoint_index.json`
2. Falls back to file system scan automatically
3. Index rebuilds on next request

---

## Documentation Updates

### User Documentation
- **README.md**: Quick start with performance notes
- **docs/inference/checkpoint-loading.md**: Detailed guide
- **docs/troubleshooting/slow-loading.md**: Troubleshooting guide

### Developer Documentation
- **CONTRIBUTING.md**: Checkpoint metadata requirements
- **docs/architecture/checkpoint-catalog-v2.md**: Update with index system
- **docs/development/testing.md**: Add performance testing guidelines

### API Documentation
- **apps/backend/README.md**: API endpoint documentation
- **OpenAPI spec**: Update with streaming endpoint
- **Frontend README**: Document streaming client usage

---

## Future Enhancements

### Phase 2 (Beyond Initial Rollout)
1. **Checkpoint Deduplication**: Identify identical checkpoints by state dict hash
2. **Automatic Cleanup**: Remove old/unused checkpoints based on policy
3. **Cloud Storage Integration**: S3/GCS support with presigned URLs
4. **Checkpoint Diffing**: Compare state dicts to show what changed
5. **Metadata Search**: Full-text search across experiment configs
6. **Checkpoint Tagging**: User-defined tags for organization
7. **Checkpoint Lineage**: Track checkpoint genealogy (fine-tuning chains)

### Performance Optimizations
1. **Parallel Index Updates**: Lock-free index updates
2. **Incremental Catalog Builds**: Only process new checkpoints
3. **Checkpoint Preview**: Generate thumbnail/preview images
4. **Smart Preloading**: Predict which checkpoints user will select

---

## Risk Assessment

### High Risk
- **Metadata generation failure**: Mitigated by fallback to slow path
- **Index corruption**: Mitigated by automatic rebuild
- **Backward compatibility**: Mitigated by maintaining old endpoint

### Medium Risk
- **SSE browser compatibility**: Mitigated by feature detection and fallback
- **Large checkpoint files**: Mitigated by validation skipping full loads
- **Concurrent catalog builds**: Mitigated by caching layer

### Low Risk
- **Metadata schema changes**: Mitigated by version field in schema
- **Disk space for index**: Minimal (<1MB for 1000 checkpoints)
- **Training overhead**: <1ms per checkpoint, negligible

---

## Questions for Review

1. Should we implement pagination immediately or as Phase 2?
2. What's the desired behavior for checkpoints that fail validation?
3. Should streaming endpoint replace the standard endpoint or coexist?
4. What monitoring/alerting thresholds should we set?
5. Should we implement checkpoint cleanup/archival as part of this work?

---

## Appendix: File Structure

```
ocr/
├── utils/
│   └── checkpoints/
│       ├── catalog.py                    # Existing (modified)
│       ├── schemas.py                    # Existing
│       ├── metadata_loader.py            # Existing
│       ├── index.py                      # NEW
│       └── validator.py                  # NEW
├── lightning_modules/
│   └── callbacks/
│       └── metadata_callback.py          # Existing (modified)

scripts/
└── checkpoints/
    ├── generate_metadata.py              # Existing (modified)
    ├── validate_checkpoints.py           # NEW
    └── verify_index.py                   # NEW

configs/
└── callbacks/
    ├── default.yaml                      # Existing (modified)
    └── metadata.yaml                     # NEW

apps/
├── backend/
│   ├── routers/
│   │   ├── inference.py                  # Existing (modified)
│   │   └── health.py                     # Existing (modified)
│   └── main.py                           # Existing (modified)
└── frontend/
    └── src/
        ├── api/
        │   ├── client.ts                 # Existing (modified)
        │   └── inference.ts              # Existing (modified)
        └── components/
            └── inference/
                ├── CheckpointPicker.tsx  # Existing (modified)
                ├── CheckpointList.tsx    # NEW
                └── CheckpointComparison.tsx  # NEW

outputs/
└── .checkpoint_index.json                # NEW (generated)
```

---

## Estimated Effort

- **Phase 1 (Metadata Generation)**: 2 hours
- **Phase 2 (Training Integration)**: 4 hours
- **Phase 3 (Progressive Loading)**: 8 hours
- **Phase 4 (Optimization)**: 8 hours
- **Phase 5 (Validation)**: 6 hours
- **Phase 6 (Backend Optimizations)**: 4 hours
- **Phase 7 (Frontend Enhancements)**: 6 hours
- **Phase 8 (Monitoring)**: 4 hours
- **Testing**: 8 hours
- **Documentation**: 4 hours

**Total**: ~54 hours (7-8 days for 1 developer)

---

**Status**: Draft - Ready for Review
**Created**: 2025-12-04
**Author**: GitHub Copilot
**Review Required**: Yes
