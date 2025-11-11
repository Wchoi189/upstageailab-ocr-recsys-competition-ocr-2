# Bug Fixes - 2025-10-21

## Summary

Fixed three critical issues with the Unified OCR App:
1. Inference mode checkpoint loading failure
2. Missing Makefile integration for unified app
3. Image orientation not respecting EXIF data

---

## Issue #1: Inference Mode Checkpoint Loading Failure

**Error**: `AttributeError: 'PathsConfig' object has no attribute 'outputs_dir'`

### Root Cause

The `load_checkpoints()` function in inference_service.py was creating a minimal `PathsConfig` class that only had `checkpoints_dir` attribute, but the `CatalogOptions.from_paths()` method expected both `outputs_dir` and `hydra_config_filenames`.

### Fix

**Files Modified**:
- ui/apps/unified_ocr_app/services/inference_service.py
- configs/ui/modes/inference.yaml

**Changes**:

1. **Import proper PathConfig class** instead of creating a minimal one:
```python
from ui.apps.inference.models.config import PathConfig
from ui.apps.inference.services.checkpoint_catalog import CatalogOptions, build_lightweight_catalog

# Create proper PathConfig instance
path_config = PathConfig(
    outputs_dir=Path(outputs_dir),
    hydra_config_filenames=[
        "config.yaml",
        "hparams.yaml",
        "train.yaml",
        "predict.yaml",
    ],
)

options = CatalogOptions.from_paths(path_config)
```

2. **Add paths configuration** to `configs/ui/modes/inference.yaml`:
```yaml
# Paths configuration
paths:
  outputs_dir: "outputs"
  checkpoints_dir: "outputs/checkpoints"
```

### Verification

```bash
# Test app startup
uv run streamlit run ui/apps/unified_ocr_app/app.py
# ✓ No AttributeError, checkpoints load successfully
```

---

## Issue #2: Missing Makefile Integration

**Problem**: Unified OCR App was not integrated with the project's Makefile and process manager system, unlike other UI apps.

### Fix

**Files Modified**:
- scripts/process_manager.py
- Makefile

**Changes**:

1. **Added unified_app to process_manager.py**:
```python
def _get_ui_path(self, ui_name: str) -> Path:
    """Get the path to a UI script."""
    ui_paths = {
        "command_builder": "ui/command_builder.py",
        "evaluation_viewer": "ui/evaluation_viewer.py",
        "inference": "ui/inference_ui.py",
        "preprocessing_viewer": "ui/preprocessing_viewer_app.py",
        "resource_monitor": "ui/resource_monitor.py",
        "unified_app": "ui/apps/unified_ocr_app/app.py",  # ← Added
    }
```

2. **Updated list_running() and stop_all() methods** to include `unified_app`

3. **Added Makefile targets**:
```makefile
# Serve
serve-unified-app:
    uv run python scripts/process_manager.py start unified_app --port=$(PORT)

# Stop
stop-unified-app:
    uv run python scripts/process_manager.py stop unified_app --port=$(PORT)

# Status
status-unified-app:
    uv run python scripts/process_manager.py status unified_app --port=$(PORT)

# Logs
logs-unified-app:
    uv run python scripts/process_manager.py logs unified_app --port=$(PORT)

# Clear logs
clear-logs-unified-app:
    uv run python scripts/process_manager.py clear-logs unified_app --port=$(PORT)
```

### Usage

```bash
# Start the unified app on default port (8501)
make serve-unified-app

# Start on custom port
make serve-unified-app PORT=8503

# Check status
make status-unified-app PORT=8503

# View logs
make logs-unified-app PORT=8503

# Stop the app
make stop-unified-app PORT=8503

# Clear logs
make clear-logs-unified-app PORT=8503
```

### Verification

```bash
# Test all commands
make serve-unified-app PORT=8503
# ✓ Started unified_app (PID: 4152430) on port 8503

make status-unified-app PORT=8503
# ✓ unified_app: Running (PID: 4152430, Port: 8503)

make stop-unified-app PORT=8503
# ✓ Stopped unified_app
```

**Logs Location**: `logs/ui/unified_app_<PORT>.out` and `logs/ui/unified_app_<PORT>.err`

---

## Issue #3: Image Orientation Not Canonical

**Problem**: Loaded images appeared non-canonical (not upright) in Step-by-Step and Side-by-Side modes. Images taken with cameras often have EXIF orientation metadata that indicates the correct rotation, but this was being ignored.

### Root Cause

The image loading code in image_upload.py was using `PIL.Image.open()` without applying EXIF orientation transformation. This caused rotated images to display incorrectly.

### Fix

**File Modified**: ui/apps/unified_ocr_app/components/shared/image_upload.py

**Changes**:

1. **Applied EXIF orientation in upload handler**:
```python
# Load image
image = Image.open(uploaded_file)

# Apply EXIF orientation (fix rotated images from cameras)
try:
    from PIL import ImageOps
    image = ImageOps.exif_transpose(image)
except Exception:
    # If EXIF processing fails, continue with original image
    pass

# Convert to RGB if needed
if image.mode != "RGB":
    image = image.convert("RGB")
```

2. **Updated load_image_from_path() to also handle EXIF**:
```python
def load_image_from_path(image_path: str | Path) -> np.ndarray | None:
    """Load image from file path with EXIF orientation handling."""
    # Load with PIL to handle EXIF orientation
    image = Image.open(image_path)

    # Apply EXIF orientation
    try:
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    # Convert to RGB and then to BGR (OpenCV format)
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return image_bgr
```

### Technical Details

**EXIF Orientation Tags**:
- EXIF tag 274 (0x0112) stores orientation information
- Values 1-8 indicate different rotation/flip combinations
- `ImageOps.exif_transpose()` automatically handles all cases

**Why This Matters**:
- Mobile phone cameras often embed orientation in EXIF
- Without processing, images appear rotated even though they look correct in photo viewers
- This affects preprocessing pipeline accuracy and visualization

### Verification

```python
# Test with rotated image
image = Image.open("rotated_image.jpg")
print(f"Before: {image.size}")  # e.g., (480, 640) - landscape

image = ImageOps.exif_transpose(image)
print(f"After: {image.size}")   # e.g., (640, 480) - portrait (correctly rotated)
```

---

## Summary of Changes

### Files Modified (7 files)

1. **ui/apps/unified_ocr_app/services/inference_service.py** - Fixed checkpoint loading
2. **configs/ui/modes/inference.yaml** - Added paths configuration
3. **ui/apps/unified_ocr_app/components/shared/image_upload.py** - Fixed EXIF orientation
4. **scripts/process_manager.py** - Added unified_app support
5. **Makefile** - Added unified_app targets

### Impact

- **Inference Mode**: ✅ Now loads checkpoints correctly
- **Process Management**: ✅ Unified app integrated with standard workflow
- **Image Display**: ✅ Images display in correct orientation
- **Preprocessing**: ✅ Pipeline processes correctly oriented images

### Testing

All fixes verified with:
```bash
# App startup
uv run streamlit run ui/apps/unified_ocr_app/app.py
# ✓ No errors, all modes functional

# Makefile integration
make serve-unified-app PORT=8503
make status-unified-app PORT=8503
make stop-unified-app PORT=8503
# ✓ All commands working

# Type checking
uv run mypy ui/apps/unified_ocr_app/
# ✓ No type errors
```

---

## Related Documentation

- **Architecture**: docs/ai_handbook/08_planning/UNIFIED_STREAMLIT_APP_ARCHITECTURE.md
- **Phase 6 Summary**: [SESSION_COMPLETE_2025-10-21_PHASE6.md](SESSION_COMPLETE_2025-10-21_PHASE6.md)
- **CHANGELOG**: docs/CHANGELOG.md

---

*Fixed: 2025-10-21*
*Unified OCR App - Critical Bug Fixes*
