# Path Management Audit & Solution Proposal

**Date**: 2025-11-20
**Status**: Proposal
**Priority**: High

## Executive Summary

The codebase has multiple inconsistent path resolution strategies, brittle `Path(__file__).parents[X]` patterns, and scattered hardcoded paths. This creates maintainability issues, runtime failures, and complicates deployment. While `ocr/utils/path_utils.py` provides a centralized solution, it's inconsistently adopted.

## Problem Scope

### 1. **Brittle PROJECT_ROOT Definitions** (CRITICAL)

Multiple modules calculate `PROJECT_ROOT` differently using fragile parent traversal:

| File | Pattern | Risk |
|------|---------|------|
| `services/playground_api/utils/paths.py` | `parents[3]` | ⚠️ Breaks if file moved |
| `ui/utils/inference/dependencies.py` | `parents[3]` | ⚠️ Breaks if file moved |
| `ui/apps/unified_ocr_app/app.py` | `parents[4]` | ⚠️ Very brittle |
| `ui/apps/unified_ocr_app/pages/2_🤖_Inference.py` | `parents[5]` | ⚠️ Extremely brittle |
| `ui/config_parser.py` | Uses `get_path_resolver()` | ✅ Good |

**Impact**: Moving files breaks path resolution, causing runtime errors.

### 2. **Hardcoded Path Strings** (HIGH)

Direct path strings scattered throughout:

```python
# Found in multiple locations
Path("configs")  # ❌ Relative, breaks from different CWD
"outputs/..."    # ❌ Hardcoded string
"data/datasets"  # ❌ Hardcoded string
```

**Locations**:
- `ui/utils/inference/engine.py`: `PROJECT_ROOT / "configs"` (✅ Good, but inconsistent)
- `ui/utils/config_parser.py`: `self.config_dir.parent / "outputs"` (⚠️ Assumes structure)
- `services/playground_api/routers/*.py`: Multiple `PROJECT_ROOT / "outputs"` patterns
- `tests/scripts/test_new_features.py`: `Path("data/datasets/images/val")` (❌ Hardcoded)

### 3. **Inconsistent Path Utility Usage** (MEDIUM)

- ✅ **Good**: `ocr/utils/path_utils.py` provides `OCRPathResolver` with environment variable support
- ❌ **Problem**: Many modules don't use it, creating their own `PROJECT_ROOT`
- ❌ **Problem**: `services/playground_api/utils/paths.py` duplicates logic instead of importing

### 4. **Web Worker Constraints** (MEDIUM)

Frontend web workers:
- ✅ Can access URLs (model paths via `/models/u2net.onnx`)
- ❌ Cannot access environment variables at runtime
- ❌ Cannot access file system
- ✅ Can receive configuration via message passing from main thread

**Current State**: Workers use hardcoded URLs, which is acceptable for frontend.

### 5. **Environment Variable Support** (LOW)

- ✅ `OCRPathResolver.from_environment()` exists
- ❌ Not widely used
- ❌ No documentation on env var names
- ❌ Not integrated with FastAPI/Streamlit startup

## Most Problematic Areas

### **Priority 1: UI Modules**
- **Location**: `ui/apps/*`, `ui/utils/*`
- **Issue**: Multiple different `PROJECT_ROOT` calculations
- **Impact**: High - causes runtime errors when structure changes
- **Files**: ~20 files with `Path(__file__).parents[X]`

### **Priority 2: Services/API Modules**
- **Location**: `services/playground_api/*`
- **Issue**: Duplicate `PROJECT_ROOT` definition, hardcoded paths
- **Impact**: Medium - API reliability
- **Files**: `routers/inference.py`, `routers/pipeline.py`, `utils/paths.py`

### **Priority 3: Test Scripts**
- **Location**: `tests/scripts/*`
- **Issue**: Hardcoded relative paths
- **Impact**: Low - affects test reliability
- **Files**: `test_new_features.py`

## Solution Architecture

### **Principle: Single Source of Truth**

All path resolution should go through `ocr/utils/path_utils.py` via `get_path_resolver()`.

### **1. Centralized PROJECT_ROOT Resolution**

Create a single, robust `PROJECT_ROOT` that works from any location:

```python
# ocr/utils/path_utils.py (ENHANCED)
def _detect_project_root() -> Path:
    """Detect project root using multiple strategies."""
    # Strategy 1: Environment variable (explicit override)
    if env_root := os.getenv("OCR_PROJECT_ROOT"):
        root = Path(env_root).resolve()
        if root.exists() and (root / "pyproject.toml").exists():
            return root

    # Strategy 2: From __file__ location (works in packages)
    # Calculate from this file's location (ocr/utils/path_utils.py -> project root)
    file_based = Path(__file__).resolve().parent.parent.parent
    if (file_based / "pyproject.toml").exists():
        return file_based

    # Strategy 3: Walk up from CWD
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        markers = ["pyproject.toml", ".git", "pyproject.toml"]
        if any((parent / marker).exists() for marker in markers):
            return parent

    # Fallback: CWD
    return cwd

PROJECT_ROOT = _detect_project_root()
```

### **2. Unified Path Utility Module**

Create a single import point for all modules:

```python
# ocr/utils/path_utils.py
# ... existing code ...

# Export a stable PROJECT_ROOT that other modules can import
PROJECT_ROOT = _detect_project_root()

# Export path resolver singleton
def get_project_root() -> Path:
    """Get project root (stable, works from any location)."""
    return PROJECT_ROOT

# Existing get_path_resolver() already exists and is good
```

### **3. Migration Strategy**

#### **Phase 1: Consolidate PROJECT_ROOT Definitions**

Replace all `Path(__file__).parents[X]` with imports:

**Before**:
```python
# services/playground_api/utils/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
```

**After**:
```python
# services/playground_api/utils/paths.py
from ocr.utils.path_utils import PROJECT_ROOT

# Re-export for backward compatibility
__all__ = ["PROJECT_ROOT"]
```

#### **Phase 2: Replace Hardcoded Paths**

**Before**:
```python
outputs_dir = Path("outputs")
config_dir = Path("configs")
```

**After**:
```python
from ocr.utils.path_utils import get_path_resolver

resolver = get_path_resolver()
outputs_dir = resolver.config.output_dir
config_dir = resolver.config.config_dir
```

#### **Phase 3: Environment Variable Integration**

**FastAPI Startup**:
```python
# services/playground_api/app.py
from ocr.utils.path_utils import setup_project_paths

@app.on_event("startup")
def setup_paths():
    """Initialize paths from environment variables."""
    setup_project_paths()  # Reads OCR_* env vars
```

**Streamlit Apps**:
```python
# ui/apps/*/app.py
from ocr.utils.path_utils import setup_project_paths

# Call once at module level or in main()
setup_project_paths()
```

### **4. Web Worker Configuration**

**Problem**: Workers can't access env vars directly.

**Solution**: Pass configuration from main thread via message:

```typescript
// frontend/src/workers/workerHost.ts
interface WorkerConfig {
  modelUrl?: string;
  apiBaseUrl?: string;
  // ... other config
}

export function getWorkerPool(config?: WorkerConfig): WorkerPool {
  // Pass config to workers via initialization message
  const worker = new Worker(workerUrl);
  if (config) {
    worker.postMessage({ type: 'init', config });
  }
  return worker;
}
```

**For Model URLs**: Keep hardcoded `/models/...` URLs in workers - they're frontend assets, not configurable paths.

## Environment Variables Design

### **Standard Variables**

| Variable | Default | Purpose |
|----------|---------|---------|
| `OCR_PROJECT_ROOT` | Auto-detected | Override project root detection |
| `OCR_CONFIG_DIR` | `{root}/configs` | Config directory |
| `OCR_OUTPUT_DIR` | `{root}/outputs` | Output directory |
| `OCR_DATA_DIR` | `{root}/data` | Data directory |

### **When to Use Environment Variables**

✅ **Good for**:
- Deployment configurations
- Docker containers
- CI/CD pipelines
- Multi-tenant setups

❌ **Avoid for**:
- Development (use auto-detection)
- Web workers (use message passing)
- Relative paths in checkpoints (they should be relative to `outputs_dir`)

### **Implementation**

Already exists in `OCRPathResolver.from_environment()` - just needs:
1. Documentation
2. Integration with app startup
3. Validation/logging

## Professional Best Practices

### **1. Path Resolution Hierarchy**

1. **Explicit** (highest priority): Environment variables
2. **Implicit**: Auto-detection via project markers
3. **Fallback**: Current working directory (with warning)

### **2. Path Types**

- **Absolute paths**: Use as-is (user-provided)
- **Relative paths**: Resolve relative to appropriate base (output_dir, config_dir, etc.)
- **Checkpoint paths**: Can be relative to outputs_dir or absolute

### **3. Validation**

```python
def validate_paths(resolver: OCRPathResolver) -> None:
    """Validate that required paths exist or can be created."""
    required = [resolver.config.config_dir]
    for path in required:
        if not path.exists():
            path.mkdir(parents=True, exist=True)
            LOGGER.info(f"Created directory: {path}")
```

### **4. Logging**

Log path resolution at startup:
```python
LOGGER.info("Path configuration:")
LOGGER.info(f"  Project root: {PROJECT_ROOT}")
LOGGER.info(f"  Config dir: {resolver.config.config_dir}")
LOGGER.info(f"  Output dir: {resolver.config.output_dir}")
if os.getenv("OCR_PROJECT_ROOT"):
    LOGGER.info(f"  (Using OCR_PROJECT_ROOT env var)")
```

## Migration Checklist

### **Immediate (High Priority)**

- [ ] Create unified `PROJECT_ROOT` in `ocr/utils/path_utils.py`
- [ ] Update `services/playground_api/utils/paths.py` to import from `ocr.utils.path_utils`
- [ ] Update `ui/utils/inference/dependencies.py` to import from `ocr.utils.path_utils`
- [ ] Replace brittle `parents[X]` in `ui/apps/*/app.py` files

### **Short-term (Medium Priority)**

- [ ] Replace hardcoded `"outputs"`, `"configs"` with `get_path_resolver()`
- [ ] Add environment variable support to FastAPI/Streamlit startup
- [ ] Add path validation and logging
- [ ] Update documentation with env var names

### **Long-term (Low Priority)**

- [ ] Audit test scripts for hardcoded paths
- [ ] Create path utility tests
- [ ] Add path resolution monitoring/logging

## Files Requiring Changes

### **Critical (Fix First)**

1. `services/playground_api/utils/paths.py` - Import PROJECT_ROOT
2. `ui/utils/inference/dependencies.py` - Import PROJECT_ROOT
3. `ui/apps/unified_ocr_app/app.py` - Remove `parents[4]`
4. `ui/apps/unified_ocr_app/pages/*.py` - Remove `parents[5]`

### **High Priority**

5. `ui/utils/config_parser.py` - Use resolver instead of `parent / "outputs"`
6. `services/playground_api/routers/inference.py` - Use resolver
7. `services/playground_api/routers/pipeline.py` - Use resolver

### **Medium Priority**

8. All other `ui/apps/*` files with `parents[X]` patterns
9. Test scripts with hardcoded paths

## Risks & Mitigations

### **Risk 1: Breaking Changes**

**Mitigation**:
- Keep backward compatibility exports
- Gradual migration
- Comprehensive testing

### **Risk 2: Environment Variable Complexity**

**Mitigation**:
- Auto-detection as default (works for 95% of cases)
- Env vars only for deployment/special cases
- Clear documentation

### **Risk 3: Web Worker Limitations**

**Mitigation**:
- Workers only need model URLs (frontend assets)
- Configuration passed via messages
- No file system access needed

## Success Metrics

- ✅ Zero `Path(__file__).parents[X]` patterns (except in `path_utils.py`)
- ✅ Single `PROJECT_ROOT` definition
- ✅ All modules use `get_path_resolver()` or `PROJECT_ROOT` import
- ✅ Environment variable support documented and working
- ✅ Path resolution works from any CWD
- ✅ Tests pass with different working directories

## Related Documentation

- [Path Management Standardization (2025-10-04)](../../archive/changelog/2025-10/04_path-management-standardization.md)
- [Path Standardization Plan](../../2025-10/refactor/04_standardize_paths_plan.md)
- `ocr/utils/path_utils.py` - Current implementation
