# Environment Variables for Path Configuration

**Last Updated**: 2025-11-20
**Status**: Active

## Overview

The OCR project supports environment variable-based path configuration for flexible deployment scenarios. This is particularly useful for:
- Docker containers
- CI/CD pipelines
- Multi-tenant deployments
- Custom directory structures

## Available Environment Variables

All path-related environment variables use the `OCR_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_PROJECT_ROOT` | Auto-detected | Override project root directory |
| `OCR_CONFIG_DIR` | `{root}/configs` | Config directory |
| `OCR_OUTPUT_DIR` | `{root}/outputs` | Output directory |
| `OCR_DATA_DIR` | `{root}/data` | Data directory |
| `OCR_IMAGES_DIR` | `{root}/data/datasets/images` | Images directory |
| `OCR_ANNOTATIONS_DIR` | `{root}/data/datasets/jsons` | Annotations directory |
| `OCR_LOGS_DIR` | `{root}/outputs/logs` | Logs directory |
| `OCR_CHECKPOINTS_DIR` | `{root}/outputs/checkpoints` | Checkpoints directory |
| `OCR_SUBMISSIONS_DIR` | `{root}/outputs/submissions` | Submissions directory |

## Usage

### FastAPI Application

Environment variables are automatically loaded on FastAPI startup:

```python
# services/playground_api/app.py
# Paths are initialized in @app.on_event("startup")
```

**Logging**: Path configuration is logged at startup:
```
INFO: === Path Configuration ===
INFO: Project root: /workspaces/upstageailab-ocr-recsys-competition-ocr-2
INFO: Config directory: /custom/configs
INFO: Output directory: /custom/outputs
INFO: Using environment variables: OCR_CONFIG_DIR, OCR_OUTPUT_DIR
```

### Streamlit Applications

Environment variables are automatically loaded when the app starts:

```python
# ui/apps/unified_ocr_app/app.py
# Paths are initialized at module level
```

**Logging**: Path configuration is logged to stderr on startup.

### Manual Usage

```python
from ocr.utils.path_utils import setup_project_paths

# Read from environment variables
resolver = setup_project_paths()

# Or use explicit config dict
resolver = setup_project_paths({
    "output_dir": "/custom/outputs",
    "config_dir": "/custom/configs"
})

# Access paths
output_dir = resolver.config.output_dir
config_dir = resolver.config.config_dir
```

## Detection Order

1. **Explicit Config** (if `setup_project_paths(config={...})` is called)
2. **Environment Variables** (if `OCR_*` vars are set)
3. **Auto-detection** (from `__file__` location or CWD walk-up)

## Examples

### Docker Container

```dockerfile
# Dockerfile
ENV OCR_OUTPUT_DIR=/data/outputs
ENV OCR_DATA_DIR=/data/datasets
ENV OCR_PROJECT_ROOT=/app

# Application will use these paths automatically
```

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml
env:
  OCR_OUTPUT_DIR: ${{ runner.temp }}/outputs
  OCR_DATA_DIR: ${{ runner.workspace }}/data

# Tests will use these paths
```

### Development Override

```bash
# .env file (loaded by your environment)
export OCR_OUTPUT_DIR=/tmp/my_outputs
export OCR_CHECKPOINTS_DIR=/tmp/my_checkpoints

# Run application
streamlit run ui/apps/unified_ocr_app/app.py
```

## Logging

Both FastAPI and Streamlit apps log path configuration at startup:

**Without Environment Variables**:
```
INFO: Using auto-detected paths (no environment variables set)
INFO: Project root: /workspaces/upstageailab-ocr-recsys-competition-ocr-2
INFO: Config directory: /workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs
INFO: Output directory: /workspaces/upstageailab-ocr-recsys-competition-ocr-2/outputs
```

**With Environment Variables**:
```
INFO: Using environment variables: OCR_OUTPUT_DIR, OCR_CONFIG_DIR
INFO: Project root: /workspaces/upstageailab-ocr-recsys-competition-ocr-2
INFO: Config directory: /custom/configs
INFO: Output directory: /custom/outputs
```

## Best Practices

1. **Development**: Don't set environment variables - use auto-detection
2. **Deployment**: Set `OCR_PROJECT_ROOT` if app runs from different directory
3. **Docker**: Set paths in Dockerfile or docker-compose.yml
4. **CI/CD**: Use temporary directories for outputs/logs
5. **Multi-tenant**: Set per-tenant directories via environment variables

## Validation

The path resolver automatically:
- Creates directories if they don't exist
- Validates that project root exists (if `OCR_PROJECT_ROOT` is set)
- Logs warnings if paths seem incorrect

## Troubleshooting

**Problem**: Paths not using environment variables

**Solution**:
- Check variable names start with `OCR_`
- Verify variables are set before app startup
- Check logs for "Using environment variables" message

**Problem**: Paths resolving incorrectly

**Solution**:
- Set `OCR_PROJECT_ROOT` explicitly
- Check current working directory
- Review startup logs for path resolution details

## Related Documentation

- [Path Management Audit](../../planning/plans/2025-11/path-management-audit-and-solution.md)
- [Path Management Implementation Progress](../../planning/plans/2025-11/path-management-implementation-progress.md)
- `ocr/utils/path_utils.py` - Implementation

