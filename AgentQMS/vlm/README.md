# VLM Image Analysis Tools

Vision Language Model (VLM) based image analysis tools for defect detection, test run failure analysis, and visual evidence generation in experiment tracking workflows.

## Overview

This module provides custom tools that use Vision Language Models to analyze images and automatically generate descriptions for:
- Defect analysis in output images
- Input image characteristics
- Before/after comparisons
- Comprehensive analysis combining all modes

## Quick Start

### Installation

```bash
# Install dependencies using uv (from project root)
uv sync

# Set up API keys (optional, for API backends)
export OPENROUTER_API_KEY="your-key-here"
export SOLAR_PRO2_API_KEY="your-key-here"
```

**Note:** All VLM dependencies are managed in the root `pyproject.toml`. The `requirements.txt` file is deprecated and kept for reference only.

### Basic Usage

```bash
# Analyze a single image for defects
uv run python -m agent_qms.vlm.cli.analyze_image_defects \
  --image path/to/image.jpg \
  --mode defect

# Analyze with VIA annotations
uv run python -m agent_qms.vlm.cli.analyze_image_defects \
  --image path/to/image.jpg \
  --mode defect \
  --via-annotations path/to/annotations.json

# Auto-populate incident report
uv run python -m agent_qms.vlm.cli.analyze_image_defects \
  --image path/to/image.jpg \
  --mode defect \
  --auto-populate \
  --incident-report path/to/incident_report.md
```

## Configuration

Centralized configuration lives in `agent_qms/vlm/config.yaml`. It defines:

- Backend priority and defaults (`backends.*`)
- API endpoints and model names per backend
- Image preprocessing limits (`image.max_resolution`, `image.default_quality`)
- Backend timeout and retry defaults (`backend_defaults.*`)
- `.env` search paths and load order (`env.*`)

Example excerpt:

```yaml
backends:
  default: "openrouter"
  openrouter:
    base_url: "https://openrouter.ai/api/v1"
    default_model: "qwen/qwen-2-vl-72b-instruct"
    api_key_env: "OPENROUTER_API_KEY"
image:
  max_resolution: 2048
  default_quality: 95
```

### Environment Variables

API keys and machine-specific overrides are supplied via `.env` files. Because dotfiles are filtered in this repository, two templates are provided:

- `agent_qms/vlm/env.example` – copy to `.env` for shared settings
- `agent_qms/vlm/env.local.example` – copy to `.env.local` for machine-specific overrides

Loading order (later entries override earlier ones):

1. `agent_qms/vlm/.env`
2. Project root `.env`
3. `agent_qms/vlm/.env.local`
4. Project root `.env.local`
5. OS environment variables

Each backend declares the environment variable it expects in `config.yaml` (e.g., `OPENROUTER_API_KEY`). The CLI also honors `VLM_BACKEND` to force a preferred backend globally.

## Architecture

The module is organized into several components:

- **core/**: Core business logic, interfaces, and data contracts
- **backends/**: VLM backend implementations (OpenRouter, Solar Pro 2, CLI Qwen)
- **integrations/**: External system integrations (experiment-tracker, VIA, reports)
- **cli/**: Command-line interface
- **prompts/**: Prompt templates (Markdown and Jinja2)
- **utils/**: Shared utilities (paths, config, logging)

## Analysis Modes

- **defect**: Analyze output images for visual artifacts and defects
- **input**: Analyze source images for unique properties
- **compare**: Compare before/after image pairs
- **full**: Comprehensive analysis combining all modes

## Backends

The module supports multiple VLM backends with automatic fallback:

1. **OpenRouter** (Qwen3 40b) - Primary, highest quality
2. **Solar Pro 2** - Secondary, alternative API
3. **CLI Qwen VLM** - Fallback, local execution

## Integration with Experiment Tracker

The VLM tools integrate seamlessly with the experiment-tracker module:

```python
from agent_qms.vlm.integrations.experiment_tracker import ExperimentTrackerIntegration

integration = ExperimentTrackerIntegration()
results = integration.analyze_experiment_artifacts(
    experiment_id="20251129_173500_experiment",
    artifact_paths=[Path("artifacts/image1.jpg")],
    mode="defect",
)
```

## VIA Integration

VGG Image Annotator (VIA) integration allows manual annotations to be included in VLM analysis:

1. Annotate images in VIA
2. Export annotations to JSON
3. Use `--via-annotations` flag when analyzing

## Examples

See `prompts/few_shot_examples.json` for example image-description pairs that can be used for few-shot learning.

## Documentation

- [Coding Standards](docs/CODING_STANDARDS.md)
- [Workflow Diagram](docs/workflow.mmd)
- [CHANGELOG](CHANGELOG.md)

## Contributing

See [Coding Standards](docs/CODING_STANDARDS.md) for development guidelines.

## License

Same as parent project.
