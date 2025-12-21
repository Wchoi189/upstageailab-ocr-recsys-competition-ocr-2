# Installation Guide

## Prerequisites

- **Python**: 3.11 or higher
- **UV Package Manager**: Fast Python package installer
- **GPU**: CUDA-compatible GPU recommended for training (8GB+ VRAM)
- **OS**: Linux, macOS, or Windows with WSL2

## Quick Installation

```bash
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone <your-repo-url>
cd upstageailab-ocr-recsys-competition-ocr-2

# 3. Run automated setup
./scripts/setup/00_setup-environment.sh
```

The setup script will:
- Create a virtual environment
- Install all Python dependencies via UV
- Set up project paths
- Verify CUDA availability

## Manual Installation

If you prefer manual setup:

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync

# Verify installation
uv run pytest tests/ -v
```

## Verify Installation

```bash
# Check Python version
python --version  # Should be 3.11+

# Check CUDA availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run tests
uv run pytest tests/ -v
```

## Frontend Setup (Optional)

For UI development:

```bash
# Install Node.js dependencies
npm install

# Start development servers
make fs  # Starts both backend and frontend
```

## Troubleshooting

### UV command not found
```bash
# Reinstall UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# Add to PATH (check UV installation output)
```

### CUDA not available
- Verify NVIDIA drivers are installed
- Check PyTorch CUDA compatibility
- Try CPU-only mode for inference

### Import errors
Always use `uv run` prefix:
```bash
# ❌ Wrong
python runners/train.py

# ✅ Correct
uv run python runners/train.py
```

## Next Steps

- [Training Guide](training.md) - Train your first model
- [Configuration Guide](../architecture/CONFIG_ARCHITECTURE.md) - Understand config system
- [Quick Start](../../README.md#-quick-start) - Basic usage examples
