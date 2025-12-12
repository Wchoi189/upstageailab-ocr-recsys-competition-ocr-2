# Troubleshooting Guide

## Common Issues

### Installation Issues

#### UV command not found
**Symptom:** `uv: command not found`

**Solution:**
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (check installation output for exact path)
export PATH="$HOME/.cargo/bin:$PATH"
```

#### Import errors after setup
**Symptom:** `ModuleNotFoundError` when running scripts

**Solution:** Always use `uv run` prefix
```bash
# ❌ Wrong
python runners/train.py

# ✅ Correct
uv run python runners/train.py
```

### Training Issues

#### CUDA out of memory
**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Option 1: Reduce batch size
uv run python runners/train.py data.batch_size=4

# Option 2: Use gradient accumulation
uv run python runners/train.py \
    data.batch_size=4 \
    trainer.accumulate_grad_batches=4

# Option 3: Use mixed precision
uv run python runners/train.py trainer.precision=16
```

#### Preprocessing maps not found
**Symptom:** Warning about missing preprocessed maps

**Solution:** Either run preprocessing or ignore (will auto-fallback to real-time)
```bash
# Run preprocessing for faster training
uv run python scripts/preprocess_maps.py
```

#### Slow training/validation
**Symptom:** Training is very slow

**Solutions:**
1. Run offline preprocessing (5-8x speedup)
2. Increase dataloader workers: `data.num_workers=8`
3. Use performance preset: `data/performance_preset=balanced`

### Configuration Issues

#### Config file not found
**Symptom:** `ConfigFileNotFoundError`

**Solution:** Use correct config paths after consolidation
```bash
# Old (deprecated)
uv run python runners/train.py preset/models=model_example

# New (correct)
uv run python runners/train.py model/presets=model_example
```

#### Hydra composition errors
**Symptom:** `ConfigCompositionException`

**Solution:** Check config paths in [CONFIG_ARCHITECTURE.md](../architecture/CONFIG_ARCHITECTURE.md)

### Inference Issues

#### Model checkpoint not loading
**Symptom:** `FileNotFoundError` or checkpoint errors

**Solution:** Use absolute paths
```bash
uv run python runners/predict.py \
    checkpoint_path="/full/path/to/checkpoint.ckpt"
```

#### UI not starting
**Symptom:** Streamlit or frontend won't start

**Solutions:**
```bash
# For Streamlit
uv run streamlit run ui/apps/inference/app.py

# For React frontend
cd apps/frontend && npm install && npm run dev

# For backend
uv run uvicorn apps.backend.main:app --reload
```

### GPU Issues

#### CUDA not available
**Symptom:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check PyTorch CUDA version matches drivers
3. Reinstall PyTorch with correct CUDA version

#### Multiple GPUs not detected
**Symptom:** Only using one GPU

**Solution:** Specify devices in config
```bash
uv run python runners/train.py trainer.devices=[0,1,2,3]
```

## Getting Help

If issues persist:

1. **Check Documentation**
   - [Installation Guide](installation.md)
   - [Training Guide](training.md)
   - [Configuration Guide](../architecture/CONFIG_ARCHITECTURE.md)

2. **Search Issues**
   - [GitHub Issues](https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-2/issues)

3. **Create New Issue**
   Include:
   - Error message (full traceback)
   - Steps to reproduce
   - Environment details:
     ```bash
     python --version
     uv --version
     nvidia-smi  # if GPU-related
     ```

## Debugging Tips

### Enable Debug Mode
```bash
# Verbose logging
uv run python runners/train.py hydra.verbose=true

# Debug config
uv run python runners/train.py debug=default
```

### Check Config Resolution
```bash
# Print resolved config
uv run python -c "
from hydra import compose, initialize_config_dir
from pathlib import Path
initialize_config_dir(config_dir=str(Path.cwd() / 'configs'), version_base=None)
cfg = compose(config_name='train')
print(cfg)
"
```

### Profile Performance
```bash
# Use performance profiling
uv run python runners/train.py callbacks=profiler
```
