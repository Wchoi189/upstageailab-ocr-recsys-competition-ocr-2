---
title: "Recommended Settings for RTX 3060 12GB + Intel i5 16-core"
author: "ai-agent"
date: "2025-11-09"
status: "active"
tags: ["hardware", "recommendations", "rtx3060", "training", "optimization"]
---

# Recommended Settings for RTX 3060 12GB + Intel i5 16-core

## Hardware Specifications
- **GPU**: RTX 3060 12GB VRAM
- **CPU**: Intel i5 16-core processor

## Quick Start

### Option 1: Use Complete Hardware Configuration (Recommended)
```bash
python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=4
```

### Option 2: Use Individual Components
```bash
# Trainer only
python runners/train.py trainer=rtx3060_12gb

# Dataloader only
python runners/train.py dataloaders=rtx3060_16core batch_size=4

# Both
python runners/train.py trainer=rtx3060_12gb dataloaders=rtx3060_16core batch_size=4
```

## Recommended Settings Summary

### Trainer Configuration
| Setting | Value | Reason |
|---------|-------|--------|
| `precision` | `32-true` | FP32 full precision (no mixed precision due to compatibility) |
| `accumulate_grad_batches` | `2` | Effective batch size = batch_size × 2 |
| `gradient_clip_val` | `5.0` | Training stability |
| `benchmark` | `true` | Enable cudnn.benchmark for speed |
| `devices` | `1` | Single GPU setup |

### Dataloader Configuration
| Setting | Value | Reason |
|---------|-------|--------|
| `batch_size` | `4-6` | Start with 4, increase to 6 if memory allows (FP32 uses more memory) |
| `num_workers` | `12` | Optimized for 16-core CPU (leaves 4 cores for system) |
| `pin_memory` | `true` | Critical for GPU transfer speed |
| `persistent_workers` | `true` | Keep workers alive between epochs |
| `prefetch_factor` | `3` | Better GPU utilization |

### Dataset Configuration
| Setting | Value | Reason |
|---------|-------|--------|
| `preload_images` | `true` | Low memory cost, high speed benefit |
| `cache_transformed_tensors` | `false` | High memory, prioritize batch size |
| `cache_images` | `true` | Low memory, good speed benefit |
| `cache_maps` | `true` | Low memory, good speed benefit |

## Performance Expectations

### Training Speed
- **Baseline**: ~250s/epoch
- **Optimized**: ~70-80s/epoch (3-3.5x speedup)

### Memory Usage
- **GPU Memory**: 8-10GB / 12GB (67-83% utilization)
- **CPU Memory**: ~4-6GB (with 12 workers)

### Resource Utilization
- **GPU Utilization**: >90% during training
- **CPU Utilization**: 75-85% (12 workers active)
- **Effective Batch Size**: 8 (batch_size=4 × accumulate_grad_batches=2)

## Batch Size Recommendations

### Conservative (Start Here)
```bash
python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=4
```
- **GPU Memory**: ~8GB
- **Safe for most models**
- **Good starting point with FP32**

### Balanced (Recommended)
```bash
python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=5
```
- **GPU Memory**: ~9GB
- **Good balance of speed and memory**
- **Most common use case with FP32**

### Aggressive (If Memory Allows)
```bash
python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=6
```
- **GPU Memory**: ~10GB
- **Maximum performance with FP32**
- **Monitor for OOM errors**

## Monitoring Commands

### GPU Monitoring
```bash
# Continuous monitoring
watch -n 1 nvidia-smi

# Or use nvidia-smi in loop
nvidia-smi -l 1
```

### CPU Monitoring
```bash
# Use htop (if available)
htop

# Or use top
top

# Or use ps
ps aux | grep python
```

### Training Progress
```bash
# Monitor logs
tail -f outputs/ocr_training/logs/*.log

# Check GPU utilization during training
watch -n 1 nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

## Troubleshooting

### Issue: Out of Memory (OOM) Errors

**Symptoms**: Training crashes with CUDA OOM error

**Solutions**:
1. Reduce batch size:
   ```bash
   python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=6
   ```

2. Increase gradient accumulation:
   ```bash
   python runners/train.py +hardware=rtx3060_12gb_i5_16core trainer.accumulate_grad_batches=4
   ```

3. Disable image preloading:
   ```bash
   python runners/train.py +hardware=rtx3060_12gb_i5_16core datasets.train.config.preload_images=false
   ```

4. Reduce batch size further:
   ```bash
   python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=3
   ```

### Issue: Low GPU Utilization (<80%)

**Symptoms**: GPU is not fully utilized, training is slow

**Solutions**:
1. Increase num_workers:
   ```bash
   python runners/train.py +hardware=rtx3060_12gb_i5_16core dataloaders.train_dataloader.num_workers=14
   ```

2. Increase prefetch_factor:
   ```bash
   python runners/train.py +hardware=rtx3060_12gb_i5_16core dataloaders.train_dataloader.prefetch_factor=4
   ```

3. Enable image preloading:
   ```bash
   python runners/train.py +hardware=rtx3060_12gb_i5_16core datasets.train.config.preload_images=true
   ```

### Issue: High CPU Usage (100%)

**Symptoms**: CPU is maxed out, system becomes unresponsive

**Solutions**:
1. Reduce num_workers:
   ```bash
   python runners/train.py +hardware=rtx3060_12gb_i5_16core dataloaders.train_dataloader.num_workers=10
   ```

2. Reduce prefetch_factor:
   ```bash
   python runners/train.py +hardware=rtx3060_12gb_i5_16core dataloaders.train_dataloader.prefetch_factor=2
   ```

3. Check for other processes using CPU:
   ```bash
   ps aux | sort -k3 -r | head -10
   ```

### Issue: Training Accuracy Issues

**Symptoms**: Model accuracy is lower than expected

**Solutions**:
1. Increase gradient clipping:
   ```bash
   python runners/train.py +hardware=rtx3060_12gb_i5_16core trainer.gradient_clip_val=10.0
   ```

2. Check for NaN/Inf values in logs

3. Verify learning rate is appropriate for your model

## Advanced Tuning

### Maximum Performance Configuration
```bash
python runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  batch_size=6 \
  dataloaders.train_dataloader.num_workers=14 \
  dataloaders.train_dataloader.prefetch_factor=4 \
  datasets.train.config.preload_images=true
```

### Memory-Constrained Configuration
```bash
python runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  batch_size=3 \
  trainer.accumulate_grad_batches=4 \
  datasets.train.config.preload_images=false \
  datasets.train.config.cache_config.cache_transformed_tensors=false
```

### Balanced Configuration (Recommended)
```bash
python runners/train.py \
  +hardware=rtx3060_12gb_i5_16core \
  batch_size=4 \
  dataloaders.train_dataloader.num_workers=12 \
  dataloaders.train_dataloader.prefetch_factor=3
```

## Configuration Files Created

1. **`configs/trainer/rtx3060_12gb.yaml`**
   - Optimized trainer settings for RTX 3060 12GB
   - FP32 full precision (no mixed precision due to compatibility)
   - Gradient accumulation configured

2. **`configs/dataloaders/rtx3060_16core.yaml`**
   - Optimized dataloader settings for 16-core CPU
   - 12 workers configured
   - Persistent workers enabled

3. **`configs/hardware/rtx3060_12gb_i5_16core.yaml`**
   - Complete hardware-optimized configuration
   - Combines trainer and dataloader optimizations
   - Includes dataset caching settings

## References

- Performance benchmarks: `docs/performance/BENCHMARK_COMMANDS.md`
- FP16 validation: `docs/performance/FP16_VALIDATION.md`
- Trainer configs: `configs/trainer/`
- Dataloader configs: `configs/dataloaders/`

