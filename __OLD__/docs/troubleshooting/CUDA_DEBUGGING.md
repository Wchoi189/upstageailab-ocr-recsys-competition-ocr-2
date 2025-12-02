# CUDA Error Debugging Guide

This guide helps you debug CUDA errors that may occur during training, particularly in WSL2 environments.

## Common CUDA Errors

### 1. Illegal Instruction Error
```
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

**Symptoms:**
- Training crashes with "illegal instruction" error
- Error occurs during CUDA operations (tensor creation, device transfer)
- Common in WSL2 environments

**Causes:**
- CUDA/PyTorch version incompatibility
- GPU driver issues
- WSL2 CUDA compatibility issues
- Memory corruption

## Debugging Methods

### Method 1: Use the Debug Script (Recommended)

The easiest way to enable CUDA debugging is to use the provided debug script:

```bash
# Run training with CUDA debugging enabled
./scripts/debug_cuda.sh +hardware=rtx3060_12gb_i5_16core batch_size=4
```

This script:
- Checks CUDA system information
- Verifies PyTorch CUDA support
- Enables synchronous CUDA operations (`CUDA_LAUNCH_BLOCKING=1`)
- Enables device-side assertions (`TORCH_USE_CUDA_DSA=1`)
- Runs a CUDA initialization test before training

### Method 2: Enable via Environment Variable

Set the `DEBUG_CUDA` environment variable before running training:

```bash
export DEBUG_CUDA=1
python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=4
```

### Method 3: Enable via Config

Add `debug_cuda: true` to your runtime configuration:

```yaml
runtime:
  debug_cuda: true
  auto_gpu_devices: false
  ddp_strategy: ddp_find_unused_parameters_false
  min_auto_devices: 1
```

Then run:
```bash
python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=4
```

### Method 4: Manual Environment Variables

Set CUDA debugging variables manually:

```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=4
```

## What These Flags Do

### `CUDA_LAUNCH_BLOCKING=1`
- Makes CUDA operations synchronous (waits for completion)
- Provides more accurate error stack traces
- Slows down training but helps identify the exact failing operation
- **Use when:** You need to find which CUDA operation is failing

### `TORCH_USE_CUDA_DSA=1`
- Enables device-side assertions in PyTorch
- Provides additional error checking
- **Use when:** You suspect memory corruption or invalid operations

## Diagnostic Information

When CUDA debugging is enabled, the training script will print:

```
[CUDA Info] CUDA available: True
[CUDA Info] CUDA device count: 1
[CUDA Info] Current device: 0
[CUDA Info] Device name: NVIDIA GeForce RTX 3060
[CUDA Info] CUDA version: 12.1
[CUDA Info] cuDNN version: 8900
[CUDA Info] CUDA initialization test passed
```

If the initialization test fails, you'll see:
```
[CUDA Warning] CUDA initialization test failed: <error message>
[CUDA Warning] Continuing anyway, but errors may occur
```

## Common Solutions

### Solution 1: Check CUDA/PyTorch Compatibility

Verify your CUDA and PyTorch versions are compatible:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
nvidia-smi  # Check driver version
```

**Compatibility:**
- PyTorch 2.8.0 requires CUDA 12.1+
- Ensure your GPU driver supports the CUDA version

### Solution 2: WSL2-Specific Issues

If running in WSL2, check:

1. **WSL2 CUDA Support:**
   ```bash
   nvidia-smi  # Should work in WSL2
   ```

2. **GPU Access:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Memory Issues:**
   - WSL2 may have memory limitations
   - Try reducing batch size
   - Check WSL2 memory allocation in `.wslconfig`

### Solution 3: Reduce Memory Usage

If errors occur due to memory issues:

```bash
# Reduce batch size
python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=2

# Increase gradient accumulation instead
python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=2 trainer.accumulate_grad_batches=4
```

### Solution 4: Clear CUDA Cache

Clear CUDA cache before training:

```python
import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

### Solution 5: Check for Other Processes

Ensure no other processes are using the GPU:

```bash
nvidia-smi  # Check GPU utilization
# Kill other processes if needed
```

## Performance Impact

**Note:** Enabling CUDA debugging (`CUDA_LAUNCH_BLOCKING=1`) will:
- **Slow down training** by 20-30% (operations are synchronous)
- **Provide better error messages** (exact failing operation)
- **Use more memory** (operations complete before next starts)

**Recommendation:**
- Use debugging flags only when investigating errors
- Disable for normal training runs

## Example: Debugging the Illegal Instruction Error

1. **Run with debugging enabled:**
   ```bash
   ./scripts/debug_cuda.sh +hardware=rtx3060_12gb_i5_16core batch_size=4
   ```

2. **Check the output:**
   - Look for `[CUDA Info]` messages
   - Check if initialization test passes
   - Note the exact operation that fails

3. **If initialization test fails:**
   - Check CUDA/PyTorch installation
   - Verify GPU driver compatibility
   - Try reinstalling PyTorch with correct CUDA version

4. **If initialization test passes but training fails:**
   - The error will show the exact CUDA operation
   - Check if it's a memory issue (reduce batch size)
   - Check if it's a model/operation issue (check model code)

## Getting Help

When reporting CUDA errors, include:

1. **System Information:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```

2. **Error Output:**
   - Full stack trace
   - `[CUDA Info]` messages
   - Any warnings

3. **Configuration:**
   - Hardware config used
   - Batch size
   - Any custom settings

4. **Environment:**
   - WSL2 or native Linux
   - Docker or local
   - CUDA debugging flags enabled

## Additional Resources

- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/notes/cuda.html)
- [WSL2 CUDA Support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [CUDA Error Codes](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html)
