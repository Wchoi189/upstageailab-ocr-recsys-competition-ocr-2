# CUDA Debugging Quick Reference

## Quick Start

### Enable CUDA Debugging (3 Methods)

**Method 1: Use Debug Script (Recommended)**
```bash
./scripts/debug_cuda.sh +hardware=rtx3060_12gb_i5_16core batch_size=4
```

**Method 2: Environment Variable**
```bash
export DEBUG_CUDA=1
python runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=4
```

**Method 3: Config File**
```yaml
runtime:
  debug_cuda: true
```

## Common Error: Illegal Instruction

**Error:**
```
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
```

**Quick Fix:**
```bash
# Enable debugging to get better error messages
./scripts/debug_cuda.sh +hardware=rtx3060_12gb_i5_16core batch_size=4
```

**Common Causes:**
- CUDA/PyTorch version mismatch
- WSL2 CUDA compatibility issues
- GPU driver issues
- Memory corruption

## Diagnostic Commands

**Check CUDA Status:**
```bash
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

**Test CUDA:**
```bash
python -c "import torch; x = torch.zeros(1, device='cuda'); torch.cuda.synchronize(); print('CUDA OK')"
```

**Check PyTorch CUDA:**
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Performance Impact

⚠️ **Warning:** CUDA debugging slows down training by 20-30%

- Use only when debugging errors
- Disable for normal training runs

## Full Documentation

See [CUDA_DEBUGGING.md](./CUDA_DEBUGGING.md) for complete guide.

