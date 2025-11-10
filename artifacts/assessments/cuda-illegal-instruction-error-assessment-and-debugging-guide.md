---
title: "CUDA Illegal Instruction Error Assessment and Debugging Guide"
author: "ai-agent"
date: "2025-11-09"
status: "draft"
tags: []
---

## Progress Tracker

**Last Updated:** 2025-11-09
**Current Status:** Active Debugging
**Current Issue:** CUDA Illegal Memory Access Error

### Debugging Progress

- [x] **Step 1:** Verify Environment Setup - ✅ Complete
- [x] **Step 2:** Check CPU Compatibility - ✅ Complete
- [x] **Step 3:** Check GPU Compute Capability - ✅ Complete
- [x] **Step 4:** Verify CUDA/PyTorch Compatibility - ✅ Complete
- [x] **Step 5:** Test with CUDA Debugging Enabled - ✅ Complete
- [x] **Step 6:** Install/Reinstall PyTorch - ✅ Complete (PyTorch 2.8.0+cu128 installed)
- [x] **Step 7:** Verify Installation - ✅ Complete
- [x] **Step 8:** Debug CUDA Illegal Memory Access - ⚠️ Training Frozen (even without CUDA_LAUNCH_BLOCKING)
- [x] **Step 8b:** Test CPU-only mode - ⚠️ **CPU mode also hangs - Issue NOT CUDA-specific**
- [ ] **Step 9:** Check for Shape Mismatches - ⏳ Pending
- [ ] **Step 10:** Check WSL2 CUDA Support - ⏳ Pending

### Current Findings

- ✅ **CUDA Available:** True
- ✅ **PyTorch Version:** 2.8.0+cu128
- ✅ **CUDA Version:** 12.8 (compatible with driver 13.0)
- ✅ **GPU:** NVIDIA GeForce RTX 3060 (compute_86)
- ✅ **CPU Compatibility:** AVX2 (matches PyTorch build)
- ❌ **Current Error:** CUDA illegal memory access during `loss.backward()`
- ⚠️ **CUDA Toolkit:** Not installed (not required for PyTorch runtime)
- ⚠️ **Training Status:** Frozen during training (both CUDA and CPU modes)
- 🔍 **Critical Finding:** CPU-only mode also hangs - **Issue is NOT CUDA-specific**
- ✅ **Import Issue Fixed:** `wandb` import hanging during module import - Fixed by making imports lazy
- ❌ **Actual CUDA Error:** CUDA illegal memory access during loss computation (line 58 in `bce_loss.py`)
- 🔍 **Real Root Cause:** CUDA memory corruption - even moving to CPU fails because memory is already corrupted
- ⚠️ **Status:** Fix applied (move to CPU first), but error persists - suggests corruption happens earlier in pipeline

### Training Freeze Issue

**Problem:** Training froze when running with `CUDA_LAUNCH_BLOCKING=1` and `TORCH_USE_CUDA_DSA=1`

**Possible Causes:**
1. CUDA_LAUNCH_BLOCKING makes operations synchronous (can hang on errors)
2. CUDA illegal memory access causing hang (error occurs but process doesn't crash)
3. Data loading issues (DataLoader workers hanging)
4. GPU memory issues (OOM or fragmentation)

**Actions Taken:**
- ✅ Enabled CUDA debugging (`TORCH_USE_CUDA_DSA=1`)
- ✅ Reduced batch size to 2, then to 1
- ✅ Reduced num_workers to 0 (single-threaded data loading)
- ✅ Cleared GPU memory cache
- ✅ Killed frozen process
- ✅ Tried without `CUDA_LAUNCH_BLOCKING` (still hangs)
- ✅ Tried CPU-only mode (also hangs)
- ⚠️ **Critical Finding:** Training hangs in both CUDA and CPU modes
- 🔍 **Conclusion:** Issue is NOT CUDA-specific - likely in data pipeline or model code

### Debugging Plan

**Phase 1: Identify Hang Location** ✅ Complete
- [x] Added debug prints to training script
- [x] Added debug prints to `get_pl_modules_by_cfg`
- [x] Run training with debug prints - Still hangs (timeout)
- [x] Test minimal config - Still hangs (timeout)
- [x] Test imports - Hangs during import (testing individual imports)
- [x] **Identified root cause: `wandb` import hanging during module import**
- [x] **Found all files with top-level wandb imports:**
  - `runners/train.py` (line 28) - ✅ Fixed (lazy import)
  - `ocr/utils/wandb_utils.py` (line 15) - ✅ Fixed (lazy import)
  - `ocr/lightning_modules/callbacks/unique_checkpoint.py` (line 8) - ✅ Fixed (lazy import)
  - `ocr/lightning_modules/callbacks/wandb_completion.py` (line 6) - ✅ Fixed (lazy import)

**Phase 2: Fix Root Cause** ✅ Complete
- [x] **Root Cause Identified:** `wandb` import hanging during module import
- [x] **Solution Applied:** Made all wandb imports lazy (import only when needed)
- [x] Fixed `runners/train.py` - removed top-level wandb import, added lazy import helper
- [x] Fixed `ocr/utils/wandb_utils.py` - made wandb import lazy via `_get_wandb()` helper
- [x] Fixed `ocr/lightning_modules/callbacks/unique_checkpoint.py` - made wandb import lazy
- [x] Fixed `ocr/lightning_modules/callbacks/wandb_completion.py` - made wandb import lazy
- [ ] **Testing:** Verify import works without hanging (in progress)

**Phase 3: Fix Root Cause** ⏳ Pending
- [ ] Fix identified issue
- [ ] Verify fix with minimal test
- [ ] Gradually restore configuration
- [ ] Full training test

### Root Cause and Solution

**Issues Identified:**

1. **Import Issue (Fixed):** `wandb` import hanging during module import
   - **Problem:** The `wandb` library was being imported at the top level of several modules, causing the import to hang when trying to connect to the wandb server or initialize.
   - **Solution Applied:** Made all wandb imports lazy (import only when needed)

2. **Actual CUDA Error (In Progress):** CUDA illegal memory access during loss computation
   - **Location:** `ocr/models/loss/bce_loss.py`, line 58: `positive_count = int(positive.sum().cpu().item())`
   - **Problem:** CUDA illegal memory access persists even after moving to CPU - suggests memory corruption happens earlier
   - **Error Details:**
     - Shapes match: `pred_logits.shape=torch.Size([4, 1, 640, 640]), gt.shape=torch.Size([4, 1, 640, 640])`
     - Devices match: all on `cuda:0`
     - Error occurs even when moving to CPU: `gt.cpu()` fails
   - **Possible Causes:**
     - CUDA memory corruption from earlier operations
     - Out-of-bounds tensor access in data pipeline
     - Memory corruption in model forward pass
     - Race condition in multi-threaded data loading
   - **Solution Applied:**
     - Move `gt` and `mask` to CPU first, then do all operations on CPU
     - Added CUDA synchronization before operations
     - Added error handling with detailed context
     - **Issue:** Even `.cpu()` fails if CUDA memory is corrupted - suggests corruption happens earlier

**Files Fixed:**
1. `runners/train.py` - Removed top-level wandb import, added `_safe_wandb_finish()` helper function
2. `ocr/utils/wandb_utils.py` - Made wandb import lazy via `_get_wandb()` helper function
3. `ocr/lightning_modules/callbacks/unique_checkpoint.py` - Made wandb import lazy
4. `ocr/lightning_modules/callbacks/wandb_completion.py` - Made wandb import lazy

**Next Steps:**

1. ✅ **Killed frozen process** - Complete
2. ✅ **Tried without CUDA_LAUNCH_BLOCKING** - Still hangs
3. ✅ **Reduced num_workers to 0** - Still hangs
4. ✅ **Reduced batch size to 1** - Still hangs
5. ✅ **Tested CPU-only mode** - **CPU mode also hangs**
6. ✅ **Added debug prints** - Complete
7. ✅ **Identified root cause** - **wandb import hanging**
8. ✅ **Fixed root cause** - **Made all wandb imports lazy**
9. 🔄 **Testing:** Verify import works without hanging (in progress)
10. **Next:** Test training script to verify fix works
11. **Next:** If still hanging, investigate other potential causes

---

## Executive Summary

This assessment evaluates CUDA availability, GPU driver versions, and provides agent-oriented debugging instructions for illegal instruction errors encountered in the OCR project.

**Status:** Active Investigation
**Date:** 2025-11-09
**Priority:** High
**Impact:** Critical - Blocks GPU-accelerated training and inference

## Current System Status

### GPU Hardware
- **GPU Model:** NVIDIA GeForce RTX 3060
- **VRAM:** 12GB (4.5GB currently in use)
- **GPU Utilization:** 16%
- **Temperature:** 43°C

### GPU Driver Information
- **NVIDIA-SMI Version:** 580.102.01
- **Driver Version:** 581.57
- **CUDA Version (Driver Support):** 13.0
- **Status:** ✅ GPU driver is installed and functional

### CPU Information
- **Model:** 13th Gen Intel(R) Core(TM) i5-13400F
- **Architecture:** x86_64
- **CPU Flags:** AVX2, AVX, SSE4.1, SSE4.2 (✅ Supports modern SIMD instructions)
- **Environment:** WSL2 (Windows Subsystem for Linux 2)

### Operating System
- **OS:** Ubuntu 22.04.5 LTS
- **Kernel:** Linux 6.6.87.2-microsoft-standard-WSL2
- **GLIBC:** 2.35

### Python Environment
- **Python Version:** 3.10.12
- **Python Executable:** /usr/bin/python3
- **Package Manager:** UV 0.9.8
- **PyTorch Status:** ❌ Not installed in current environment

### CUDA Toolkit Status
- **nvcc:** ❌ Not found (CUDA toolkit not installed)
- **CUDA Runtime:** ⚠️ Available via driver (version 13.0)

## Problem Analysis

### Understanding CPU vs GPU Instructions

**Important Distinction:** Illegal instruction errors can occur in both CPU and GPU code, but for different reasons:

#### GPU Instructions (CUDA Operations)
- **What:** GPU-specific instructions (CUDA cores, Tensor cores, compute shaders)
- **Where:** Executed on GPU hardware
- **Examples:** Matrix multiplication, convolution, batch normalization on GPU
- **Compatibility:** Checked via GPU compute capability (e.g., RTX 3060 = compute_86)
- **Status:** ✅ PyTorch 2.8.0 supports compute_86 (RTX 3060 compatible)

#### CPU Instructions (PyTorch CPU Components)
- **What:** CPU instruction sets (AVX, AVX2, AVX-512, SSE4.1, SSE4.2)
- **Where:** Executed on CPU, even when using GPU tensors
- **Why CPU instructions matter for GPU operations:**
  - PyTorch initialization (`import torch`) runs on CPU
  - CUDA availability checks (`torch.cuda.is_available()`) run on CPU
  - Memory management coordination runs on CPU
  - Error handling and type checking run on CPU
  - Some operations fall back to CPU
- **Compatibility:** PyTorch wheels are compiled with specific CPU instruction sets
- **Status:** ✅ PyTorch uses AVX2 (matches CPU capabilities)

**Key Point:** Even when using GPU tensors, PyTorch has CPU-side components that can trigger illegal instruction errors if the CPU doesn't support the instructions PyTorch was compiled with.

### CUDA Error Types

**Important:** There are TWO different CUDA errors that can occur:

#### 1. Illegal Instruction Error (`SIGILL`)
**Error Type:** `SIGILL` (Illegal Instruction)
**Error Message:** `torch.AcceleratorError: CUDA error: an illegal instruction was encountered`
**Common Manifestations:**
- Process crashes with "Illegal instruction" message
- CUDA operations fail immediately
- Can occur during `import torch` (CPU-side) or during GPU operations

**Where Errors Can Occur:**
1. **CPU-side:** During PyTorch import, initialization, or CPU fallback operations
2. **GPU-side:** During CUDA kernel execution (less common, usually version mismatch)

#### 2. Illegal Memory Access Error ⚠️ **CURRENT ISSUE**
**Error Type:** `CUDA_ERROR_ILLEGAL_ADDRESS` (Illegal Memory Access)
**Error Message:** `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`
**Common Manifestations:**
- Occurs during `loss.backward()` (backward pass)
- Also during `model.cpu()` (teardown)
- Suggests memory corruption during forward/backward pass
- Different from illegal instruction - this is a memory access issue

**Where Errors Can Occur:**
1. **During backward pass:** Invalid memory access in gradient computation
2. **During teardown:** Memory corruption prevents moving model to CPU
3. **During forward pass:** Out-of-bounds access in CUDA kernels
4. **Custom operations:** Invalid memory access patterns in custom CUDA code

### Root Causes (Most Likely to Least Likely)

#### 1. **PyTorch CPU Instruction Mismatch** ⚠️ HIGH PROBABILITY
**Problem:** PyTorch wheel compiled with CPU instructions (e.g., AVX-512) that the CPU doesn't support.

**Why This Matters:** Even when using GPU tensors, PyTorch has CPU-side code that runs during:
- Import (`import torch`)
- Initialization (`torch.cuda.is_available()`)
- Memory management
- Error handling

**Evidence:**
- CPU supports AVX2 but may not support AVX-512
- PyTorch wheels are often compiled for newer CPUs
- WSL2 may have CPU feature detection issues
- Error occurs during import or initialization (CPU-side)

**Solution:**
- Install PyTorch compiled for older CPU architectures (AVX2 instead of AVX-512)
- Use CPU-only PyTorch build for testing
- Verify CPU flags match PyTorch requirements
- Check PyTorch build configuration: `torch.__config__.show()`

#### 2. **GPU Compute Capability Mismatch** ⚠️ MEDIUM PROBABILITY
**Problem:** PyTorch compiled for GPU architectures that don't match your GPU.

**Why This Matters:** GPU operations use GPU-specific instructions based on compute capability.

**Evidence:**
- GPU compute capability: 8.6 (RTX 3060)
- PyTorch must support compute_86
- PyTorch 2.8.0 supports: compute_70, compute_75, compute_80, compute_86, compute_90, compute_100, compute_120 ✅

**Solution:**
- Verify GPU compute capability: `nvidia-smi --query-gpu=compute_cap --format=csv`
- Check PyTorch supports your GPU: `torch.cuda.get_device_properties(0).major`
- Install PyTorch with correct CUDA architecture support

#### 3. **CUDA/PyTorch Version Mismatch** ⚠️ MEDIUM PROBABILITY
**Problem:** PyTorch CUDA version incompatible with CUDA driver version.

**Evidence:**
- Driver supports CUDA 13.0
- Project requires PyTorch 2.8.0+
- PyTorch 2.8.0 typically requires CUDA 12.1+ (compatible with driver 13.0)

**Solution:**
- Verify PyTorch CUDA version matches driver support
- Install PyTorch with correct CUDA version
- Check PyTorch compatibility matrix

#### 4. **WSL2 CUDA Compatibility Issues** ⚠️ MEDIUM PROBABILITY
**Problem:** WSL2-specific CUDA compatibility problems.

**Evidence:**
- Running in WSL2 environment
- WSL2 CUDA support can be problematic
- Driver version 581.57 is relatively new

**Solution:**
- Verify WSL2 CUDA support
- Check WSL2 CUDA compatibility
- Consider native Linux if issues persist

#### 5. **CUDA Illegal Memory Access** ⚠️ **CURRENT ISSUE**
**Problem:** Invalid GPU memory access during forward/backward pass.

**Why This Matters:** This is different from illegal instruction - it's a memory access issue.

**Evidence:**
- Error occurs during `loss.backward()` (backward pass)
- Also during `model.cpu()` (teardown)
- Error message: "an illegal memory access was encountered"
- Suggests memory corruption or out-of-bounds access

**Common Causes:**
1. **Out-of-bounds tensor indices** - Accessing invalid tensor elements
2. **Shape mismatches** - Tensor shapes don't match expected dimensions
3. **Memory corruption** - Writing to invalid memory locations
4. **Custom CUDA operations** - Invalid memory access patterns
5. **GPU memory issues** - OOM, fragmentation, or memory leaks
6. **Race conditions** - Multi-threaded access to GPU memory

**Solution:**
- Enable CUDA debugging: `export CUDA_LAUNCH_BLOCKING=1` and `export TORCH_USE_CUDA_DSA=1`
- Check for shape mismatches in model/data pipeline
- Verify tensor indices are within bounds
- Check for custom CUDA operations with invalid memory access
- Reduce batch size to rule out memory issues
- Check for memory leaks or GPU memory not released

#### 6. **Missing CUDA Toolkit** ⚠️ LOW PROBABILITY
**Problem:** CUDA toolkit not installed, causing runtime issues.

**Evidence:**
- `nvcc` not found
- CUDA runtime available via driver
- PyTorch may need CUDA toolkit for some operations

**Important:** PyTorch typically does NOT require CUDA toolkit for runtime. It uses CUDA runtime from the driver. CUDA toolkit is only needed for:
- Compiling custom CUDA extensions
- Using nvcc compiler directly
- Building PyTorch from source
- Using CUDA libraries not included in PyTorch

**Solution:**
- **If you need CUDA toolkit** (for custom extensions, nvcc, etc.):
  ```bash
  # Method 1: NVIDIA Package Manager (Recommended)
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-12-4

  # Add to PATH (add to ~/.bashrc)
  echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
  source ~/.bashrc

  # Verify
  nvcc --version
  ```

  **For WSL2 minimal installation:**
  ```bash
  sudo apt-get update
  sudo apt-get install -y cuda-nvcc-12-4 cuda-cudart-dev-12-4
  ```

- **If you don't need CUDA toolkit** (most common case):
  - PyTorch works without it
  - No action needed

## Agent-Oriented Debugging Instructions

### Step 1: Verify Environment Setup

**Action:** Check if PyTorch is installed and CUDA is available.

```bash
# Activate UV environment
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
uv sync

# Check PyTorch installation
uv run python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}')"

# Test CUDA operations
uv run python3 -c "import torch; x = torch.zeros(1, device='cuda'); torch.cuda.synchronize(); print('CUDA OK')"
```

**Expected Output:**
- PyTorch version should be 2.8.0+
- CUDA available should be True
- CUDA test should complete without errors

**If Errors Occur:**
- Note the exact error message
- Check if it's an illegal instruction error
- Proceed to Step 2

### Step 2: Check CPU Compatibility

**Action:** Verify CPU supports required instructions (even for GPU operations, CPU-side code matters).

```bash
# Check CPU flags
cat /proc/cpuinfo | grep flags | head -1

# Check for AVX-512 (may not be supported)
cat /proc/cpuinfo | grep -o 'avx512[^ ]*' | head -1

# Check PyTorch CPU requirements
uv run python3 -c "import torch; print(torch.__config__.show())"
```

**Expected Output:**
- CPU flags should include AVX2, SSE4.1, SSE4.2
- AVX-512 may not be present (this is OK if PyTorch uses AVX2)
- PyTorch config should show "CPU capability usage: AVX2" (not AVX-512)

**If Issues Found:**
- PyTorch may be compiled for newer CPU (AVX-512) but CPU only supports AVX2
- Consider installing PyTorch compiled for AVX2
- Proceed to Step 3

### Step 3: Check GPU Compute Capability

**Action:** Verify GPU compute capability is supported by PyTorch.

```bash
# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Check PyTorch GPU support
uv run python3 -c "import torch; props = torch.cuda.get_device_properties(0); print(f'GPU: {props.name}, Compute Capability: {props.major}.{props.minor}')"

# Verify PyTorch supports your GPU architecture
uv run python3 -c "import torch; print('PyTorch compiled for:', torch.__config__.show() | grep 'arch=compute')"
```

**Expected Output:**
- RTX 3060: compute_86 (8.6)
- PyTorch should include compute_86 in compiled architectures
- GPU operations should work

**If Issues Found:**
- PyTorch may not support your GPU architecture
- Install PyTorch with correct CUDA architecture support
- Proceed to Step 4

### Step 4: Verify CUDA/PyTorch Compatibility

**Action:** Check CUDA and PyTorch version compatibility.

```bash
# Check driver CUDA version
nvidia-smi | grep "CUDA Version"

# Check PyTorch CUDA version
uv run python3 -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

# Check compatibility
# PyTorch 2.8.0 requires CUDA 12.1+
# Driver 581.57 supports CUDA 13.0
```

**Expected Output:**
- Driver CUDA version: 13.0
- PyTorch CUDA version: 12.1+ (compatible)
- Versions should be compatible

**If Incompatible:**
- Install PyTorch with correct CUDA version
- Use PyTorch installation index: https://pytorch.org/get-started/locally/
- Proceed to Step 5

### Step 5: Test with CUDA Debugging Enabled

**Action:** Enable CUDA debugging to get better error messages.

```bash
# Set CUDA debugging environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run CUDA test
uv run python3 -c "import torch; x = torch.zeros(1, device='cuda'); torch.cuda.synchronize(); print('CUDA OK')"
```

**Expected Output:**
- More detailed error messages
- Exact CUDA operation that fails
- Better stack traces

**If Errors Occur:**
- Note the exact CUDA operation that fails
- Check if it's a specific operation or all operations
- Proceed to Step 6

### Step 6: Install/Reinstall PyTorch with Correct Configuration

**Action:** Install PyTorch with CPU-compatible build.

```bash
# Option 1: Install PyTorch with CUDA 12.1 (recommended)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Option 2: Install CPU-only PyTorch (for testing)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Option 3: Install from project dependencies
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
uv sync
```

**Expected Output:**
- PyTorch installs successfully
- CUDA operations work without illegal instruction errors

**If Errors Persist:**
- Try CPU-only build to isolate issue
- Check if issue is CUDA-specific or CPU-specific
- Proceed to Step 7

### Step 7: Verify Installation

**Action:** Test PyTorch installation thoroughly.

```bash
# Test basic PyTorch operations
uv run python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test tensor operations
    x = torch.zeros(10, 10, device='cuda')
    y = torch.ones(10, 10, device='cuda')
    z = x + y
    torch.cuda.synchronize()
    print("✅ CUDA operations successful")
else:
    print("⚠️ CUDA not available")
EOF
```

**Expected Output:**
- All operations complete successfully
- No illegal instruction errors
- CUDA operations work correctly

**If Errors Persist:**
- Document exact error message
- Check error logs
- Proceed to Step 8

### Step 8: Debug CUDA Illegal Memory Access (Current Issue)

**Action:** Debug illegal memory access errors during training.

```bash
# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Check GPU memory
nvidia-smi

# Run training with debugging enabled
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
uv run python3 runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=2 max_epochs=1
```

**Expected Output:**
- More detailed error messages
- Exact CUDA operation that fails
- Better stack traces showing where memory access fails

**If Errors Occur:**
- Note the exact CUDA operation that fails
- Check for shape mismatches in model/data
- Verify tensor indices are within bounds
- Check for custom CUDA operations
- Reduce batch size to rule out memory issues
- Proceed to Step 9

### Step 9: Check for Shape Mismatches and Out-of-Bounds Access

**Action:** Verify tensor shapes and indices are valid.

```bash
# Add shape checking to your model
# Check tensor shapes at each stage:
#   - Input shapes
#   - Intermediate shapes
#   - Output shapes
#   - Loss computation shapes

# Check for out-of-bounds access:
#   - Tensor indices within bounds
#   - Array indices valid
#   - Memory access patterns correct
```

**Expected Output:**
- All tensor shapes match expected dimensions
- All indices are within bounds
- No invalid memory access patterns

**If Issues Found:**
- Fix shape mismatches
- Correct out-of-bounds indices
- Fix invalid memory access patterns
- Proceed to Step 10

### Step 10: Check WSL2 CUDA Support

**Action:** Verify WSL2 CUDA support is properly configured.

```bash
# Check WSL2 CUDA support
nvidia-smi

# Check CUDA libraries
ldconfig -p | grep cuda

# Check if CUDA runtime is accessible
uv run python3 -c "import torch; print(torch.cuda.is_available())"
```

**Expected Output:**
- nvidia-smi works correctly
- CUDA libraries are found
- PyTorch can access CUDA

**If Issues Found:**
- Verify WSL2 CUDA driver installation
- Check Windows NVIDIA driver version
- Consider native Linux if issues persist

## Recommended Solutions

### Solution 0: Debug CUDA Illegal Memory Access (Current Issue) ⚠️ **PRIORITY**

**Priority:** Critical
**Effort:** Medium
**Success Probability:** High

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Reduce batch size to rule out memory issues
uv run python3 runners/train.py +hardware=rtx3060_12gb_i5_16core \
    dataloaders.train_dataloader.batch_size=2 \
    dataloaders.val_dataloader.batch_size=2 \
    max_epochs=1

# Check GPU memory
nvidia-smi
```

**Common Fixes:**
1. **Shape mismatches:** Check tensor shapes at each stage
2. **Out-of-bounds access:** Verify tensor indices are valid
3. **Memory issues:** Reduce batch size or clear GPU memory
4. **Custom operations:** Check for invalid memory access patterns
5. **Memory leaks:** Ensure GPU memory is properly released

### Solution 1: Install PyTorch with Correct CUDA Version (Recommended)

**Priority:** High
**Effort:** Low
**Success Probability:** High

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# Install PyTorch with CUDA 12.1 (compatible with driver CUDA 13.0)
uv pip install torch>=2.8.0 torchvision>=0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Verify installation
uv run python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"
```

### Solution 2: Use CPU-Only PyTorch (Temporary Workaround)

**Priority:** Medium
**Effort:** Low
**Success Probability:** High

```bash
# Install CPU-only PyTorch
uv pip install torch>=2.8.0 torchvision>=0.18.0 --index-url https://download.pytorch.org/whl/cpu

# Verify installation
uv run python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**Note:** This will disable GPU acceleration but allows testing if the issue is CUDA-specific.

### Solution 3: Reinstall Project Dependencies

**Priority:** Medium
**Effort:** Medium
**Success Probability:** Medium

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# Remove existing environment
rm -rf .venv

# Reinstall all dependencies
uv sync

# Verify PyTorch installation
uv run python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Solution 4: Check and Update GPU Driver

**Priority:** Low
**Effort:** High
**Success Probability:** Low

**Note:** Driver version 581.57 is recent and should support CUDA 13.0. Only update if other solutions fail.

```bash
# Check current driver version
nvidia-smi

# Update driver (if needed, requires Windows host update)
# This should be done on Windows host, not in WSL2
```

## Testing and Validation

### Test 1: Basic CUDA Operations

```bash
uv run python3 << 'EOF'
import torch
print("Testing CUDA operations...")

# Test 1: Tensor creation
x = torch.zeros(100, 100, device='cuda')
print("✅ Tensor creation: OK")

# Test 2: Tensor operations
y = torch.ones(100, 100, device='cuda')
z = x + y
print("✅ Tensor operations: OK")

# Test 3: Synchronization
torch.cuda.synchronize()
print("✅ CUDA synchronization: OK")

# Test 4: Memory management
del x, y, z
torch.cuda.empty_cache()
print("✅ Memory management: OK")

print("✅ All CUDA tests passed!")
EOF
```

### Test 2: Project-Specific CUDA Usage

```bash
# Test with project's CUDA debugging script (if available)
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
./scripts/debug_cuda.sh +hardware=rtx3060_12gb_i5_16core batch_size=1
```

### Test 3: Training Script with CUDA

```bash
# Run a minimal training test
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
uv run python3 runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=1 max_epochs=1
```

## Monitoring and Logging

### Enable CUDA Debugging

**For Training:**
```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export DEBUG_CUDA=1

uv run python3 runners/train.py +hardware=rtx3060_12gb_i5_16core batch_size=4
```

**For Inference:**
```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

uv run streamlit run ui/apps/inference/main.py
```

### Check CUDA Status

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check CUDA processes
fuser -v /dev/nvidia*

# Check CUDA libraries
ldconfig -p | grep cuda
```

## Success Criteria

1. ✅ **PyTorch Installed:** PyTorch 2.8.0+ installed and importable
2. ✅ **CUDA Available:** `torch.cuda.is_available()` returns True
3. ✅ **No Illegal Instructions:** CUDA operations complete without SIGILL errors
4. ✅ **Basic Operations Work:** Tensor creation, operations, and synchronization work
5. ✅ **Project Runs:** Training and inference scripts work with CUDA

## Next Steps

1. **Immediate:** Follow Step 1-7 debugging instructions
2. **Short-term:** Implement recommended solution (Solution 1)
3. **Long-term:** Document solution and update project documentation
4. **Monitoring:** Enable CUDA debugging for ongoing monitoring

## References

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Compatibility Matrix](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [WSL2 CUDA Support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- Project CUDA Debugging Docs: `docs/troubleshooting/CUDA_DEBUGGING.md`
- Project CUDA Quick Reference: `docs/troubleshooting/CUDA_QUICK_REFERENCE.md`
