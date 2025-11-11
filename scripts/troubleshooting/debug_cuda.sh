#!/bin/bash
# CUDA Debugging Helper Script
# This script helps debug CUDA errors by setting appropriate environment variables
# and running training with enhanced error reporting.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== CUDA Debugging Helper ===${NC}"
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA may not be properly installed.${NC}"
    exit 1
fi

echo -e "${YELLOW}CUDA System Information:${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Set CUDA debugging environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export DEBUG_CUDA=1

echo -e "${GREEN}Enabled CUDA debugging flags:${NC}"
echo "  - CUDA_LAUNCH_BLOCKING=1 (synchronous CUDA operations)"
echo "  - TORCH_USE_CUDA_DSA=1 (device-side assertions)"
echo "  - DEBUG_CUDA=1 (enables additional debugging in training script)"
echo ""

# Check PyTorch CUDA availability
echo -e "${YELLOW}Checking PyTorch CUDA support:${NC}"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')

    # Test basic CUDA operation
    try:
        x = torch.zeros(1, device='cuda')
        torch.cuda.synchronize()
        print('CUDA test: PASSED')
    except Exception as e:
        print(f'CUDA test: FAILED - {e}')
        exit(1)
else:
    print('CUDA test: SKIPPED (CUDA not available)')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}PyTorch CUDA test failed. Please check your CUDA/PyTorch installation.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}All checks passed. Running training with CUDA debugging enabled...${NC}"
echo ""

# Run training with all arguments passed to this script
# The training script will automatically detect DEBUG_CUDA=1
exec python runners/train.py "$@"

