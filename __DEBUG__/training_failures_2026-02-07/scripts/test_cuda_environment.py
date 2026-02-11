#!/usr/bin/env python3
"""
CUDA Environment Diagnostic Script
Captures PyTorch, CUDA, and GPU configuration details
"""

import sys
import subprocess
from pathlib import Path


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    print_section("CUDA Environment Diagnostic")

    # Python version
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}\n")

    # PyTorch info
    try:
        import torch
        print_section("PyTorch Information")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Capability: {torch.cuda.get_device_capability(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
    except Exception as e:
        print(f"Error getting PyTorch info: {e}")

    # NVIDIA Driver
    print_section("NVIDIA Driver Information")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"nvidia-smi failed: {result.stderr}")
    except FileNotFoundError:
        print("nvidia-smi not found")

    # CUDA Runtime (nvcc)
    print_section("CUDA Runtime (nvcc)")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"nvcc not found or failed: {result.stderr}")
    except FileNotFoundError:
        print("nvcc not found in PATH")

    # Docker GPU configuration
    print_section("Docker GPU Configuration")
    if Path('/.dockerenv').exists():
        print("Running in Docker container")
        try:
            result = subprocess.run(['env'], capture_output=True, text=True)
            cuda_vars = [line for line in result.stdout.split('\n')
                        if 'CUDA' in line or 'NVIDIA' in line or 'GPU' in line]
            if cuda_vars:
                print("CUDA-related environment variables:")
                for var in cuda_vars:
                    print(f"  {var}")
            else:
                print("No CUDA-related environment variables found")
        except Exception as e:
            print(f"Error checking environment: {e}")
    else:
        print("Not running in Docker")

    # Test basic CUDA operations
    print_section("CUDA Operation Tests")
    try:
        import torch
        if torch.cuda.is_available():
            print("Test 1: Create CUDA tensor")
            device = torch.device("cuda:0")
            tensor = torch.randn(10, device=device)
            print(f"  ✓ Created tensor shape {tensor.shape} on {tensor.device}")

            print("\nTest 2: Simple computation")
            result = tensor * 2 + 1
            print(f"  ✓ Computation successful")

            print("\nTest 3: Cleanup")
            del tensor, result
            torch.cuda.empty_cache()
            print(f"  ✓ Cleanup successful")

            print("\nTest 4: Multiple allocations")
            tensors = [torch.randn(100, 100, device=device) for _ in range(10)]
            print(f"  ✓ Created {len(tensors)} tensors")
            del tensors
            torch.cuda.empty_cache()
            print(f"  ✓ Cleanup successful")

            print("\n✅ All basic CUDA tests passed!")
        else:
            print("❌ CUDA not available - skipping tests")
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        import traceback
        traceback.print_exc()

    print_section("Summary")
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA Environment: READY")
            print(f"   PyTorch: {torch.__version__}")
            print(f"   CUDA: {torch.version.cuda}")
            print(f"   Devices: {torch.cuda.device_count()}")
        else:
            print("⚠️  CUDA not available in PyTorch")
    except:
        print("❌ PyTorch not properly installed")


if __name__ == "__main__":
    main()
