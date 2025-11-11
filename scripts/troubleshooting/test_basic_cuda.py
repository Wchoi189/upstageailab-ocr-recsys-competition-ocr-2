#!/usr/bin/env python3
"""
Test basic CUDA operations to diagnose GPU stability issues.
This isolates GPU problems from multiprocessing/wandb issues.
"""

import torch
import torch.nn as nn

print("=" * 60)
print("CUDA Basic Operations Test")
print("=" * 60)

# Test 1: CUDA availability
print("\n[Test 1] CUDA Availability")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA version: {torch.version.cuda}")
print(f"  cuDNN version: {torch.backends.cudnn.version()}")
print(f"  Device count: {torch.cuda.device_count()}")
print(f"  Device name: {torch.cuda.get_device_name(0)}")

# Test 2: Simple tensor operations
print("\n[Test 2] Simple Tensor Operations")
try:
    x = torch.randn(10, 10, device='cuda')
    y = x + x
    z = y.sum()
    print(f"  ✓ Addition works: {z.item():.2f}")
except Exception as e:
    print(f"  ✗ Addition failed: {e}")
    exit(1)

# Test 3: Convolution (forward)
print("\n[Test 3] Convolution Forward Pass")
try:
    x = torch.randn(4, 3, 224, 224, device='cuda')
    conv = nn.Conv2d(3, 64, 3, padding=1).cuda()
    y = conv(x)
    print(f"  ✓ Convolution works: {y.shape}")
except Exception as e:
    print(f"  ✗ Convolution failed: {e}")
    exit(1)

# Test 4: Backward pass (THIS IS WHERE IT CRASHES)
print("\n[Test 4] Backward Pass")
try:
    x = torch.randn(4, 3, 224, 224, device='cuda', requires_grad=True)
    conv = nn.Conv2d(3, 64, 3, padding=1).cuda()
    y = conv(x)
    loss = y.sum()
    loss.backward()
    print(f"  ✓ Backward pass works")
    print(f"  ✓ Gradient shape: {x.grad.shape}")
except Exception as e:
    print(f"  ✗ Backward pass failed: {e}")
    print(f"\n  This is the root cause!")
    exit(1)

# Test 5: Multiple backward passes (stress test)
print("\n[Test 5] Repeated Backward Passes (10 iterations)")
try:
    conv = nn.Conv2d(3, 64, 3, padding=1).cuda()
    optimizer = torch.optim.Adam(conv.parameters())

    for i in range(10):
        x = torch.randn(4, 3, 224, 224, device='cuda')
        y = conv(x)
        loss = y.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Iteration {i+1}/10 ✓")

    print(f"  ✓ All iterations passed")
except Exception as e:
    print(f"  ✗ Failed at iteration {i+1}: {e}")
    exit(1)

print("\n" + "=" * 60)
print("All CUDA tests passed! ✓")
print("GPU is working correctly.")
print("=" * 60)
