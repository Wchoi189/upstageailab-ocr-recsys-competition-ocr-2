#!/usr/bin/env python3
"""
Test the actual OCR model architecture to find which operation causes CUDA errors.
This isolates the problem to a specific layer or operation.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from ocr.models.decoder.fpn_decoder import FPNDecoder
from ocr.models.encoder.timm_backbone import TimmBackbone
from ocr.models.head.db_head import DBHead

print("=" * 70)
print("Testing OCR Model Architecture")
print("=" * 70)

# Test configuration
batch_size = 4
img_size = 224

print("\nTest Configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Image size: {img_size}x{img_size}")
print("  Device: cuda")

# Test 1: Encoder (ResNet18)
print("\n[Test 1] Testing Encoder (ResNet18)")
try:
    encoder = TimmBackbone(model_name="resnet18", pretrained=False).cuda()
    x = torch.randn(batch_size, 3, img_size, img_size, device="cuda")
    features = encoder(x)
    print("  ✓ Forward pass works")
    print(f"  Feature shapes: {[f.shape for f in features]}")

    # Test backward
    loss = sum(f.sum() for f in features)
    loss.backward()
    print("  ✓ Backward pass works")
except Exception as e:
    print(f"  ✗ Encoder failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 2: Decoder (FPN)
print("\n[Test 2] Testing Decoder (FPN)")
try:
    encoder = TimmBackbone(model_name="resnet18", pretrained=False).cuda()
    decoder = FPNDecoder(in_channels=[64, 128, 256, 512], out_channels=256).cuda()

    x = torch.randn(batch_size, 3, img_size, img_size, device="cuda", requires_grad=True)
    features = encoder(x)
    decoded = decoder(features)
    print("  ✓ Forward pass works")
    print(f"  Decoded shape: {decoded.shape}")

    # Test backward
    loss = decoded.sum()
    loss.backward()
    print("  ✓ Backward pass works")
except Exception as e:
    print(f"  ✗ Decoder failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 3: Head (DBHead)
print("\n[Test 3] Testing Head (DBHead)")
try:
    encoder = TimmBackbone(model_name="resnet18", pretrained=False).cuda()
    decoder = FPNDecoder(in_channels=[64, 128, 256, 512], out_channels=256).cuda()
    head = DBHead(in_channels=256, k=50).cuda()

    x = torch.randn(batch_size, 3, img_size, img_size, device="cuda", requires_grad=True)
    features = encoder(x)
    decoded = decoder(features)
    output = head(decoded, return_loss=False)
    print("  ✓ Forward pass works")
    print(f"  Output keys: {output.keys()}")

    # Test backward
    loss = output["binary_map"].sum()
    loss.backward()
    print("  ✓ Backward pass works")
except Exception as e:
    print(f"  ✗ Head failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Test 4: Full model with multiple iterations
print("\n[Test 4] Testing Full Model Pipeline (10 iterations)")
try:
    encoder = TimmBackbone(model_name="resnet18", pretrained=False).cuda()
    decoder = FPNDecoder(in_channels=[64, 128, 256, 512], out_channels=256).cuda()
    head = DBHead(in_channels=256, k=50).cuda()

    # Create optimizer
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)

    for i in range(10):
        optimizer.zero_grad()

        x = torch.randn(batch_size, 3, img_size, img_size, device="cuda")
        features = encoder(x)
        decoded = decoder(features)
        output = head(decoded, return_loss=False)

        # Simulate loss computation
        loss = output["binary_map"].sum()
        if "thresh" in output:
            loss = loss + output["thresh"].sum()
        if "thresh_binary" in output:
            loss = loss + output["thresh_binary"].sum()
        loss.backward()
        optimizer.step()

        print(f"  Iteration {i+1}/10 ✓ (loss: {loss.item():.2f})")

    print("  ✓ All iterations passed")
except Exception as e:
    print(f"  ✗ Failed at iteration {i+1}: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("All model tests passed! ✓")
print("Model architecture is working correctly.")
print("=" * 70)
