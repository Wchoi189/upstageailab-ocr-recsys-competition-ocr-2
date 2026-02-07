#!/usr/bin/env python3
"""
Visual Feature Padding Analyzer - Simplified Version

Analyzes ResNet encoder output for padding patterns.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from omegaConf import OmegaConf
import torch

def main():
    print("=" * 80)
    print("Visual Feature Padding Analyzer")
    print("=" * 80)

    # Load config manually
    config_root = Path(__file__).parents[3] / "configs"
    exp_cfg = OmegaConf.load(config_root / "experiment" / "rec_baseline_v1.yaml")
    domain_cfg = OmegaConf.load(config_root / "domain" / "recognition.yaml")
    cfg = OmegaConf.merge(domain_cfg, exp_cfg)

    print("\n[1/4] Loading data and model...")

    from ocr.core.data import OCRDataModule
    from ocr.pipelines.orchestrator import Orchestrator

    data_module = OCRDataModule(cfg)
    data_module.setup("fit")

    orchestrator = Orchestrator(cfg, data_module.dataset)
    model = orchestrator.setup_modules()
    model.eval()

    print(f"  ✓ Model: {model.__class__.__name__}")
    print(f"  ✓ Encoder: {model.encoder.__class__.__name__}")

    # Load sample batch
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))

    images = batch["images"]
    print(f"\n[2/4] Input images: {images.shape}")

    # Run encoder
    print("\n[3/4] Running encoder...")

    with torch.no_grad():
        features = model.encoder(images)

    if isinstance(features, (list, tuple)):
        visual_feat = features[-1]
        print(f"  - Using last feature map")
    else:
        visual_feat = features

    print(f"\n[4/4] Visual Features Analysis")
    print("=" * 80)

    b, c, h, w = visual_feat.shape
    sequence_len = h * w

    print(f"\nDimensions: [B={b}, C={c}, H={h}, W={w}] → S={sequence_len}")
    print(f"Global Stats: Mean={visual_feat.mean().item():.4f}, Std={visual_feat.std().item():.4f}")

    # Reshape to [B, S, C]
    visual_memory = visual_feat.permute(0, 2, 3, 1).flatten(1, 2)
    print(f"Reshaped: {visual_memory.shape}")

    # Position statistics
    position_means = visual_memory.mean(dim=(0, 2))
    position_stds = visual_memory.std(dim=(0, 2))

    zero_threshold = 0.01
    near_zero = (position_means.abs() < zero_threshold) & (position_stds < zero_threshold)
    num_near_zero = near_zero.sum().item()

    if num_near_zero > 0:
        print(f"\n⚠️  Found {num_near_zero}/{sequence_len} near-zero positions (potential padding)")
    else:
        print(f"\n✓ All {sequence_len} positions have active features")

    # Conclusion
    print("\n" + "=" * 80)
    print("Conclusion")
    print("=" * 80)

    if num_near_zero > 0:
        print("\n⚠️  RECOMMENDATION: Implement memory_key_padding_mask")
        print(f"   Found {num_near_zero} positions with near-zero activation")
    else:
        print("\n✓ No obvious padding detected, but memory_key_padding_mask")
        print("   is still recommended to prevent attention dilution.")


if __name__ == "__main__":
    main()
