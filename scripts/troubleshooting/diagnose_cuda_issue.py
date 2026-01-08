#!/usr/bin/env python3
"""
Diagnostic script for BUG-20251110-002 CUDA errors.
Tests data loading and model forward pass without training.
"""

import argparse
import sys
from pathlib import Path

import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# Add project root to path before project imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ocr.core.utils.path_utils import setup_project_paths


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for diagnostics."""
    parser = argparse.ArgumentParser(description="Diagnose CUDA/data issues for the OCR pipeline.")
    parser.add_argument(
        "--config-name",
        default="train",
        help="Name of the Hydra config to compose (default: train).",
    )
    parser.add_argument(
        "--check-dataloader",
        action="store_true",
        help="Convenience flag to focus on dataloader diagnostics (all tests still run).",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip the data loading test.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip the model forward pass test.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style overrides, e.g. dataloaders.train_dataloader.batch_size=2",
    )
    return parser.parse_args()


def _compose_config(args: argparse.Namespace, extra_overrides: list[str] | None = None):
    """Compose a Hydra config using the provided CLI arguments."""
    overrides = list(args.overrides)
    if extra_overrides:
        overrides.extend(extra_overrides)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name=args.config_name, overrides=overrides, return_hydra_config=True)

    diagnostics_dir = (project_root / "outputs" / "diagnostics").resolve()
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    if "runtime" in cfg.hydra:
        cfg.hydra.runtime.cwd = str(project_root)
        cfg.hydra.runtime.output_dir = str(diagnostics_dir)
    if "job" in cfg.hydra:
        cfg.hydra.job.name = "diagnose_cuda_issue"
        cfg.hydra.job.config_name = args.config_name

    HydraConfig().set_config(cfg)
    return cfg


def test_cuda_setup():
    """Test CUDA is working correctly."""
    print("=" * 80)
    print("CUDA Setup Diagnostics")
    print("=" * 80)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")

        # Clear CUDA cache
        print("\nClearing CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ CUDA cache cleared")
    else:
        print("ERROR: CUDA not available!")
        return False

    return True


def test_basic_operations():
    """Test basic CUDA tensor operations."""
    print("\n" + "=" * 80)
    print("Basic CUDA Operations Test")
    print("=" * 80)

    try:
        # Simple tensor creation and operations
        x = torch.randn(4, 3, 224, 224, device="cuda", requires_grad=True)
        print(f"✓ Created tensor: {x.shape}")

        # Test sigmoid (our step function fix)
        y = torch.sigmoid(50 * x)
        print(f"✓ Sigmoid operation: {y.shape}")

        # Test backward pass
        loss = y.sum()
        loss.backward()
        print("✓ Backward pass successful")

        # Check for NaN/Inf
        if torch.isnan(x.grad).any():
            print("✗ NaN detected in gradients!")
            return False
        if torch.isinf(x.grad).any():
            print("✗ Inf detected in gradients!")
            return False

        print("✓ No NaN/Inf in gradients")

        # Clean up
        del x, y, loss
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return True

    except Exception as e:
        print(f"✗ Basic operations failed: {e}")
        return False


def test_data_loading(args: argparse.Namespace):
    """Test data loading without model."""
    print("\n" + "=" * 80)
    print("Data Loading Test")
    print("=" * 80)

    try:
        from hydra.utils import instantiate
        from torch.utils.data import DataLoader

        from ocr.data.datasets import get_datasets_by_cfg

        cfg = _compose_config(args)

        # Instantiate datasets and collate function to match training pipeline
        datasets = get_datasets_by_cfg(cfg.datasets, getattr(cfg, "data", None), cfg)
        train_dataset = datasets["train"]
        print(f"✓ Dataset created: {len(train_dataset)} samples")

        collate_fn = instantiate(cfg.collate_fn)
        if hasattr(collate_fn, "inference_mode"):
            collate_fn.inference_mode = False

        train_loader_cfg = OmegaConf.to_container(cfg.dataloaders.train_dataloader, resolve=True)
        if not isinstance(train_loader_cfg, dict):
            raise TypeError("train_dataloader config must resolve to a dict.")

        # Apply user overrides via CLI while keeping debugging-friendly defaults
        if train_loader_cfg.get("num_workers", 0) == 0:
            train_loader_cfg.pop("prefetch_factor", None)
            train_loader_cfg.pop("persistent_workers", None)

        train_loader = DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            **train_loader_cfg,
        )

        # Load first batch
        print("\nLoading first batch...")
        batch = next(iter(train_loader))

        # Validate batch
        print("\nBatch contents:")
        print(f"  images: {batch['images'].shape}, dtype={batch['images'].dtype}")
        print(f"    range: [{batch['images'].min().item():.4f}, {batch['images'].max().item():.4f}]")

        if "prob_maps" in batch:
            print(f"  prob_maps: {batch['prob_maps'].shape}, dtype={batch['prob_maps'].dtype}")
            print(f"    range: [{batch['prob_maps'].min().item():.4f}, {batch['prob_maps'].max().item():.4f}]")
            print(f"    non-zero elements: {(batch['prob_maps'] > 0).sum().item()}")

            if (batch["prob_maps"] == 0).all():
                print("  ⚠️  WARNING: All prob_maps are zero! This will cause training issues.")

        if "thresh_maps" in batch:
            print(f"  thresh_maps: {batch['thresh_maps'].shape}, dtype={batch['thresh_maps'].dtype}")
            print(f"    range: [{batch['thresh_maps'].min().item():.4f}, {batch['thresh_maps'].max().item():.4f}]")
            print(f"    non-zero elements: {(batch['thresh_maps'] > 0).sum().item()}")

        if "polygons" in batch:
            total_polygons = sum(len(polys) for polys in batch["polygons"])
            print(f"  polygons: {len(batch['polygons'])} samples, {total_polygons} total polygons")

            if total_polygons == 0:
                print("  ⚠️  WARNING: No polygons in batch! This will cause training issues.")

        # Check for NaN/Inf in batch
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    print(f"  ✗ NaN detected in {key}!")
                    return False
                if torch.isinf(value).any():
                    print(f"  ✗ Inf detected in {key}!")
                    return False

        print("\n✓ First batch loaded successfully")
        print("✓ No NaN/Inf detected in batch")

        # Try to move batch to CUDA
        print("\nMoving batch to CUDA...")
        {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        torch.cuda.synchronize()
        print("✓ Batch moved to CUDA successfully")

        return True

    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_forward(args: argparse.Namespace):
    """Test model forward pass without training."""
    print("\n" + "=" * 80)
    print("Model Forward Pass Test")
    print("=" * 80)

    try:
        cfg = _compose_config(
            args,
            extra_overrides=[
                "+hardware=rtx3060_12gb_i5_16core",
            ],
        )

        # Create model
        print("Creating model...")
        from ocr.core.models import get_model_by_cfg

        model = get_model_by_cfg(cfg.model)
        model = model.cuda()
        model.eval()  # Evaluation mode for testing
        print("✓ Model created and moved to CUDA")

        # Create dummy input
        print("\nCreating dummy input...")
        dummy_images = torch.randn(2, 3, 640, 640, device="cuda")
        dummy_prob_maps = torch.zeros(2, 1, 640, 640, device="cuda")
        dummy_thresh_maps = torch.zeros(2, 1, 640, 640, device="cuda")

        # Add a few non-zero values to ground truth maps
        dummy_prob_maps[:, :, 100:200, 100:200] = 1.0
        dummy_thresh_maps[:, :, 100:200, 100:200] = 0.5

        print("✓ Dummy input created")

        # Test forward pass (training mode)
        print("\nTesting forward pass (training mode)...")
        with torch.no_grad():
            result = model(
                images=dummy_images,
                prob_maps=dummy_prob_maps,
                thresh_maps=dummy_thresh_maps,
                return_loss=True,
            )

        print("\nModel output:")
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape if value.ndim > 0 else 'scalar'}")
                if value.ndim > 0:
                    print(f"    range: [{value.min().item():.6f}, {value.max().item():.6f}]")
                else:
                    print(f"    value: {value.item():.6f}")

                # Check for NaN/Inf
                if torch.isnan(value).any():
                    print(f"    ✗ NaN detected in {key}!")
                    return False
                if torch.isinf(value).any():
                    print(f"    ✗ Inf detected in {key}!")
                    return False
            elif isinstance(value, dict):
                print(f"  {key}: (dict with {len(value)} items)")
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        val = v.item() if v.ndim == 0 else f"tensor {v.shape}"
                        print(f"    {k}: {val}")

        print("\n✓ Forward pass successful")
        print("✓ No NaN/Inf in model output")

        # Test backward pass
        print("\nTesting backward pass...")
        model.train()
        model.zero_grad()

        result = model(
            images=dummy_images,
            prob_maps=dummy_prob_maps,
            thresh_maps=dummy_thresh_maps,
            return_loss=True,
        )

        loss = result.get("loss")
        if loss is None:
            print("✗ No loss in model output!")
            return False

        print(f"Loss value: {loss.item():.6f}")

        try:
            loss.backward()
            print("✓ Backward pass successful")
        except RuntimeError as e:
            if "FIND was unable to find an engine" in str(e):
                print(f"✗ cuDNN FIND error during backward: {e}")
                print("\nThis suggests:")
                print("  1. Incompatible tensor shapes or strides")
                print("  2. cuDNN version mismatch")
                print("  3. Memory corruption")
                return False
            else:
                raise

        # Check gradients
        print("\nChecking gradients...")
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                if torch.isnan(param.grad).any():
                    print(f"  ✗ NaN gradient in {name}")
                    return False
                if torch.isinf(param.grad).any():
                    print(f"  ✗ Inf gradient in {name}")
                    return False

        if has_grad:
            print("✓ Gradients computed successfully")
            print("✓ No NaN/Inf in gradients")
        else:
            print("⚠️  No gradients computed (might be expected)")

        return True

    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 80)
    print("BUG-20251110-002 CUDA Diagnostics")
    print("=" * 80)
    print()

    args = parse_args()

    # Ensure project paths (outputs, logs, etc.) exist before running diagnostics
    setup_project_paths()

    results: dict[str, bool] = {}

    results["CUDA Setup"] = test_cuda_setup()
    results["Basic Operations"] = test_basic_operations()

    if not args.skip_data:
        results["Data Loading"] = test_data_loading(args)
    else:
        results["Data Loading"] = True

    if not args.skip_model:
        results["Model Forward Pass"] = test_model_forward(args)
    else:
        results["Model Forward Pass"] = True

    print("\n" + "=" * 80)
    print("Diagnostic Results")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All diagnostics passed!")
        print("\nThe CUDA error is likely caused by:")
        print("  1. Specific batch configurations during training")
        print("  2. Gradient accumulation over multiple steps")
        print("  3. Interaction with PyTorch Lightning's training loop")
        print("\nNext steps:")
        print("  1. Try running with CUDA_LAUNCH_BLOCKING=1")
        print("  2. Reduce batch size to 1")
        print("  3. Check for memory leaks")
    else:
        print("✗ Some diagnostics failed")
        print("\nThe CUDA error is caused by:")
        failed_tests = [name for name, passed in results.items() if not passed]
        for test in failed_tests:
            print(f"  - {test}")
        print("\nFix these issues before proceeding with training.")

    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
