#!/usr/bin/env python3
"""Diagnostic script to test imports incrementally and identify hang source."""

import sys
import time

def test_import(module_name, description=""):
    """Test importing a module with timing."""
    print(f"\n{'='*60}")
    print(f"Testing: {module_name}")
    if description:
        print(f"Description: {description}")
    print(f"{'='*60}")

    start = time.time()
    try:
        exec(f"import {module_name}")
        elapsed = time.time() - start
        print(f"✓ SUCCESS: {module_name} imported in {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ FAILED: {module_name} after {elapsed:.2f}s")
        print(f"Error: {e}")
        return False

def main():
    print("=" * 60)
    print("IMPORT DIAGNOSTIC TEST")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")

    # Test basic imports first
    test_import("sys", "Standard library - system")
    test_import("pathlib", "Standard library - paths")

    # Test core scientific stack
    test_import("numpy", "NumPy - numerical computing")
    test_import("pandas", "Pandas - data frames")

    # Test PyTorch
    test_import("torch", "PyTorch - deep learning framework")

    # Test if CUDA is available
    print("\n" + "=" * 60)
    print("CUDA Availability Check")
    print("=" * 60)
    import torch
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

    # Test Lightning
    test_import("lightning", "PyTorch Lightning - training framework")
    test_import("lightning.pytorch", "PyTorch Lightning specific")

    # Test Transformers (often slow/problematic)
    test_import("transformers", "Hugging Face Transformers")

    # Test specific transformers components
    print("\n" + "=" * 60)
    print("Testing Transformers Components")
    print("=" * 60)
    try:
        from transformers import LayoutLMv3Processor
        print("✓ LayoutLMv3Processor imported")
    except Exception as e:
        print(f"✗ LayoutLMv3Processor failed: {e}")

    try:
        from transformers import AutoTokenizer
        print("✓ AutoTokenizer imported")
    except Exception as e:
        print(f"✗ AutoTokenizer failed: {e}")

    # Test other ML libraries
    test_import("torchmetrics", "TorchMetrics - metrics for PyTorch")
    test_import("torchvision", "TorchVision - vision utilities")
    test_import("PIL", "Pillow - image processing")

    # Test project modules
    print("\n" + "=" * 60)
    print("Testing Project Modules")
    print("=" * 60)

    from pathlib import Path
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        print(f"Added to path: {project_root}")

    test_import("ocr", "OCR package")
    test_import("ocr.data", "OCR data module")
    test_import("ocr.data.datasets", "OCR datasets module")
    test_import("ocr.data.datasets.kie_dataset", "KIE Dataset")
    test_import("ocr.models", "OCR models module")
    test_import("ocr.models.kie_models", "KIE models")
    test_import("ocr.lightning_modules", "Lightning modules")
    test_import("ocr.lightning_modules.kie_pl", "KIE PyTorch Lightning module")

    print("\n" + "=" * 60)
    print("ALL IMPORTS TESTED")
    print("=" * 60)

if __name__ == "__main__":
    main()
