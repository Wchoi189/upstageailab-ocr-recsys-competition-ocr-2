#!/usr/bin/env python3
"""
Verify GPU usage for rembg background removal.

This script checks if rembg is actually using GPU and provides detailed diagnostics.
"""

import sys
import time
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import numpy as np
from PIL import Image

try:
    from rembg import new_session, remove
    import onnxruntime as ort
except ImportError as e:
    print(f"Error importing rembg: {e}")
    sys.exit(1)


def check_gpu_availability():
    """Check if GPU is available."""
    print("=" * 80)
    print("GPU AVAILABILITY CHECK")
    print("=" * 80)

    # Check ONNX Runtime providers
    providers = ort.get_available_providers()
    print(f"Available ONNX providers: {providers}")
    print(f"CUDA available: {'CUDAExecutionProvider' in providers}")
    print(f"TensorRT available: {'TensorrtExecutionProvider' in providers}")

    return providers


def check_rembg_session():
    """Check rembg session provider usage."""
    print("\n" + "=" * 80)
    print("REMBG SESSION CHECK")
    print("=" * 80)

    # Create session
    print("Creating rembg session with 'silueta' model...")
    sess = new_session("silueta")

    # Check inner session
    if hasattr(sess, "inner_session"):
        inner = sess.inner_session
        providers = inner.get_providers()
        print(f"Session providers: {providers}")
        print(f"Active provider (first): {providers[0] if providers else 'None'}")

        if providers and "CUDAExecutionProvider" in providers[0]:
            print("✓ GPU (CUDA) is the active provider")
            return True, sess
        elif providers and "TensorrtExecutionProvider" in providers[0]:
            print("✓ TensorRT is the active provider")
            return True, sess
        else:
            print(f"⚠ CPU provider is active: {providers[0] if providers else 'Unknown'}")
            return False, sess
    else:
        print("⚠ Could not access inner session")
        return False, sess


def test_inference_with_monitoring(sess):
    """Test inference and monitor GPU usage."""
    print("\n" + "=" * 80)
    print("INFERENCE TEST")
    print("=" * 80)

    # Create test image
    test_img = Image.new("RGB", (640, 640), color="red")
    print(f"Test image size: {test_img.size}")

    # Run inference
    print("Running inference...")
    start = time.perf_counter()
    result = remove(test_img, session=sess)
    elapsed = time.perf_counter() - start

    print(f"Inference time: {elapsed:.3f}s")
    print(f"Result shape: {np.array(result).shape}")

    # Check session provider again
    if hasattr(sess, "inner_session"):
        inner = sess.inner_session
        providers = inner.get_providers()
        print(f"Providers after inference: {providers}")

    return elapsed


def main():
    """Main verification function."""
    print("\n" + "=" * 80)
    print("REMBG GPU USAGE VERIFICATION")
    print("=" * 80)

    # Check GPU availability
    providers = check_gpu_availability()

    if "CUDAExecutionProvider" not in providers:
        print("\n❌ CUDA provider not available!")
        print("Make sure:")
        print("  1. onnxruntime-gpu is installed: uv add onnxruntime-gpu")
        print("  2. cuDNN is installed: sudo apt-get install libcudnn9-cuda-12")
        print("  3. GPU is accessible: nvidia-smi")
        return 1

    # Check rembg session
    gpu_active, sess = check_rembg_session()

    if not gpu_active:
        print("\n⚠ Warning: GPU provider detected but may not be active")
        print("This could be due to:")
        print("  - Model operations falling back to CPU")
        print("  - Small image size (GPU overhead not worth it)")
        print("  - rembg internal fallback logic")

    # Test inference
    elapsed = test_inference_with_monitoring(sess)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"CUDA provider available: {'CUDAExecutionProvider' in providers}")
    print(f"Session using GPU: {gpu_active}")
    print(f"Inference time: {elapsed:.3f}s")

    if gpu_active:
        print("\n✓ GPU is configured and should be used by rembg")
        print("Note: For small images (640px), GPU speedup may be minimal")
        print("      due to CPU-GPU transfer overhead.")
    else:
        print("\n⚠ GPU may not be actively used")
        print("Check the session provider list above.")

    return 0 if gpu_active else 1


if __name__ == "__main__":
    sys.exit(main())

