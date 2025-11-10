#!/usr/bin/env python3
"""Minimal test to isolate the crash point.

This script tests each component individually to find the exact failure point.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

print("=" * 80)
print("MINIMAL INFERENCE TEST - Finding exact crash point")
print("=" * 80)

# Test 1: Basic imports
print("\n[1/8] Testing imports...")
try:
    import cv2
    from PIL import Image

    print("  ✓ Basic imports OK")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Inference engine import
print("\n[2/8] Testing inference engine import...")
try:
    from ui.utils.inference.engine import InferenceEngine

    print("  ✓ InferenceEngine import OK")
except Exception as e:
    print(f"  ✗ InferenceEngine import failed: {e}")
    sys.exit(1)

# Test 3: Find test files
print("\n[3/8] Finding test files...")
checkpoint = Path("./outputs/transforms_test-dbnetpp-dbnetpp_decoder-resnet18/checkpoints/epoch-18_step-003895.ckpt")
test_image = Path("data/datasets/LOW_PERFORMANCE_IMGS_canonical/drp.en_ko.in_house.selectstar_003949.jpg")

if not checkpoint.exists():
    checkpoints = list(Path(".").glob("outputs/**/checkpoints/*.ckpt"))
    if checkpoints:
        checkpoint = checkpoints[0]
    else:
        print("  ✗ No checkpoints found")
        sys.exit(1)

if not test_image.exists():
    images = list(Path(".").glob("data/**/*.jpg"))
    if images:
        test_image = images[0]
    else:
        print("  ✗ No test images found")
        sys.exit(1)

print(f"  ✓ Checkpoint: {checkpoint}")
print(f"  ✓ Test image: {test_image}")

# Test 4: Create engine
print("\n[4/8] Creating InferenceEngine...")
try:
    engine = InferenceEngine()
    print(f"  ✓ Engine created, device: {engine.device}")
except Exception as e:
    print(f"  ✗ Engine creation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Load model
print("\n[5/8] Loading model (this may take 10-30 seconds)...")
try:
    start = time.time()
    success = engine.load_model(str(checkpoint), None)
    elapsed = time.time() - start
    if success:
        print(f"  ✓ Model loaded in {elapsed:.2f}s")
    else:
        print("  ✗ Model loading returned False")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Model loading failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 6: Run inference
print("\n[6/8] Running inference (this may take 5-15 seconds)...")
try:
    start = time.time()
    result = engine.predict_image(
        str(test_image),
        binarization_thresh=0.3,
        box_thresh=0.4,
        max_candidates=300,
        min_detection_size=5,
    )
    elapsed = time.time() - start

    if result:
        polygons = result.get("polygons", "")
        num_polygons = len(polygons.split("|")) if polygons else 0
        print(f"  ✓ Inference completed in {elapsed:.2f}s")
        print(f"    - Detected {num_polygons} polygons")
    else:
        print("  ✗ Inference returned None")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Inference failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 7: Load image for display
print("\n[7/8] Loading image array...")
try:
    image = cv2.imread(str(test_image))
    if image is None:
        raise ValueError(f"Could not load image: {test_image}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"  ✓ Image loaded: {image_rgb.shape}, dtype: {image_rgb.dtype}")
except Exception as e:
    print(f"  ✗ Image loading failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 8: Simulate Streamlit image display (without Streamlit)
print("\n[8/8] Testing PIL image operations...")
try:
    from PIL import Image, ImageDraw

    pil_image = Image.fromarray(image_rgb)
    print(f"  ✓ PIL Image created: {pil_image.size}, mode: {pil_image.mode}")

    # Test drawing polygons
    draw = ImageDraw.Draw(pil_image, "RGBA")

    polygons = result["polygons"].split("|")
    print(f"  ✓ Will draw {len(polygons)} polygons")

    for idx, polygon_str in enumerate(polygons[:5]):  # Test first 5 only
        import re

        tokens = re.findall(r"-?\d+(?:\.\d+)?", polygon_str)
        if len(tokens) >= 8 and len(tokens) % 2 == 0:
            coords = [float(token) for token in tokens]
            points = [(int(x), int(y)) for x, y in zip(coords[0::2], coords[1::2], strict=False)]
            draw.polygon(points, outline=(255, 0, 0, 255), fill=(255, 0, 0, 30))

    print(f"  ✓ Drew {min(5, len(polygons))} test polygons")

    # Save test image
    output_path = Path("/tmp/test_inference_output.png")
    pil_image.save(output_path)
    print(f"  ✓ Saved test image to {output_path}")

except Exception as e:
    print(f"  ✗ PIL operations failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - Components work outside Streamlit")
print("=" * 80)
print("\nConclusion:")
print("  If this script works but Streamlit crashes, the issue is:")
print("  1. Streamlit-specific (session state, caching, threading)")
print("  2. Multiple inference accumulation")
print("  3. Streamlit's image rendering")
print("\nNext: Run this script and report if it succeeds or where it fails.")
