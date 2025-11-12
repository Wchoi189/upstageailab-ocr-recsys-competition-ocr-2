"""Test script for comparison service integration.

This script tests the integrated comparison service with real images.
"""

import numpy as np

# Import comparison service
from ui.apps.unified_ocr_app.services.comparison_service import ComparisonService


def create_test_image() -> np.ndarray:
    """Create a simple test image."""
    # Create a 100x100 grayscale image with some pattern
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[25:75, 25:75] = [255, 255, 255]  # White square in center
    return image


def test_preprocessing_comparison():
    """Test preprocessing comparison with multiple configurations."""
    print("\n=== Testing Preprocessing Comparison ===")

    service = ComparisonService()
    image = create_test_image()

    # Create test configurations
    configurations = [
        {
            "label": "Config A - No processing",
            "params": {},
        },
        {
            "label": "Config B - With background removal",
            "params": {
                "background_removal": {
                    "enable": True,
                    "model": "u2net",
                },
            },
        },
    ]

    try:
        results = service.run_preprocessing_comparison(image, configurations)

        print("✓ Preprocessing comparison completed")
        print(f"  Number of results: {len(results)}")

        for i, result in enumerate(results):
            label = result.get("config_label", "Unknown")
            metrics = result.get("metrics", {})
            time_taken = result.get("processing_time", 0)

            print(f"\n  Result {i + 1}: {label}")
            print(f"    Processing time: {time_taken:.3f}s")
            print(f"    Metrics: {metrics}")

        return True

    except Exception as e:
        print(f"✗ Preprocessing comparison failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_inference_comparison():
    """Test inference comparison (will fail without checkpoint)."""
    print("\n=== Testing Inference Comparison (Expected to warn) ===")

    service = ComparisonService()
    image = create_test_image()

    # Create test configurations (without actual checkpoint)
    configurations = [
        {
            "label": "Config A - Default params",
            "params": {
                "text_threshold": 0.7,
                "link_threshold": 0.4,
                "low_text": 0.4,
                # Note: No checkpoint provided - should warn
            },
        },
    ]

    try:
        results = service.run_inference_comparison(image, configurations)

        print("✓ Inference comparison completed (with warnings)")
        print(f"  Number of results: {len(results)}")

        return True

    except Exception as e:
        print(f"✗ Inference comparison failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_end_to_end_comparison():
    """Test end-to-end comparison."""
    print("\n=== Testing End-to-End Comparison ===")

    service = ComparisonService()
    image = create_test_image()

    # Create test configurations
    configurations = [
        {
            "label": "Config A - Preprocessing only",
            "params": {
                "preprocessing": {},
                "inference": {},  # No checkpoint
            },
        },
    ]

    try:
        results = service.run_end_to_end_comparison(image, configurations)

        print("✓ End-to-end comparison completed")
        print(f"  Number of results: {len(results)}")

        for i, result in enumerate(results):
            label = result.get("config_label", "Unknown")
            metrics = result.get("metrics", {})
            time_taken = result.get("processing_time", 0)

            print(f"\n  Result {i + 1}: {label}")
            print(f"    Processing time: {time_taken:.3f}s")
            print(f"    Metrics: {metrics}")

        return True

    except Exception as e:
        print(f"✗ End-to-end comparison failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Comparison Service Integration Tests")
    print("=" * 60)

    results = {
        "preprocessing": test_preprocessing_comparison(),
        "inference": test_inference_comparison(),
        "end_to_end": test_end_to_end_comparison(),
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 60)

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
