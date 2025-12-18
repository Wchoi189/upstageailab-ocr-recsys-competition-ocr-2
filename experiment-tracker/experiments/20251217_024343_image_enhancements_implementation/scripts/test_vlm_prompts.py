#!/usr/bin/env python3
"""Test VLM prompts for image enhancement experiment.

Tests all three new analysis modes:
- image_quality
- enhancement_validation
- preprocessing_diagnosis
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from AgentQMS.vlm.core.client import VLMClient
from AgentQMS.vlm.core.contracts import AnalysisMode, AnalysisRequest


def test_image_quality_mode():
    """Test image_quality analysis mode."""
    print("\n" + "=" * 80)
    print("TEST 1: Image Quality Analysis")
    print("=" * 80)

    # Use a test image from worst performers
    test_image = project_root / "data" / "zero_prediction_worst_performers"
    if not test_image.exists():
        print(f"❌ Test image directory not found: {test_image}")
        print("   Skipping image_quality test")
        return False

    # Get first available image
    images = list(test_image.glob("*.jpg"))
    if not images:
        print(f"❌ No test images found in: {test_image}")
        return False

    test_img = images[0]
    print(f"📷 Test image: {test_img.name}")

    try:
        client = VLMClient(backend_preference="openrouter")
        request = AnalysisRequest(
            mode=AnalysisMode.IMAGE_QUALITY,
            image_paths=[test_img],
            output_format="markdown",
        )

        print("⏳ Analyzing...")
        result = client.analyze(request)

        print("✅ Analysis completed successfully!")
        print(f"📝 Response length: {len(result.analysis_text)} characters")
        print("\n--- First 500 characters ---")
        print(result.analysis_text[:500])
        print("...")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_enhancement_validation_mode():
    """Test enhancement_validation analysis mode."""
    print("\n" + "=" * 80)
    print("TEST 2: Enhancement Validation")
    print("=" * 80)

    # Check if we have before/after comparison images
    comparison_dir = (
        project_root
        / "experiment-tracker"
        / "experiments"
        / "20251217_024343_image_enhancements_implementation"
        / "outputs"
        / "comparisons"
    )

    if not comparison_dir.exists():
        print(f"⚠️  Comparison directory not found: {comparison_dir}")
        print("   Creating a synthetic before/after test image...")

        # For testing, we'll use the same image twice (just to test the prompt)
        test_image = project_root / "data" / "zero_prediction_worst_performers"
        if not test_image.exists():
            print("❌ Cannot create test comparison - no test images available")
            return False

        images = list(test_image.glob("*.jpg"))
        if not images:
            print("❌ No test images found")
            return False

        test_img = images[0]
        print(f"📷 Using test image (simulated comparison): {test_img.name}")
        print("   Note: In real usage, this would be a side-by-side before/after image")

        try:
            client = VLMClient(backend_preference="openrouter")
            request = AnalysisRequest(
                mode=AnalysisMode.ENHANCEMENT_VALIDATION,
                image_paths=[test_img],
                output_format="markdown",
                initial_description="Test validation mode with single image (should be before/after comparison)",
            )

            print("⏳ Analyzing...")
            result = client.analyze(request)

            print("✅ Analysis completed successfully!")
            print(f"📝 Response length: {len(result.analysis_text)} characters")
            print("\n--- First 500 characters ---")
            print(result.analysis_text[:500])
            print("...")

            return True

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    # Use existing comparison if available
    comparisons = list(comparison_dir.glob("*.jpg")) + list(comparison_dir.glob("*.png"))
    if not comparisons:
        print(f"⚠️  No comparison images found in: {comparison_dir}")
        print("   Creating synthetic comparison for testing...")
        
        # Use a test image (pretend it's a before/after comparison)
        test_image = project_root / "data" / "zero_prediction_worst_performers"
        if not test_image.exists():
            print("❌ Cannot create test comparison - no test images available")
            return False
            
        images = list(test_image.glob("*.jpg"))
        if not images:
            print("❌ No test images found")
            return False
            
        test_img = images[0]
        print(f"📷 Using test image (simulated comparison): {test_img.name}")
    else:
        test_img = comparisons[0]
        print(f"📷 Test comparison: {test_img.name}")

    try:
        client = VLMClient(backend_preference="openrouter")
        request = AnalysisRequest(
            mode=AnalysisMode.ENHANCEMENT_VALIDATION,
            image_paths=[test_img],
            output_format="markdown",
        )

        print("⏳ Analyzing...")
        result = client.analyze(request)

        print("✅ Analysis completed successfully!")
        print(f"📝 Response length: {len(result.analysis_text)} characters")
        print("\n--- First 500 characters ---")
        print(result.analysis_text[:500])
        print("...")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_preprocessing_diagnosis_mode():
    """Test preprocessing_diagnosis analysis mode."""
    print("\n" + "=" * 80)
    print("TEST 3: Preprocessing Diagnosis")
    print("=" * 80)

    # Use a test image from worst performers
    test_image = project_root / "data" / "zero_prediction_worst_performers"
    if not test_image.exists():
        print(f"❌ Test image directory not found: {test_image}")
        print("   Skipping preprocessing_diagnosis test")
        return False

    images = list(test_image.glob("*.jpg"))
    if not images:
        print(f"❌ No test images found in: {test_image}")
        return False

    test_img = images[0]
    print(f"📷 Test image: {test_img.name}")
    print("   Context: Simulating a preprocessing failure for diagnosis")

    try:
        client = VLMClient(backend_preference="openrouter")
        request = AnalysisRequest(
            mode=AnalysisMode.PREPROCESSING_DIAGNOSIS,
            image_paths=[test_img],
            output_format="markdown",
            initial_description=(
                "Preprocessing applied: White-balance correction (gray-world algorithm). "
                "Expected: Neutral white background. "
                "Actual: Background still shows cream/yellow tint. "
                "Failure type: Partial correction."
            ),
        )

        print("⏳ Analyzing...")
        result = client.analyze(request)

        print("✅ Analysis completed successfully!")
        print(f"📝 Response length: {len(result.analysis_text)} characters")
        print("\n--- First 500 characters ---")
        print(result.analysis_text[:500])
        print("...")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all VLM prompt tests."""
    print("\n🧪 VLM Prompt Testing Suite")
    print("=" * 80)
    print("Testing three new analysis modes for image enhancement experiment:")
    print("  1. image_quality")
    print("  2. enhancement_validation")
    print("  3. preprocessing_diagnosis")
    print("=" * 80)

    results = {
        "image_quality": test_image_quality_mode(),
        "enhancement_validation": test_enhancement_validation_mode(),
        "preprocessing_diagnosis": test_preprocessing_diagnosis_mode(),
    }

    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    for mode, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {mode}")

    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All tests PASSED!")
        print("=" * 80)
        return 0
    else:
        print("⚠️  Some tests FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
