#!/usr/bin/env python3
"""Quick test script for new features: pipeline status and gallery images endpoints."""

from pathlib import Path

import requests

# API base URL
BASE_URL = "http://127.0.0.1:8000"


def test_pipeline_preview():
    """Test pipeline preview endpoint creates a job."""
    print("Testing /api/pipelines/preview...")

    # Try to find a real image file first
    gallery_root = Path("data/datasets/images/val")
    test_image = None
    if gallery_root.exists():
        for img_path in gallery_root.glob("*.jpg"):
            if img_path.exists():
                test_image = str(img_path)
                break

    if test_image:
        # Use image_path if we found a real image
        payload = {
            "pipeline_id": "test-pipeline",
            "image_path": test_image,
            "params": {"autocontrast": True},
        }
    else:
        # Fallback to base64 with a minimal 1x1 pixel PNG
        # This is a valid 1x1 transparent PNG in base64
        minimal_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        payload = {
            "pipeline_id": "test-pipeline",
            "image_base64": minimal_png,
            "params": {"autocontrast": True},
        }

    response = requests.post(
        f"{BASE_URL}/api/pipelines/preview",
        json=payload,
        timeout=5,
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "job_id" in data, "Response should contain job_id"
    print(f"✅ Pipeline preview created job: {data['job_id']}")
    return data["job_id"]


def test_pipeline_status(job_id: str):
    """Test pipeline job status endpoint."""
    print(f"Testing /api/pipelines/status/{job_id}...")
    response = requests.get(f"{BASE_URL}/api/pipelines/status/{job_id}", timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["job_id"] == job_id, "Job ID should match"
    assert "status" in data, "Response should contain status"
    assert "pipeline_id" in data, "Response should contain pipeline_id"
    print(f"✅ Job status: {data['status']}")
    print(f"   Pipeline: {data['pipeline_id']}")
    print(f"   Backend: {data['routed_backend']}")
    return data


def test_gallery_images():
    """Test gallery images listing endpoint."""
    print("Testing /api/evaluation/gallery-images...")
    response = requests.get(f"{BASE_URL}/api/evaluation/gallery-images?limit=10", timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert isinstance(data, list), "Response should be a list"
    print(f"✅ Gallery images: {len(data)} images found")
    if data:
        print(f"   First image: {data[0]['name']} ({data[0]['size_mb']}MB)")
    return data


def test_gallery_root():
    """Test gallery root endpoint."""
    print("Testing /api/evaluation/gallery-root...")
    response = requests.get(f"{BASE_URL}/api/evaluation/gallery-root", timeout=5)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "gallery_root" in data, "Response should contain gallery_root"
    print(f"✅ Gallery root: {data['gallery_root']}")
    return data


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing New Features")
    print("=" * 60)
    print()

    try:
        # Test gallery endpoints first (don't require job creation)
        test_gallery_root()
        print()
        test_gallery_images()
        print()

        # Test pipeline endpoints
        job_id = test_pipeline_preview()
        print()
        test_pipeline_status(job_id)
        print()

        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server")
        print("   Make sure the server is running:")
        print("   uv run uvicorn services.playground_api.app:app --reload")
        return 1
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
