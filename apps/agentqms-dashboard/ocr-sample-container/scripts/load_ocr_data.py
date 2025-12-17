#!/usr/bin/env python
"""
OCR Sample Data Loader and Validator
Loads and validates the COCO format annotations and images.
"""

import json
from pathlib import Path

from PIL import Image


class OCRSampleLoader:
    """Load and validate OCR sample dataset."""

    def __init__(self, data_path: str = "/app/data"):
        self.data_path = Path(data_path)
        self.annotations_file = self.data_path / "annotations.json"
        self.images_dir = self.data_path / "images"

        self.coco = None
        self.images = {}
        self.annotations = {}

    def load(self) -> bool:
        """Load annotations and validate dataset."""
        try:
            # Load COCO annotations
            with open(self.annotations_file) as f:
                self.coco = json.load(f)

            # Index images
            for img in self.coco.get("images", []):
                self.images[img["id"]] = img

            # Index annotations by image_id
            for ann in self.coco.get("annotations", []):
                img_id = ann["image_id"]
                if img_id not in self.annotations:
                    self.annotations[img_id] = []
                self.annotations[img_id].append(ann)

            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def validate(self) -> tuple[bool, list[str]]:
        """Validate dataset integrity."""
        issues = []

        if not self.coco:
            issues.append("Dataset not loaded")
            return False, issues

        # Check required fields
        if "images" not in self.coco:
            issues.append("Missing 'images' field in annotations")

        if "annotations" not in self.coco:
            issues.append("Missing 'annotations' field in annotations")

        # Validate image files exist
        for img_info in self.coco.get("images", []):
            img_path = self.images_dir / img_info["file_name"]
            if not img_path.exists():
                issues.append(f"Missing image file: {img_info['file_name']}")

        # Validate annotations reference valid images
        image_ids = {img["id"] for img in self.coco.get("images", [])}
        for ann in self.coco.get("annotations", []):
            if ann["image_id"] not in image_ids:
                issues.append(f"Annotation references non-existent image ID: {ann['image_id']}")

        return len(issues) == 0, issues

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return {
            "num_images": len(self.coco.get("images", [])),
            "num_annotations": len(self.coco.get("annotations", [])),
            "categories": len(self.coco.get("categories", [])),
            "total_area": sum(ann.get("area", 0) for ann in self.coco.get("annotations", [])),
            "avg_text_regions_per_image": (len(self.coco.get("annotations", [])) / max(len(self.coco.get("images", [])), 1)),
        }

    def load_image(self, image_id: int) -> tuple[Image.Image, str]:
        """Load a specific image by ID."""
        if image_id not in self.images:
            raise ValueError(f"Image ID {image_id} not found")

        img_info = self.images[image_id]
        img_path = self.images_dir / img_info["file_name"]

        return Image.open(img_path), img_info["file_name"]

    def get_annotations_for_image(self, image_id: int) -> list[dict]:
        """Get annotations for a specific image."""
        return self.annotations.get(image_id, [])

    def print_summary(self):
        """Print dataset summary."""
        if not self.coco:
            print("Dataset not loaded")
            return

        stats = self.get_statistics()
        print("\n" + "=" * 50)
        print("OCR SAMPLE DATASET SUMMARY")
        print("=" * 50)
        print(f"Images:               {stats['num_images']}")
        print(f"Annotations:          {stats['num_annotations']}")
        print(f"Categories:           {stats['categories']}")
        print(f"Avg Regions/Image:    {stats['avg_text_regions_per_image']:.2f}")
        print("=" * 50)

        print("\nIMAGES:")
        for img_info in self.coco.get("images", []):
            annotations = self.get_annotations_for_image(img_info["id"])
            print(f"  [{img_info['id']}] {img_info['file_name']}")
            print(f"       Size: {img_info['width']}x{img_info['height']}")
            print(f"       Regions: {len(annotations)}")
            for ann in annotations:
                print(f"         - {ann['text']}")


def main():
    """Load and validate dataset."""
    loader = OCRSampleLoader()

    print("Loading OCR sample dataset...")
    if not loader.load():
        print("Failed to load dataset")
        return 1

    print("Validating dataset...")
    valid, issues = loader.validate()
    if not valid:
        print("Validation failed:")
        for issue in issues:
            print(f"  ❌ {issue}")
        return 1

    print("✅ Dataset valid")
    loader.print_summary()
    return 0


if __name__ == "__main__":
    exit(main())
