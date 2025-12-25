#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def validate_dataset(dataset_path: Path, json_filename: str = "input.json", sample_size: int = 0) -> bool:
    """
    Validate OCR dataset structure and image existence.
    """
    json_path = dataset_path / json_filename
    if not json_path.exists():
        # Try fallback names
        for fallback in ["ufo.json", "train.json", "val.json"]:
            if (dataset_path / fallback).exists():
                json_path = dataset_path / fallback
                break

        if not json_path.exists():
            print(f"Error: JSON annotation file not found in {dataset_path}")
            print(f"Expected {json_filename} or one of [ufo.json, train.json, val.json]")
            return False

    print(f"Loading annotations from {json_path}...")
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to parse JSON: {e}")
        return False

    if "images" not in data:
        print("Error: Root key 'images' missing in JSON")
        return False

    images = data["images"]
    total_images = len(images)
    print(f"Found {total_images} image entries.")

    missing_files = []
    invalid_polygons = []
    checked_count = 0

    image_names = list(images.keys())
    if sample_size > 0:
        image_names = image_names[:sample_size]
        print(f"Validating first {sample_size} samples...")

    for img_name in image_names:
        checked_count += 1
        img_info = images[img_name]

        # Check file existence
        img_path = dataset_path / "images" / img_name
        if not img_path.exists():
            # Try root if images/ subdir doesn't exist
            img_path_root = dataset_path / img_name
            if not img_path_root.exists():
                missing_files.append(img_name)

        # Check fields
        if "words" in img_info:
            for word_id, word_data in img_info["words"].items():
                if "points" in word_data:
                    points = word_data["points"]
                    if not isinstance(points, list):
                        invalid_polygons.append(f"{img_name}:{word_id} (not a list)")
                        continue
                    # Basic point validation (list of lists/tuples)
                    for p in points:
                        if not (isinstance(p, (list, tuple)) and len(p) == 2):
                            invalid_polygons.append(f"{img_name}:{word_id} (invalid point format)")
                            break

    print("\nValidation Report:")
    print(f"Checked: {checked_count}/{total_images}")

    status = True
    if missing_files:
        print(f"❌ Missing Image Files: {len(missing_files)}")
        if len(missing_files) < 10:
            for f in missing_files:
                print(f"  - {f}")
        else:
            print(f"  (First 10): {missing_files[:10]}")
        status = False
    else:
        print("✅ All image files found.")

    if invalid_polygons:
        print(f"❌ Invalid Polygons: {len(invalid_polygons)}")
        if len(invalid_polygons) < 10:
            for p in invalid_polygons:
                print(f"  - {p}")
        else:
            print(f"  (First 10): {invalid_polygons[:10]}")
        status = False
    else:
        print("✅ Polygon structures appear valid.")

    return status


def main():
    parser = argparse.ArgumentParser(description="Validate OCR Inference Console Dataset")
    parser.add_argument("--path", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--json", type=str, default="input.json", help="Name of annotation JSON file")
    parser.add_argument("--sample", type=int, default=0, help="Number of samples to validate (0 for all)")

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path {args.path} does not exist.")
        sys.exit(1)

    success = validate_dataset(args.path, args.json, args.sample)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
