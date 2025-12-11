import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError

# Add project root to sys.path to import ocr
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    from ocr.validation.models import PolygonArray
except ImportError:
    print("Warning: Could not import ocr.validation.models. Validation will be limited.")
    PolygonArray = None

def validate_data(data_path, sample_count, output_file):
    base_path = Path(data_path)
    json_path = base_path / "jsons/val.json"
    images_dir = base_path / "images_val_canonical"

    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        sys.exit(1)

    print(f"Loading annotations from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "images" not in data:
        print("Error: 'images' key not found in JSON.")
        sys.exit(1)

    images_map = data["images"]
    total_images = len(images_map)
    print(f"Found {total_images} images in JSON.")

    images_to_check = list(images_map.keys())
    if sample_count > 0:
        images_to_check = images_to_check[:sample_count]

    print(f"Validating {len(images_to_check)} samples...")

    stats = {
        "images_checked": 0,
        "missing_images": 0,
        "invalid_records": 0,
        "status": "ok"
    }

    for filename in images_to_check:
        stats["images_checked"] += 1

        # 1. Check Image Existence
        img_path = images_dir / filename
        if not img_path.exists():
            print(f"Missing image: {filename}")
            stats["missing_images"] += 1
            stats["status"] = "error"
            continue

        # 2. Check Record Structure
        record = images_map[filename]
        if "words" not in record:
            print(f"Invalid record (missing 'words'): {filename}")
            stats["invalid_records"] += 1
            stats["status"] = "error"
            continue

        # 3. Validate Polygons
        if PolygonArray:
            for word_id, word_data in record["words"].items():
                if "points" not in word_data:
                    continue
                # Convert to numpy for validation logic reuse
                try:
                    pts = np.array(word_data["points"], dtype=np.float32)
                    # PolygonArray calls _validate_points
                    # We instantiate the model to trigger validation
                    PolygonArray(points=pts)
                except Exception as e:
                     print(f"Invalid polygon in {filename}, word {word_id}: {e}")
                     stats["invalid_records"] += 1
                     stats["status"] = "error"

    if output_file:
        with open(output_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Validation results written to {output_file}")

    print(f"Validation complete: {stats}")

    if stats["missing_images"] > 0 or stats["invalid_records"] > 0:
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset for OCR Inference Console")
    parser.add_argument("--path", required=True, help="Path to data/datasets directory")
    parser.add_argument("--sample", type=int, default=0, help="Number of samples to check (0 for all)")
    parser.add_argument("--out", help="Output JSON file for validation results")

    args = parser.parse_args()
    validate_data(args.path, args.sample, args.out)
