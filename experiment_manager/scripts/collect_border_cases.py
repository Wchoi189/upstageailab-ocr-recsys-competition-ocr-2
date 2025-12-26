#!/usr/bin/env python3
"""
Scaffold script to collect border cases for the experiment.
Generates a manifest of images to be processed.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Collect border cases")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiment_manager/experiments/20251218_1900_border_removal_preprocessing/outputs")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "border_cases_manifest.json"

    # Mock data for scaffolding
    manifest = {
        "experiment_id": "20251218_1900_border_removal_preprocessing",
        "cases": ["data/sample_001.jpg", "data/sample_002.jpg", "data/sample_003.jpg"],
    }

    print("Collecting border cases...")
    # In a real script, this would scan a directory or query a dataset

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest created at: {manifest_path}")


if __name__ == "__main__":
    main()
