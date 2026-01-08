import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def link_dataset(parquet_path, optimized_dir, output_path):
    print(f"Processing {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    new_paths = []
    found_count = 0

    optimized_dir_path = Path(optimized_dir)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        orig_path = Path(row["image_path"])
        filename = orig_path.name

        # Look for the file in the optimized directory
        # We assume flat structure in optimized_dir or matching structure?
        # optimize_images_v2 saved to `optimized_images_v2/baseline_train/filename`.
        # However, export/baseline_kie might contain train/val/test splits.
        # The optimized folder `baseline_train` likely processes ALL training images.
        # But what about val/test?
        # If `processed/baseline_train.parquet` only contained TRAIN images, then valid/test are missing from optimization!

        # Check if candidate exists
        candidate = optimized_dir_path / filename
        if candidate.exists():
            new_paths.append(str(candidate.absolute()))
            found_count += 1
        else:
            # Fallback or error?
            # If we missed optimizing val/test, we need to optimize them too.
            new_paths.append(None)

    print(f"Found {found_count}/{len(df)} optimized images.")

    if found_count < len(df):
        print("WARNING: Some images were not found in the optimized directory!")
        print("This suggests `processed/baseline_train.parquet` was incomplete or splits differ.")

    # Update df
    df["image_path"] = new_paths
    df = df.dropna(subset=["image_path"])

    df.to_parquet(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--optimized_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    link_dataset(args.input, args.optimized_dir, args.output)
