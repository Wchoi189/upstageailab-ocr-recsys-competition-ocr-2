import pandas as pd
import glob
from pathlib import Path
import os
from PIL import Image

def get_image_dimensions(image_path):
    try:
        if image_path.startswith('s3://'):
            # Just a fallback if local lookup fails, but we prefer local paths
            return 0, 0
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                return img.width, img.height
    except Exception:
        pass
    return 0, 0

def merge_and_export(split_pattern, export_path, dataset_name):
    print(f"Merging {dataset_name} from {split_pattern}...")
    files = sorted(glob.glob(split_pattern))
    if not files:
        print(f"WARNING: No files found for {dataset_name}")
        return

    dfs = [pd.read_parquet(f) for f in files]
    merged_df = pd.concat(dfs, ignore_index=True)

    # Deduplicate by image_filename just in case
    if 'image_filename' in merged_df.columns:
        initial_len = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['image_filename'])
        if len(merged_df) < initial_len:
            print(f"  Deduplicated {initial_len - len(merged_df)} rows")

    # Check dimensions
    if 'width' not in merged_df.columns or 'height' not in merged_df.columns:
        print("  Missing width/height columns. Adding them...")
        merged_df['width'] = 0
        merged_df['height'] = 0

    # Fill missing dimensions if possible
    missing_dims = merged_df[(merged_df['width'] == 0) | (merged_df['height'] == 0) | (merged_df['width'].isna())]
    if not missing_dims.empty:
        print(f"  Found {len(missing_dims)} rows with missing dimensions. Attempting to fill...")

        # We need to look up original local paths if current paths are S3
        # or if they are valid local paths

        for idx, row in missing_dims.iterrows():
            w, h = get_image_dimensions(row['image_path'])
            if w > 0 and h > 0:
                merged_df.at[idx, 'width'] = w
                merged_df.at[idx, 'height'] = h

    # Ensure export directory exists
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)

    merged_df.to_parquet(export_path, index=False)
    print(f"  Exported {dataset_name} to {export_path} ({len(merged_df)} rows)")

def main():
    base_dir = "aws-batch-processor/data"
    export_dir = f"{base_dir}/export/baseline_kie"

    # Train
    merge_and_export(
        f"{base_dir}/output/splits_train/*.parquet",
        f"{export_dir}/train.parquet",
        "Train"
    )

    # Test
    merge_and_export(
        f"{base_dir}/output/splits_test/*.parquet",
        f"{export_dir}/test.parquet",
        "Test"
    )

    # Validation (Already merged, but we'll move/rename it to be consistent)
    val_source = f"{base_dir}/output/baseline_val_canonical_pseudo_labels.parquet"
    if os.path.exists(val_source):
        merge_and_export(
            val_source,
            f"{export_dir}/val.parquet",
            "Validation"
        )
    else:
        print("WARNING: Validation source file not found")

if __name__ == "__main__":
    main()
