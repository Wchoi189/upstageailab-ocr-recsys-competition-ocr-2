import pandas as pd
import glob
from pathlib import Path
import os
from PIL import Image

def get_image_dimensions(image_path):
    try:
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

    # Deduplicate
    if 'image_filename' in merged_df.columns:
        initial_len = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['image_filename'])
        if len(merged_df) < initial_len:
            print(f"  Deduplicated {initial_len - len(merged_df)} rows")

    # Check dimensions
    if 'width' not in merged_df.columns or 'height' not in merged_df.columns:
        merged_df['width'] = 0
        merged_df['height'] = 0

    missing_dims = merged_df[(merged_df['width'] == 0) | (merged_df['height'] == 0) | (merged_df['width'].isna())]
    if not missing_dims.empty:
        print(f"  Found {len(missing_dims)} rows with missing dimensions. Attempting to fill...")
        # Assume images are in default location
        base_img_dir = "data/raw/competition/baseline_text_detection/images/test"
        for idx, row in missing_dims.iterrows():
            filename = os.path.basename(row['image_path'])
            local_path = os.path.join(base_img_dir, filename)
            w, h = get_image_dimensions(local_path)
            if w > 0:
                merged_df.at[idx, 'width'] = w
                merged_df.at[idx, 'height'] = h

    # Ensure export directory exists
    Path(export_path).parent.mkdir(parents=True, exist_ok=True)

    merged_df.to_parquet(export_path, index=False)
    print(f"  Exported {dataset_name} to {export_path} ({len(merged_df)} rows)")

def main():
    base_dir = "aws-batch-processor/data"
    export_dir = f"{base_dir}/export/baseline_dp"

    # Test (Document Parse)
    merge_and_export(
        f"{base_dir}/output/splits_test/baseline_test_p*_dp.parquet",
        f"{export_dir}/test.parquet",
        "Test (Document Parse)"
    )

if __name__ == "__main__":
    main()
