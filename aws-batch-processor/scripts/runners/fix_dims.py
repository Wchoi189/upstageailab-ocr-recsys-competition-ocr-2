import pandas as pd
from PIL import Image
import os
from pathlib import Path

def fix_dims(parquet_path, image_base_dir):
    print(f"Fixing dimensions for {parquet_path} using images in {image_base_dir}")
    if not os.path.exists(parquet_path):
        print(f"ERROR: File not found: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)

    if 'width' not in df.columns:
        df['width'] = 0
    if 'height' not in df.columns:
        df['height'] = 0

    updated_count = 0

    for idx, row in df.iterrows():
        # Only update if missing
        if row['width'] > 0 and row['height'] > 0:
            continue

        filename = os.path.basename(row['image_path'])
        local_path = os.path.join(image_base_dir, filename)

        if os.path.exists(local_path):
            with Image.open(local_path) as img:
                df.at[idx, 'width'] = img.width
                df.at[idx, 'height'] = img.height
                updated_count += 1
        else:
            # Try recursive search if flat structure fails
            pass # Simplified for now

    print(f"  Updated dimensions for {updated_count} rows")
    df.to_parquet(parquet_path, index=False)
    print("  Saved updated file.")

def main():
    base_dir = "aws-batch-processor/data/export/baseline_dp"

    # Train
    fix_dims(
        f"{base_dir}/train.parquet",
        "data/raw/competition/baseline_text_detection/images/train"
    )

    # Val
    fix_dims(
        f"{base_dir}/val.parquet",
        "data/raw/competition/baseline_text_detection/images/val"
    )

if __name__ == "__main__":
    main()
