import glob
import os

import pandas as pd

# Configuration
EXPORT_DIR = "aws-batch-processor/data/export"
DATA_CATALOG = {
    "train": "data/datasets/images/train",
    "val": "data/datasets/images_val_canonical",
    "test": "data/datasets/images/test"
}

S3_PREFIX = "s3://ocr-batch-processing/images/"

def fix_paths(file_path):
    print(f"Processing {file_path}...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Determine split from filename or directory structure if possible,
    # but based on the file listing: baseline_kie/train.parquet, etc.
    # checking the parent folder name might be risky if they are named similarly,
    # but the filename itself 'train.parquet', 'val.parquet', 'test.parquet' matches the keys.

    filename = os.path.basename(file_path)
    split_name = filename.replace(".parquet", "")

    if split_name not in DATA_CATALOG:
        print(f"Skipping {file_path}: Unknown split '{split_name}'")
        return

    local_base_path = DATA_CATALOG[split_name]

    def replacer(path):
        if not isinstance(path, str):
            return path
        if path.startswith(S3_PREFIX):
            # Extract filename
            basename = os.path.basename(path)
            # Construct new local path
            new_path = os.path.join(local_base_path, basename)

            # Verify existence (optional but recommended in plan)
            # We check relative to the project root
            if not os.path.exists(new_path):
                # Try checking if it's just the basename in the dir
                # Sometimes s3 path might have extra subdirs?
                # Based on previous ls, files are directly in train/
                pass
                # verification can be expensive for all rows, let's strictly replace first
                # checking strict existence for *every* file might be slow if 4000 files, but 4000 is small.
                # Let's check correctness.

            return new_path
        return path

    # Check if image_path column exists
    if "image_path" not in df.columns:
        print(f"Skipping {file_path}: 'image_path' column missing")
        return

    # Apply replacement
    df["image_path"] = df["image_path"].apply(replacer)

    # Validation
    missing_files = []
    for idx, row in df.iterrows():
        path = row["image_path"]
        if isinstance(path, str) and not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        print(f"WARNING: {len(missing_files)} files not found locally for {file_path}")
        print(f"Example missing: {missing_files[0]}")
    else:
        print(f"All files verified locally for {file_path}")

    # Save back
    df.to_parquet(file_path)
    print(f"Updated {file_path}")

def main():
    # Find all parquet files in export dir
    patterns = [
        os.path.join(EXPORT_DIR, "**", "*.parquet")
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    print(f"Found {len(files)} parquet files.")

    for f in files:
        fix_paths(f)

if __name__ == "__main__":
    main()
