import glob
import os

import pandas as pd


def count_splits():
    split_dir = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/archive/output/splits_train"
    files = glob.glob(os.path.join(split_dir, "*.parquet"))

    total_rows = 0
    print(f"Found {len(files)} split files.")

    for f in files:
        try:
            df = pd.read_parquet(f)
            count = len(df)
            print(f"{os.path.basename(f)}: {count} rows")
            total_rows += count
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"\nTotal rows in splits: {total_rows}")
    print("Expected (KIE train var): 3125")


if __name__ == "__main__":
    count_splits()
