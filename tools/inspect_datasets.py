
import pandas as pd
import os

def inspect_parquet(path, name):
    print(f"--- Inspecting {name} ---")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None, set()

    df = pd.read_parquet(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Try to find an image identifier column
    id_col = None
    if 'image_path' in df.columns:
        id_col = 'image_path'
    elif 'id' in df.columns:
        id_col = 'id'

    if id_col:
        # Normalize paths to just filenames for comparison
        filenames = df[id_col].apply(lambda x: os.path.basename(str(x))).unique()
        print(f"Unique {id_col} count: {len(filenames)}")
        print(f"Sample {id_col}s: {filenames[:5]}")
        return df, set(filenames)
    else:
        print("No obvious ID column found ('image_path' or 'id')")
        return df, set()

def main():
    kie_path = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/export/baseline_kie/train.parquet"
    dp_path = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/export/baseline_dp/train.parquet"

    _, kie_files = inspect_parquet(kie_path, "baseline_kie")
    _, dp_files = inspect_parquet(dp_path, "baseline_dp")

    if kie_files and dp_files:
        intersection = kie_files.intersection(dp_files)
        print(f"\nIntersection count: {len(intersection)}")
        if len(intersection) < 10 and len(intersection) > 0:
             print(f"Overlapping files: {list(intersection)}")
        elif len(intersection) == 0:
             print("NO OVERLAP FOUND.")

if __name__ == "__main__":
    main()
