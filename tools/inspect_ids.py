
import pandas as pd
import os

def inspect_ids(path, name):
    print(f"\n--- Inspecting {name} IDs ---")
    df = pd.read_parquet(path)

    if 'id' in df.columns:
        print(f"Sample IDs:")
        print(df['id'].head(5).values)
    else:
        print("No 'id' column found/.")

def main():
    kie_path = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/export/baseline_kie/train.parquet"
    dp_path = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/export/baseline_dp/train.parquet"

    inspect_ids(kie_path, "baseline_kie")
    inspect_ids(dp_path, "baseline_dp")

if __name__ == "__main__":
    main()
