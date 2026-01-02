
import pandas as pd
import os

def inspect_content(path, name):
    print(f"\n--- Inspecting {name} Content ---")
    df = pd.read_parquet(path)

    # Check raw paths
    print(f"Sample raw paths:")
    print(df['image_path'].head(3).values)

    # Check labels sample
    if 'labels' in df.columns:
        print(f"Sample labels (first row):")
        print(df['labels'].iloc[0])

    if 'texts' in df.columns:
        print(f"Sample texts (first row):")
        print(df['texts'].iloc[0])

def main():
    kie_path = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/export/baseline_kie/train.parquet"
    dp_path = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/export/baseline_dp/train.parquet"

    inspect_content(kie_path, "baseline_kie")
    inspect_content(dp_path, "baseline_dp")

if __name__ == "__main__":
    main()
