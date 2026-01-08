import pandas as pd
from pathlib import Path
import glob

def merge_kie_datasets():
    # Target files pattern
    pattern = "data/kie_datasets/baseline_train*.parquet"
    files = glob.glob(pattern)
    
    if not files:
        print("No files found.")
        return
        
    print(f"Found {len(files)} files to merge: {files}")
    
    dfs = []
    for f in files:
        print(f"Reading {f}...")
        dfs.append(pd.read_parquet(f))
        
    merged_df = pd.concat(dfs, ignore_index=True)
    initial_count = len(merged_df)
    
    # Deduplicate based on image_path or id
    merged_df = merged_df.drop_duplicates(subset=['image_path'], keep='last')
    
    
    # Transform S3 paths to Relative Paths (Project Root relative)
    relative_base = "data/datasets/images/train"
    print(f"Converting paths to relative: {relative_base}")
    
    def transform_path(s3_path):
        filename = Path(s3_path).name
        return f"{relative_base}/{filename}"
        
    merged_df['image_path'] = merged_df['image_path'].apply(transform_path)
    
    final_count = len(merged_df)
    
    print(f"Merged {initial_count} rows -> {final_count} unique rows.")
    
    output_path = "data/kie_datasets/merged_baseline_train.parquet"
    merged_df.to_parquet(output_path)
    print(f"Saved merged dataset to {output_path}")

if __name__ == "__main__":
    merge_kie_datasets()
