import pandas as pd
from pathlib import Path
import os

def update_paths():
    # Base paths requested by user (Relative to Repo Root)
    # We will simply prepend these strings to the filenames
    
    # 1. Update Test
    test_parquet = Path("data/kie_datasets/baseline_test_pseudo_labels.parquet")
    if test_parquet.exists():
        print(f"Updating {test_parquet}...")
        df = pd.read_parquet(test_parquet)
        # Target format: data/datasets/images/test/{filename}
        base_test = "data/datasets/images/test"
        df['image_path'] = df['image_path'].apply(lambda x: f"{base_test}/{Path(x).name}")
        df.to_parquet(test_parquet)
        print("Done.")
    else:
        print(f"Skipping {test_parquet} (Not found)")

    # 2. Update Val
    val_parquet = Path("data/kie_datasets/baseline_val_pseudo_labels.parquet")
    if val_parquet.exists():
        print(f"Updating {val_parquet}...")
        df = pd.read_parquet(val_parquet)
        # Target format: data/datasets/images_val_canonical/{filename}
        base_val = "data/datasets/images_val_canonical"
        df['image_path'] = df['image_path'].apply(lambda x: f"{base_val}/{Path(x).name}")
        df.to_parquet(val_parquet)
        print("Done.")
    else:
        print(f"Skipping {val_parquet} (Not found)")

if __name__ == "__main__":
    update_paths()
