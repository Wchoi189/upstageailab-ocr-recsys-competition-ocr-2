
import pandas as pd
from pathlib import Path
import math

def migrate_test_checkpoint():
    output_file = Path("data/output/baseline_test_pseudo_labels.parquet")
    checkpoint_dir = Path("data/checkpoints/baseline_test_serial")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if not output_file.exists():
        print("No output file found, nothing to migrate.")
        return

    df = pd.read_parquet(output_file)
    print(f"Loaded {len(df)} records from {output_file}")
    
    # Save as batch_00000.parquet?
    # Serial script logic: cp_idx = (len(all) + len(buff)) // 10
    # If we have 412 items, we should save them in chunks of 10 or one big chunk?
    # Serial script loads ALL parquet files in dir.
    # So we can just save one big file: batch_init_migration.parquet
    
    checkpoint_path = checkpoint_dir / "batch_init_migration.parquet"
    df.to_parquet(checkpoint_path)
    print(f"Saved {len(df)} records to {checkpoint_path}")
    print("Serial script will now detect these and resume from 412.")

if __name__ == "__main__":
    migrate_test_checkpoint()
