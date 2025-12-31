import pandas as pd
from pathlib import Path
import sys

def merge_datasets():
    print("Starting merge process...")
    
    # 1. Load initial serial checkpoints (Account #1)
    checkpoint_dir = Path("data/checkpoints/baseline_train_serial")
    checkpoints = sorted(checkpoint_dir.glob("*.parquet"))
    print(f"Found {len(checkpoints)} checkpoint files in {checkpoint_dir}")
    
    initial_dfs = []
    for cp in checkpoints:
        initial_dfs.append(pd.read_parquet(cp))
        
    if initial_dfs:
        df_initial = pd.concat(initial_dfs, ignore_index=True)
        print(f"Initial run (Account #1): {len(df_initial)} rows")
    else:
        print("WARNING: No checkpoints found for initial run!")
        df_initial = pd.DataFrame()

    # 2. Load Part 1 (Account #2)
    part1_path = Path("data/output/baseline_train_pseudo_labels_acc2_part1.parquet")
    if part1_path.exists():
        df_part1 = pd.read_parquet(part1_path)
        print(f"Part 1 (Account #2): {len(df_part1)} rows")
    else:
        print(f"ERROR: Part 1 file not found at {part1_path}")
        return

    # 3. Load Part 2 (Account #2)
    part2_path = Path("data/output/baseline_train_pseudo_labels_acc2_part2.parquet")
    if part2_path.exists():
        df_part2 = pd.read_parquet(part2_path)
        print(f"Part 2 (Account #2): {len(df_part2)} rows")
    else:
        print(f"ERROR: Part 2 file not found at {part2_path}")
        return

    # 3.5 Load Missing Shard (Account #2 - Cleanup)
    missing_path = Path("data/output/baseline_train_pseudo_labels_missing.parquet")
    if missing_path.exists():
        df_missing = pd.read_parquet(missing_path)
        print(f"Missing Shard (Account #2): {len(df_missing)} rows")
    else:
        print(f"WARNING: Missing shard file not found at {missing_path}")
        df_missing = pd.DataFrame()

    # 4. Concatenate all
    final_df = pd.concat([df_initial, df_part1, df_part2, df_missing], ignore_index=True)
    total_rows = len(final_df)
    print(f"Total merged rows: {total_rows}")
    
    # 5. Deduplicate (just in case of overlaps)
    # Using 'image_path' as unique key if available, otherwise just use all columns
    if 'image_path' in final_df.columns:
        final_df = final_df.drop_duplicates(subset=['image_path'])
        print(f"After deduplication: {len(final_df)} rows")
    else:
        print("Warning: 'image_path' not found, skipping specific deduplication.")

    # 6. Verify count
    EXPECTED_COUNT = 3272
    if len(final_df) == EXPECTED_COUNT:
        print("SUCCESS: Exact match with expected count!")
    else:
        print(f"WARNING: Count mismatch! Expected {EXPECTED_COUNT}, got {len(final_df)}")
        if len(final_df) < EXPECTED_COUNT:
             print(f"Missing {EXPECTED_COUNT - len(final_df)} images.")
        else:
             print(f"Excess {len(final_df) - EXPECTED_COUNT} images.")

    # 7. Save
    output_path = Path("data/output/baseline_train_pseudo_labels.parquet")
    final_df.to_parquet(output_path)
    print(f"Saved merged dataset to {output_path}")

if __name__ == "__main__":
    merge_datasets()
