#!/usr/bin/env python3
"""Merge processed splits back into single dataset."""
import sys
import argparse
import pandas as pd
from pathlib import Path

def merge_splits(dataset_name: str):
    """Merge 3 processed splits into single output."""
    print(f"\n{'='*80}")
    print(f"Merging splits for {dataset_name}")
    print(f"{'='*80}\n")

    splits_dir = Path("data/output/splits")
    output_file = Path(f"data/output/{dataset_name}_pseudo_labels.parquet")

    parts = []
    for part_num in [1, 2, 3]:
        split_name = f"{dataset_name}_part{part_num}"
        split_file = splits_dir / f"{split_name}_pseudo_labels.parquet"

        if not split_file.exists():
            print(f"ERROR: Split {part_num} not found: {split_file}")
            return False

        df = pd.read_parquet(split_file)
        parts.append(df)
        print(f"  ✓ Loaded part {part_num}: {len(df)} rows")

    # Merge
    merged_df = pd.concat(parts, ignore_index=True)
    print(f"\n  Total rows: {len(merged_df)}")

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_file, index=False)
    print(f"  ✓ Saved merged dataset: {output_file}")
    print()

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge processed splits")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")

    args = parser.parse_args()

    success = merge_splits(args.dataset)
    sys.exit(0 if success else 1)
