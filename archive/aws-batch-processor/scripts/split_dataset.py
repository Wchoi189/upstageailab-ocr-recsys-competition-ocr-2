#!/usr/bin/env python3
"""Split a dataset into 3 parts by index for parallel processing."""
import sys
import argparse
import pandas as pd
from pathlib import Path

def split_dataset(input_file: Path, output_dir: Path, dataset_name: str):
    """Split dataset into 3 parts."""
    print(f"Splitting {input_file} into 3 parts...")

    # Read dataset
    df = pd.read_parquet(input_file)
    total_rows = len(df)
    print(f"  Total rows: {total_rows}")

    # Calculate split points
    part_size = total_rows // 3
    remainder = total_rows % 3

    splits = []
    start = 0
    for i in range(3):
        # Distribute remainder across first parts
        size = part_size + (1 if i < remainder else 0)
        end = start + size
        splits.append((start, end))
        start = end

    print(f"  Part 1: rows {splits[0][0]}-{splits[0][1]-1} ({splits[0][1]-splits[0][0]} rows)")
    print(f"  Part 2: rows {splits[1][0]}-{splits[1][1]-1} ({splits[1][1]-splits[1][0]} rows)")
    print(f"  Part 3: rows {splits[2][0]}-{splits[2][1]-1} ({splits[2][1]-splits[2][0]} rows)")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    for i, (start_idx, end_idx) in enumerate(splits, 1):
        part_df = df.iloc[start_idx:end_idx].copy()
        output_file = output_dir / f"{dataset_name}_part{i}.parquet"
        part_df.to_parquet(output_file, index=False)
        print(f"  ✓ Saved part {i}: {output_file} ({len(part_df)} rows)")

    print()
    print(f"✓ Split complete. Parts saved to: {output_dir}")
    return splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into 3 parts")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., baseline_val_canonical)")
    parser.add_argument("--input-dir", type=str, default="data/input", help="Input directory")
    parser.add_argument("--output-dir", type=str, default="data/input/splits", help="Output directory for splits")

    args = parser.parse_args()

    input_file = Path(args.input_dir) / f"{args.dataset}.parquet"
    output_dir = Path(args.output_dir)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    split_dataset(input_file, output_dir, args.dataset)
