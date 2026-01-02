#!/usr/bin/env python3
"""Monitor progress of dataset processing."""
import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

def monitor_progress(dataset_name: str):
    """Monitor processing progress."""
    print(f"\n{'='*80}")
    print(f"Progress Monitor: {dataset_name}")
    print(f"{'='*80}\n")

    # Check splits
    splits_dir = Path("data/input/splits")
    output_splits_dir = Path("data/output/splits")
    checkpoints_base = Path("data/checkpoints")

    total_input = 0
    total_processed = 0

    for part_num in [1, 2, 3]:
        split_name = f"{dataset_name}_part{part_num}"
        input_file = splits_dir / f"{split_name}.parquet"
        output_file = output_splits_dir / f"{split_name}_pseudo_labels.parquet"
        checkpoint_dir = checkpoints_base / f"{split_name}_prebuilt"

        # Get input count
        input_count = 0
        if input_file.exists():
            try:
                df = pd.read_parquet(input_file)
                input_count = len(df)
                total_input += input_count
            except:
                pass

        # Get processed count
        processed_count = 0
        if output_file.exists():
            try:
                df = pd.read_parquet(output_file)
                processed_count = len(df)
                total_processed += processed_count
            except:
                pass

        # Check checkpoint
        checkpoint_count = 0
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*.parquet"))
            if checkpoint_files:
                try:
                    # Get latest checkpoint
                    latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                    df = pd.read_parquet(latest)
                    checkpoint_count = len(df)
                except:
                    pass

        # Status
        if processed_count == input_count and input_count > 0:
            status = "✓ COMPLETE"
        elif checkpoint_count > 0:
            status = f"⏳ IN PROGRESS ({checkpoint_count}/{input_count})"
        else:
            status = "⏸️  NOT STARTED"

        print(f"Part {part_num}: {status}")
        print(f"  Input: {input_count} rows")
        print(f"  Processed: {processed_count} rows")
        if checkpoint_count > 0:
            print(f"  Checkpoint: {checkpoint_count} rows")
        print()

    # Overall progress
    if total_input > 0:
        progress_pct = (total_processed / total_input) * 100
        print(f"{'='*80}")
        print(f"Overall Progress: {total_processed}/{total_input} ({progress_pct:.1f}%)")
        print(f"{'='*80}\n")

    # Check if all complete
    if total_processed == total_input and total_input > 0:
        print("✓ All parts complete! Ready to merge.")
        return True
    else:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor processing progress")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")

    args = parser.parse_args()
    monitor_progress(args.dataset)
