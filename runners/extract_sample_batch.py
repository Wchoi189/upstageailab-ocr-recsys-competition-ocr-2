#!/usr/bin/env python3
"""Extract diverse sample batch from baseline dataset for validation testing.

This script samples images from the baseline training dataset using stratified
sampling to ensure diverse representation across ID ranges and image characteristics.

Usage:
    uv run python runners/extract_sample_batch.py --output data/samples/validation_batch_50 --count 50
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def extract_stratified_sample(parquet_file: Path, output_dir: Path, sample_count: int = 50):
    """Extract stratified sample from baseline dataset.

    Args:
        parquet_file: Path to baseline parquet file
        output_dir: Output directory for sample images
        sample_count: Number of images to sample
    """
    # Load baseline data
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} records from {parquet_file}")

    # Stratified sampling across ID ranges
    # Split dataset into quantiles for diverse sampling
    df['id_numeric'] = df['id'].str.extract(r'(\d+)').astype(int)
    df['quantile'] = pd.qcut(df['id_numeric'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    # Sample proportionally from each quantile
    samples_per_quantile = sample_count // 5
    sample_df = df.groupby('quantile', group_keys=False).apply(
        lambda x: x.sample(n=min(samples_per_quantile, len(x)), random_state=42)
    )

    # If we need more samples, add random ones
    if len(sample_df) < sample_count:
        remaining = sample_count - len(sample_df)
        additional = df[~df.index.isin(sample_df.index)].sample(n=remaining, random_state=42)
        sample_df = pd.concat([sample_df, additional])

    print(f"\nSampled {len(sample_df)} images:")
    print(sample_df.groupby('quantile').size())

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy sampled images
    copied = 0
    for _, row in sample_df.iterrows():
        src_path = Path(row['image_path'])
        if src_path.exists():
            dst_path = output_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            print(f"Warning: Image not found: {src_path}")

    # Save sample manifest
    manifest_path = output_dir / "sample_manifest.csv"
    sample_df[['id', 'image_filename', 'width', 'height', 'quantile']].to_csv(
        manifest_path, index=False
    )

    print(f"\n✓ Copied {copied} images to {output_dir}")
    print(f"✓ Saved manifest to {manifest_path}")

    return sample_df


def main():
    parser = argparse.ArgumentParser(description="Extract sample batch from baseline dataset")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/processed/baseline_train.parquet"),
        help="Path to baseline parquet file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/samples/validation_batch_50"),
        help="Output directory for sample images"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of images to sample"
    )

    args = parser.parse_args()

    extract_stratified_sample(args.parquet, args.output, args.count)


if __name__ == "__main__":
    main()
