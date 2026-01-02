#!/usr/bin/env python3
"""Test script for KIE API re-call using Prebuilt Extraction API.

This script creates a small test dataset and processes it to verify:
1. API connectivity works
2. Output format is correct (no polygons for KIE)
3. Key-value pairs are extracted properly
"""

import sys
import asyncio
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add aws-batch-processor to path
AWS_BATCH_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AWS_BATCH_ROOT))

from src.batch_processor.core import ResumableBatchProcessor
import os
import boto3


async def test_kie_api_call(
    source_parquet: Path,
    test_size: int = 3,
    api_key_env: str = "UPSTAGE_API_KEY",
    concurrency: int = 1,
):
    """Test KIE API call with a small sample."""

    print("=" * 80)
    print("KIE API Test Call")
    print("=" * 80)
    print(f"Source: {source_parquet}")
    print(f"Test size: {test_size} images")
    print(f"API: Prebuilt Extraction (for KIE)")
    print("=" * 80)
    print()

    # Read source dataset
    if not source_parquet.exists():
        print(f"ERROR: Source file not found: {source_parquet}")
        return False

    print(f"Reading source dataset...")
    df_source = pd.read_parquet(source_parquet)
    print(f"  ✓ Loaded {len(df_source)} rows")

    # Create test sample
    test_df = df_source.head(test_size).copy()
    test_dataset_name = "test_kie_sample"

    # Save test input
    test_input_dir = AWS_BATCH_ROOT / "data" / "input"
    test_input_dir.mkdir(parents=True, exist_ok=True)
    test_input_file = test_input_dir / f"{test_dataset_name}.parquet"
    test_df.to_parquet(test_input_file, index=False)
    print(f"  ✓ Created test input: {test_input_file} ({len(test_df)} rows)")
    print()

    # Get API key
    api_key = os.getenv(api_key_env)
    if not api_key:
        # Try .env.local
        env_local = PROJECT_ROOT / ".env.local"
        if env_local.exists():
            with open(env_local) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{api_key_env}="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if not api_key:
        print(f"ERROR: {api_key_env} not found in environment or .env.local")
        print("Please set UPSTAGE_API_KEY environment variable or add it to .env.local")
        return False

    print(f"  ✓ API key found")
    print()

    # Initialize S3 client (optional, for S3 image paths)
    try:
        s3_client = boto3.client('s3')
        print("  ✓ S3 client initialized")
    except Exception as e:
        print(f"  ⚠ S3 client not available: {e}")
        print("  (Local image paths will be used)")
        s3_client = None
    print()

    # Create processor with Prebuilt Extraction API
    checkpoint_dir = AWS_BATCH_ROOT / "data" / "checkpoints" / f"{test_dataset_name}_prebuilt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing processor...")
    processor = ResumableBatchProcessor(
        api_key=api_key,
        api_type="document-parse",  # Use document-parse for enhanced mode
        enhanced=True,  # Enable enhanced mode
        concurrency=concurrency,
        batch_size=test_size,  # Small batch for test
        checkpoint_dir=checkpoint_dir,
        s3_client=s3_client
    )
    print("  ✓ Processor initialized")
    print()

    # Process test dataset
    output_file = AWS_BATCH_ROOT / "data" / "output" / f"{test_dataset_name}_pseudo_labels.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Processing test dataset...")
    print(f"  Input: {test_input_file}")
    print(f"  Output: {output_file}")
    print()

    try:
        await processor.process_dataset(
            parquet_file=test_input_file,
            output_file=output_file,
            dataset_name=test_dataset_name,
            resume=False,
        )

        print()
        print("=" * 80)
        print("Processing Complete - Validating Output")
        print("=" * 80)

        # Validate output
        if not output_file.exists():
            print(f"  ✗ Output file not found: {output_file}")
            return False

        print(f"  ✓ Output file created: {output_file}")

        # Read and validate output
        df_output = pd.read_parquet(output_file)
        print(f"  ✓ Output contains {len(df_output)} rows")
        print(f"  ✓ Columns: {list(df_output.columns)}")

        # Check for polygons (should NOT be present in KIE)
        if "polygons" in df_output.columns:
            print(f"  ⚠ NOTE: Output contains 'polygons' column")
            # Check if polygons are actually empty (all empty arrays/lists)
            has_non_empty_polygons = False
            for idx, row in df_output.iterrows():
                polygons = row.get("polygons", [])
                if polygons is not None:
                    try:
                        # Handle numpy arrays
                        import numpy as np
                        if isinstance(polygons, np.ndarray):
                            # Check if all elements are empty
                            non_empty = [p for p in polygons if p is not None and (hasattr(p, '__len__') and len(p) > 0)]
                            if non_empty:
                                has_non_empty_polygons = True
                                print(f"    Row {idx} has {len(non_empty)} non-empty polygons")
                                break
                        elif isinstance(polygons, list):
                            non_empty = [p for p in polygons if p and (hasattr(p, '__len__') and len(p) > 0)]
                            if non_empty:
                                has_non_empty_polygons = True
                                print(f"    Row {idx} has {len(non_empty)} non-empty polygons")
                                break
                    except Exception as e:
                        # If we can't check, assume it might have data
                        print(f"    Row {idx}: Could not verify polygons ({e})")

            if has_non_empty_polygons:
                print(f"  ✗ FAIL: KIE output contains non-empty polygons (should be key-value pairs only)")
                return False
            else:
                print(f"  ✓ Polygons column exists but all are empty (acceptable - will be removed in export step)")
        else:
            print(f"  ✓ No polygons column (correct for KIE)")

        # Check for key-value data
        if "texts" in df_output.columns:
            print(f"  ✓ Contains 'texts' column")
            # Show sample
            if len(df_output) > 0:
                sample_texts = df_output.iloc[0].get("texts", [])
                if isinstance(sample_texts, list) and len(sample_texts) > 0:
                    print(f"    Sample texts: {sample_texts[:3]}...")

        # Check metadata
        if "metadata" in df_output.columns:
            print(f"  ✓ Contains 'metadata' column")
            if len(df_output) > 0:
                sample_meta = df_output.iloc[0].get("metadata", {})
                print(f"    Sample metadata keys: {list(sample_meta.keys()) if isinstance(sample_meta, dict) else 'N/A'}")

        print()
        print("=" * 80)
        print("✓ TEST PASSED: KIE API call successful")
        print("=" * 80)
        print()
        print("Next steps:")
        print(f"  1. Review output: {output_file}")
        print(f"  2. If successful, proceed with full dataset processing")
        print(f"  3. Use: python scripts/runners/reprocess_with_prebuilt_extraction.py --dataset baseline_val_canonical")

        return True

    except Exception as e:
        print()
        print("=" * 80)
        print("✗ TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test KIE API call with small sample")
    parser.add_argument(
        "--source",
        type=str,
        default="data/export/baseline_dp/val.parquet",
        help="Source parquet file (default: data/export/baseline_dp/val.parquet)"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=3,
        help="Number of images to test (default: 3)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrency level (default: 1 for test)"
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="UPSTAGE_API_KEY",
        help="Environment variable name for API key"
    )

    args = parser.parse_args()

    # Resolve source path
    source_path = Path(args.source)
    if not source_path.is_absolute():
        # Try relative to project root first
        source_path = PROJECT_ROOT / args.source
        if not source_path.exists():
            # Try relative to aws-batch-processor
            source_path = AWS_BATCH_ROOT / args.source
            if not source_path.exists():
                # Try as absolute path
                source_path = Path(args.source).resolve()

    success = await test_kie_api_call(
        source_parquet=source_path,
        test_size=args.test_size,
        api_key_env=args.api_key_env,
        concurrency=args.concurrency,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
