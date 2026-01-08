#!/usr/bin/env python3
"""Process KIE datasets locally using Prebuilt Extraction API (no AWS/S3).

This script processes KIE datasets using local image paths only.
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add aws-batch-processor root to path
AWS_BATCH_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(AWS_BATCH_ROOT))

from src.batch_processor.core import ResumableBatchProcessor
import os


async def process_kie_dataset_local(
    dataset_name: str,
    input_file: Path,
    batch_size: int = 500,
    concurrency: int = 2,
    resume: bool = False,
    api_key_env: str = "UPSTAGE_API_KEY",
):
    """Process KIE dataset locally (no S3)."""

    print(f"\n{'='*80}")
    print(f"Processing KIE Dataset: {dataset_name}")
    print(f"Mode: LOCAL (no AWS/S3)")
    print(f"API: Prebuilt Extraction (for KIE)")
    print(f"{'='*80}\n")

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return False

    # Get API key
    api_key = os.getenv(api_key_env)
    if not api_key:
        # Try .env.local from project root
        env_local = Path(__file__).parent.parent.parent.parent / ".env.local"
        if env_local.exists():
            with open(env_local) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{api_key_env}="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if not api_key:
        print(f"ERROR: {api_key_env} not found in environment or .env.local")
        return False

    print(f"✓ API key found")

    # NO S3 client - local processing only
    s3_client = None
    print(f"✓ Local processing mode (no S3)")
    print()

    # Create processor with Prebuilt Extraction
    checkpoint_dir = Path(f"data/checkpoints/{dataset_name}_kie_prebuilt")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    processor = ResumableBatchProcessor(
        api_key=api_key,
        api_type="prebuilt-extraction",  # KIE uses prebuilt-extraction
        concurrency=concurrency,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        s3_client=s3_client  # None for local-only
    )

    # Output file
    output_file = Path(f"data/output/{dataset_name}_kie_pseudo_labels.parquet")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Concurrency: {concurrency}")
    print()

    # Process dataset
    try:
        await processor.process_dataset(
            parquet_file=input_file,
            output_file=output_file,
            dataset_name=dataset_name,
            resume=resume,
        )

        print()
        print("=" * 80)
        print(f"✓ Successfully processed {dataset_name}")
        print("=" * 80)
        print(f"Output: {output_file}")
        print()
        print("Next step: Run export script to remove polygons and create final KIE dataset")

        return True

    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ Error processing {dataset_name}")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="Process KIE datasets locally with Prebuilt Extraction API")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["baseline_val", "baseline_train", "baseline_test"],
        help="Dataset to process"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Input parquet file path (default: data/input/{dataset}.parquet)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for checkpointing (default: 500)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Concurrency level (default: 2, lower for rate limits)"
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="UPSTAGE_API_KEY",
        help="Environment variable name for API key"
    )

    args = parser.parse_args()

    # Determine input file
    if args.input_file:
        input_file = Path(args.input_file)
    else:
        # Default: look in data/input
        input_file = Path(f"data/input/{args.dataset}.parquet")
        # If not found, try to create from export
        if not input_file.exists():
            export_file = Path(f"../../data/export/baseline_dp/{args.dataset.split('_')[-1]}.parquet")
            if export_file.exists():
                print(f"Input file not found, will use export file: {export_file}")
                input_file = export_file

    success = await process_kie_dataset_local(
        dataset_name=args.dataset,
        input_file=input_file,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        resume=args.resume,
        api_key_env=args.api_key_env,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
