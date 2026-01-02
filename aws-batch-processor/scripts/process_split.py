#!/usr/bin/env python3
"""Process a dataset split with Prebuilt Extraction API."""
import sys
import asyncio
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.batch_processor.core import ResumableBatchProcessor
import os

async def process_split(
    dataset_name: str,
    part_num: int,
    api_key_env: str = "UPSTAGE_API_KEY",
    concurrency: int = 1,
    resume: bool = False,
):
    """Process a dataset split."""
    split_name = f"{dataset_name}_part{part_num}"
    input_file = Path(f"data/input/splits/{split_name}.parquet")
    output_file = Path(f"data/output/splits/{split_name}_pseudo_labels.parquet")
    checkpoint_dir = Path(f"data/checkpoints/{split_name}_prebuilt")

    print(f"\n{'='*80}")
    print(f"Processing {split_name}")
    print(f"API Key: {api_key_env}")
    print(f"Concurrency: {concurrency}")
    print(f"Resume: {resume}")
    print(f"{'='*80}\n")

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return False

    # Get API key
    api_key = os.getenv(api_key_env)
    if not api_key:
        env_local = Path(".env.local")
        if env_local.exists():
            with open(env_local) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{api_key_env}="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if not api_key:
        print(f"ERROR: {api_key_env} not found")
        return False

    # Create output directories
    output_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create processor
    processor = ResumableBatchProcessor(
        api_key=api_key,
        api_type="prebuilt-extraction",
        concurrency=concurrency,
        batch_size=500,
        checkpoint_dir=checkpoint_dir,
        s3_client=None,  # Local processing, no S3
    )

    # Process
    try:
        await processor.process_dataset(
            parquet_file=input_file,
            output_file=output_file,
            dataset_name=split_name,
            resume=resume,
        )
        print(f"\n✓ Successfully processed {split_name}")
        return True
    except Exception as e:
        print(f"\n✗ Error processing {split_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    parser = argparse.ArgumentParser(description="Process a dataset split")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., baseline_val_canonical)")
    parser.add_argument("--part", type=int, required=True, choices=[1, 2, 3], help="Part number (1, 2, or 3)")
    parser.add_argument("--api-key-env", type=str, default="UPSTAGE_API_KEY", help="API key environment variable")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrency level")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    success = await process_split(
        args.dataset,
        args.part,
        args.api_key_env,
        args.concurrency,
        args.resume,
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())
