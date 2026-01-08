#!/usr/bin/env python3
"""Reprocess datasets with Prebuilt Extraction API."""
import sys
import asyncio
import argparse
import boto3
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.batch_processor.core import ResumableBatchProcessor
import os

async def reprocess_dataset(dataset_name, batch_size=500, concurrency=3, resume=False, api_key_env="UPSTAGE_API_KEY"):
    """Reprocess a dataset with Prebuilt Extraction."""
    print(f"\n{'='*80}")
    print(f"Reprocessing {dataset_name} with Prebuilt Extraction API")
    print(f"Resume: {resume}")
    print(f"API Key Env: {api_key_env}")
    print(f"{'='*80}\n")
    
    input_file = Path(f"data/input/{dataset_name}.parquet")
    output_file = Path(f"data/output/{dataset_name}_pseudo_labels.parquet")
    
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
        
    # Initialize S3 client
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3')
        print("✓ S3 client initialized")
    except Exception as e:
        print(f"WARNING: Failed to initialize S3 client: {e}")
        print("Image downloading from S3 will likely fail.")
        s3_client = None
    
    # Create processor with Prebuilt Extraction
    checkpoint_dir = Path(f"data/checkpoints/{dataset_name}_prebuilt")
    processor = ResumableBatchProcessor(
        api_key=api_key,
        api_type="prebuilt-extraction",
        concurrency=concurrency,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        s3_client=s3_client
    )
    
    # Process dataset
    try:
        await processor.process_dataset(
            parquet_file=input_file,
            output_file=output_file,
            dataset_name=dataset_name,
            resume=resume, 
        )
        print(f"\n✓ Successfully reprocessed {dataset_name}")
        return True
    except Exception as e:
        print(f"\n✗ Error reprocessing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main reprocessing function."""
    parser = argparse.ArgumentParser(description="Reprocess datasets with Prebuilt Extraction API")
    parser.add_argument("--dataset", type=str, default="baseline_train", help="Dataset name to process (default: baseline_train)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrency level")
    parser.add_argument("--api-key-env", type=str, default="UPSTAGE_API_KEY", help="Environment variable name for API key")
    
    args = parser.parse_args()
    
    success = await reprocess_dataset(
        args.dataset, 
        batch_size=args.batch_size, 
        concurrency=args.concurrency,
        resume=args.resume,
        api_key_env=args.api_key_env
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
