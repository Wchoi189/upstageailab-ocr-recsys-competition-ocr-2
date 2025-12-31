import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import boto3
from src.batch_processor.core import ResumableBatchProcessor
from src.batch_processor.processor import S3ResumableBatchProcessor
from src.batch_processor.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_s3_client():
    try:
        return boto3.client('s3')
    except Exception as e:
        logger.warning(f"Failed to init S3 client: {e}")
        return None

async def main():
    parser = argparse.ArgumentParser(description="Unified Batch Processor for Upstage API")
    
    # Dataset / Input args
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., baseline_train)")
    parser.add_argument("--input-file", help="Explicit input parquet file")
    
    # Processing args
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--workers", type=int, default=cfg["processing"]["default_concurrency"], help="Concurrency (workers)")
    parser.add_argument("--batch-size", type=int, default=cfg["processing"]["default_batch_size"], help="Batch size")
    
    # API args
    parser.add_argument("--api-key-env", default="UPSTAGE_API_KEY", help="Env var for API Key")
    parser.add_argument("--api-type", default="document-parse", choices=["document-parse", "prebuilt-extraction"], help="API type")
    parser.add_argument("--enhanced", action="store_true", help="Use Enhanced Mode (document-parse only)")
    
    # Environment args
    parser.add_argument("--local-only", action="store_true", help="Disable S3 features")
    parser.add_argument("--s3-bucket", help="S3 bucket for checkpoints/output (optional)")
    parser.add_argument("--checkpoint-dir", help="Explicit checkpoint directory (local)")
    parser.add_argument("--output-dir", help="Explicit output directory (local)")
    
    args = parser.parse_args()
    
    # 1. Setup API Key
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        # Try loading from .env.local if existing
        local_env = Path(".env.local")
        if not local_env.exists():
            local_env = Path("../.env.local")
        
        if local_env.exists():
            with open(local_env) as f:
                for line in f:
                    if line.startswith(f"{args.api_key_env}="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    
    if not api_key:
        logger.error(f"API Key not found in env var {args.api_key_env}")
        sys.exit(1)
        
    # 2. Setup S3
    s3_client = None
    if not args.local_only:
        s3_client = get_s3_client()
        
    # 3. Determine Paths
    dataset_name = args.dataset
    
    # Input
    if args.input_file:
        input_path = Path(args.input_file)
    else:
        # Default path convention
        input_path = Path(cfg["paths"]["input"]) / f"{dataset_name}.parquet"
        
    if not input_path.exists():
        # Check if we need to download from S3 (if input_file arg was actually an S3 key? No, input-file is usually local path)
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Output
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_doc_parse.parquet"
    
    # Checkpoints
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = Path(cfg["paths"]["checkpoints"]) / f"{dataset_name}_checkpoints"
    
    # 4. Instantiate Processor
    # Use S3 processor if bucket is specified, otherwise standard
    if args.s3_bucket and s3_client:
        logger.info(f"Using S3ResumableBatchProcessor with bucket: {args.s3_bucket}")
        processor = S3ResumableBatchProcessor(
            api_key=api_key,
            s3_bucket=args.s3_bucket,
            concurrency=args.workers,
            batch_size=args.batch_size,
            checkpoint_dir=checkpoint_dir,
            api_type=args.api_type,
            enhanced=args.enhanced,
            s3_client=s3_client # Passed explicitly though S3Processor creates one too
        )
    else:
        logger.info(f"Using ResumableBatchProcessor (Local Checkpoints)")
        processor = ResumableBatchProcessor(
            api_key=api_key,
            concurrency=args.workers,
            batch_size=args.batch_size,
            checkpoint_dir=checkpoint_dir,
            s3_client=s3_client, # Pass client for image downloading
            api_type=args.api_type,
            enhanced=args.enhanced
        )
        
    # 5. Run parameters
    logger.info(f"Starting processing for {dataset_name}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info(f"Mode: {args.api_type} (Enhanced: {args.enhanced})")
    
    await processor.process_dataset(
        parquet_file=input_path,
        output_file=output_path,
        dataset_name=dataset_name,
        resume=args.resume
    )

if __name__ == "__main__":
    asyncio.run(main())
