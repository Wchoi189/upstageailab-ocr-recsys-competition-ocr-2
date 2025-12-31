
import os
import sys
import time
import argparse
import boto3
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.batch_processor.schemas import OCRStorageItem

# API Configuration
API_URL = "https://api.upstage.ai/v1/information-extraction"

def process_image(row, api_key, s3_client=None):
    """Process a single image synchronously."""
    image_path_str = row['image_path']
    temp_file_path = None
    
    try:
        # Handle S3
        if image_path_str.startswith('s3://'):
            if s3_client:
                bucket, key = image_path_str[5:].split('/', 1)
                import uuid
                temp_file = f"temp_image_{uuid.uuid4()}.jpg"
                s3_client.download_file(bucket, key, temp_file)
                image_path = Path(temp_file)
                temp_file_path = image_path
            else:
                return None
        else:
            image_path = Path(image_path_str)
            
        if not image_path.exists():
            return None
            
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # Retry loop
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with open(image_path, "rb") as f:
                    response = requests.post(
                        API_URL,
                        headers=headers,
                        data={"model": "receipt-extraction"},
                        files={"document": f}
                    )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    return None
            except Exception as e:
                print(f"Request failed: {e}")
                time.sleep(1)
                
        return None
        
    finally:
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-key-env", default="UPSTAGE_API_KEY")
    parser.add_argument("--start-index", type=int, default=0, help="Start index for processing")
    parser.add_argument("--end-index", type=int, default=None, help="End index for processing")
    parser.add_argument("--checkpoint-suffix", type=str, default="", help="Suffix for checkpoint directory")
    args = parser.parse_args()
    
    # Setup
    input_file = Path(f"data/input/{args.dataset}.parquet")
    output_file = Path(f"data/output/{args.dataset}_pseudo_labels{args.checkpoint_suffix}.parquet")
    checkpoint_dir = Path(f"data/checkpoints/{args.dataset}_serial{args.checkpoint_suffix}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # API Key
    api_key = os.getenv(args.api_key_env) or os.getenv("UPSTAGE_API_KEY") # Fallback
    # (Checking .env.local logic omitted for brevity, verify manually if needed)
    
    if not api_key:
        # Quick .env.local check
        if Path(".env.local").exists():
             with open(".env.local") as f:
                for line in f:
                    if line.startswith(f"{args.api_key_env}="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    
    if not api_key:
        print("No API Key found")
        sys.exit(1)
        
    s3 = boto3.client('s3')
    
    df = pd.read_parquet(input_file)
    
    # Apply slicing for sharding
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else len(df)
    
    # Ensure start_idx is within bounds
    if start_idx >= len(df):
        print(f"Start index {start_idx} is out of bounds for dataset of size {len(df)}")
        sys.exit(0)
        
    df = df.iloc[start_idx:end_idx]
    total = len(df)
    print(f"Processing range: {start_idx} to {end_idx} (Total: {total} images)")
    
    # Resume
    processed_count = 0
    all_results = []
    if args.resume and checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("*.parquet"))
        if checkpoints:
            for cp in checkpoints:
                cdf = pd.read_parquet(cp)
                all_results.extend(cdf.to_dict('records'))
            processed_count = len(all_results)
            print(f"Resuming from {processed_count} (relative to shard)")
            
    # Loop
    results_buffer = []
    checkpoint_interval = 10
    
    # Adjust iteration to account for already processed items in this shard
    iter_df = df.iloc[processed_count:]
    
    for idx, row in tqdm(iter_df.iterrows(), total=total-processed_count):
        row_idx = row.name if hasattr(row, 'name') else idx
        api_result = process_image(row, api_key, s3)
        
        if api_result:
            # Parse result
            texts = []
            labels = []
            polygons = [] 
            
            def parse_field(f_item):
                f_val = f_item.get('refinedValue', f_item.get('value'))
                f_key = f_item.get('key')
                f_type = f_item.get('type')
                
                if f_val:
                    texts.append(f_val)
                    if f_key:
                        labels.append(f_key.split('.')[-1])
                    else:
                        labels.append(f_type if f_type else 'text')
                    
                    # Handle Polygons
                    poly_raw = f_item.get('boundingPoly', [])
                    if not poly_raw:
                        bboxes = f_item.get('boundingBoxes', [])
                        if bboxes and len(bboxes) > 0:
                            vertices = bboxes[0].get('vertices', [])
                            poly_raw = [[v.get('x',0), v.get('y',0)] for v in vertices]
                            
                    polygons.append(poly_raw if poly_raw else [])

                props = f_item.get('properties', [])
                for prop in props:
                    parse_field(prop)

            fields = api_result.get('fields', [])
            for field in fields:
                parse_field(field)

            # Handle groups/properties if needed (simple recursion for properties)
            # (Skipping deep recursion for basic receive fields which are usually top-level or 1-deep)
            
            # Create Schema Item
            # Handle S3 paths for ID generation
            image_path_str = row['image_path']
            stem = Path(image_path_str).stem
            
            item = OCRStorageItem(
                id=f"{args.dataset}_pseudo_{stem}",
                split="pseudo",
                image_path=image_path_str,
                image_filename=Path(image_path_str).name,
                width=int(row.get('width', 0)),
                height=int(row.get('height', 0)),
                polygons=polygons,
                texts=texts,
                labels=labels,
                metadata={
                    "source": "upstage_api_prebuilt_extraction_serial",
                    "api_type": "prebuilt-extraction"
                }
            )
            results_buffer.append(item.model_dump())
        
        # Checkpoint
        if len(results_buffer) >= checkpoint_interval:
            cp_idx = (len(all_results) + len(results_buffer)) // checkpoint_interval
            cp_path = checkpoint_dir / f"batch_{cp_idx:05d}.parquet"
            pd.DataFrame(results_buffer).to_parquet(cp_path)
            all_results.extend(results_buffer)
            results_buffer = [] # Clear buffer
            
    # Final save
    if results_buffer:
        cp_path = checkpoint_dir / f"batch_final.parquet"
        pd.DataFrame(results_buffer).to_parquet(cp_path)
        all_results.extend(results_buffer)
        
    # Combine all
    final_df = pd.DataFrame(all_results)
    final_df.to_parquet(output_file)
    print(f"Done! Saved to {output_file}")
if __name__ == "__main__":
    main()
