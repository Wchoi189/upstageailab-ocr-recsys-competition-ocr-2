import argparse
import logging
import multiprocessing
import os
from functools import partial
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_image(row, output_root, max_dim=1024, image_dir=None):
    """
    Process a single image:
    1. Resolve path
    2. Open and rotate based on EXIF
    3. Resize if needed
    4. Convert to RGB (standardize)
    5. Save as JPEG
    """
    try:
        # Extract path from row
        orig_path_str = row.get("image_path")
        if not orig_path_str:
            return None

        orig_path = Path(orig_path_str)

        # Resolve full path
        full_path = orig_path
        if image_dir:
            # If image_dir is provided, treat orig_path as relative to it
            # But check if orig_path is already absolute or relative to cwd
            if not orig_path.is_absolute():
                 candidate = Path(image_dir) / orig_path
                 if candidate.exists():
                     full_path = candidate

        if not full_path.exists():
             # Try resolving relative to CWD if the above failed or wasn't tried
             if (Path.cwd() / orig_path).exists():
                 full_path = Path.cwd() / orig_path
             # Fallback: Try finding filename in image_dir if provided
             elif image_dir:
                 candidate = Path(image_dir) / orig_path.name
                 if candidate.exists():
                     full_path = candidate
                 else:
                     return None
             else:
                 # Last ditch: check if it's already in the output dir (idempotency)
                 return None

        # Prepare output path
        # Maintain filename, save to output_root
        # structure: output_root / filename (flattened) or keep structure?
        # Flattening is safer for avoiding deep directory hell, but might have collisions.
        # The previous script flattened. Let's keep it simple: output_root / filename
        filename = full_path.name
        # Ensure unique name if collision? For now assume unique filenames or use partial path
        # To be safe, let's use the parent name as prefix if possible, or just filename if unique enough.
        # Competition data usually has unique IDs.

        output_path = Path(output_root) / filename

        # Optimization: If output exists, check size/validity?
        # For now, overwrite to ensure EXIF fix is applied.

        with Image.open(full_path) as img:
            # 1. EXIF Transpose (Fix Rotation)
            # This applies the rotation specified in EXIF tags to the pixel data
            img = ImageOps.exif_transpose(img)

            # 2. Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 3. Resize
            w, h = img.size
            if w > max_dim or h > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            # 4. Save
            # Compress slightly to save space, but keep high quality
            img.save(output_path, "JPEG", quality=85)

        return str(output_path)

    except Exception as e:
        logger.warning(f"Failed to process {row.get('image_path')}: {e}")
        return None

def process_dataset(parquet_path, output_dir, max_dim, num_workers, image_dir=None):
    logger.info(f"Processing dataset: {parquet_path}")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error(f"Could not read parquet {parquet_path}: {e}")
        return

    # Create output subdirectory for this dataset
    dataset_name = Path(parquet_path).stem
    dataset_out_dir = Path(output_dir) / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Images will be saved to: {dataset_out_dir}")

    # Prepare arguments for worker
    rows = df.to_dict('records')

    # Use partial to bind constant arguments
    worker_func = partial(process_single_image, output_root=dataset_out_dir, max_dim=max_dim, image_dir=image_dir)

    # Run in parallel
    new_paths = []
    with multiprocessing.Pool(num_workers) as pool:
        # Use tqdm for progress bar
        for result in tqdm(pool.imap(worker_func, rows), total=len(rows), desc=f"Optimizing {dataset_name}"):
            new_paths.append(result)

    # Update DataFrame
    # Filter out failures (None)
    success_count = sum(1 for p in new_paths if p is not None)
    fail_count = len(rows) - success_count

    if fail_count > 0:
        logger.warning(f"Failed to process {fail_count} images. They will be dropped from the dataset.")

    # We need to preserve the order to match the dataframe or just filter
    # Since we iterated row by row, `new_paths` corresponds index-wise to `df`

    # Add new column or replace? Replace `image_path` as we want the training to use the new ones
    df['image_path'] = new_paths

    # Drop rows where image_path is None
    df_clean = df.dropna(subset=['image_path'])

    # Save new parquet
    output_parquet_name = f"{dataset_name}_optimized_v2.parquet"
    output_parquet_path = Path(parquet_path).parent / output_parquet_name

    df_clean.to_parquet(output_parquet_path)
    logger.info(f"Saved optimized dataset to: {output_parquet_path}")
    logger.info(f"Original size: {len(df)}, Optimized size: {len(df_clean)}")

def main():
    parser = argparse.ArgumentParser(description="Optimize images (Resize & EXIF Rotate) and update Parquet paths")
    parser.add_argument("--files", nargs="+", required=True, help="List of parquet files to process")
    parser.add_argument("--image_dirs", nargs="+", help="Corresponding root directories for images. Use 'None' to skip/auto-detect.")
    parser.add_argument("--output_dir", default="data/optimized_images_v2", help="Root output directory")
    parser.add_argument("--max_dim", type=int, default=1024, help="Maximum image dimension (width or height)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes")

    args = parser.parse_args()

    # Handle image_dirs mapping
    image_dirs = args.image_dirs
    if not image_dirs:
        image_dirs = [None] * len(args.files)
    elif len(image_dirs) != len(args.files):
        logger.error("Error: Number of --image_dirs must match number of --files")
        return

    for parquet_file, img_dir in zip(args.files, image_dirs):
        # Handle string 'None' from CLI
        real_img_dir = img_dir if img_dir != 'None' else None

        process_dataset(
            parquet_path=parquet_file,
            output_dir=args.output_dir,
            max_dim=args.max_dim,
            num_workers=args.workers,
            image_dir=real_img_dir
        )

if __name__ == "__main__":
    main()
