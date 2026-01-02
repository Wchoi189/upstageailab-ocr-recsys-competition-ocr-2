
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing
from functools import partial
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import orientation utilities
try:
    from ocr.utils.orientation import normalize_pil_image
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import ocr.utils.orientation, EXIF normalization will be skipped")
    normalize_pil_image = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resize_and_save(row, output_root, max_dim=1024, image_dir=None):
    """
    Resizes image and saves to output_root. Returns new path.
    """
    try:
        # Extract path
        orig_path = Path(row["image_path"])

        # Resolve full path
        full_path = orig_path
        if image_dir and not orig_path.is_absolute():
             full_path = Path(image_dir) / orig_path

        if not full_path.exists():
             # Try resolving relative to CWD if implied
             if (Path.cwd() / full_path).exists():
                 full_path = Path.cwd() / full_path
             elif (Path.cwd() / "data/raw/external/aihub_public_admin_doc/validation/images" / orig_path).exists():
                 # Fallback hardcoded for safety? No, rely on args.
                 pass
             else:
                 return None # Fail

        orig_path = full_path

        # Define new path
        # Structure: output_root / filename
        # To avoid collisions, maybe allow subdirs?
        # Using flat structure with unique hash or keeping original structure is best.
        # Let's keep original filename but put in output_root

        new_filename = orig_path.name
        # Check for collisions? The basenames might not be unique across datasets.
        # Use parent name as prefix
        new_filename = f"{orig_path.parent.name}_{orig_path.name}"

        output_path = output_root / new_filename

        if output_path.exists():
            return str(output_path)

        try:
            with Image.open(orig_path) as img:
                # Convert to RGB first
                img = img.convert("RGB")

                # Apply EXIF orientation normalization to ensure canonical orientation
                if normalize_pil_image:
                    img, _ = normalize_pil_image(img)
                    # After normalization, EXIF orientation should be effectively 1
                    # We'll strip EXIF data when saving to ensure clean output

                # Resize if needed
                w, h = img.size
                if w > max_dim or h > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

                # Save with EXIF data stripped to ensure canonical orientation
                # This prevents EXIF rotation tags from being preserved
                img.save(
                    output_path,
                    "JPEG",
                    quality=85,
                    exif=b"",  # Strip EXIF data to ensure no rotation tags remain
                )

            return str(output_path)
        except Exception as e:
            logger.error(f"Error processing {orig_path}: {e}")
            return None

    except Exception as e:
        return None

def process_dataset(parquet_path, output_dir, max_dim, num_workers=8, image_dir=None):
    logger.info(f"Processing {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Identify image folder name for this dataset to organize output
    dataset_name = Path(parquet_path).stem
    dataset_out_dir = output_root / dataset_name
    dataset_out_dir.mkdir(exist_ok=True)

    # Prepare task
    func = partial(resize_and_save, output_root=dataset_out_dir, max_dim=max_dim, image_dir=image_dir)

    # Convert df to records
    rows = df.to_dict('records')

    new_paths = []
    with multiprocessing.Pool(num_workers) as pool:
        for result in tqdm(pool.imap(func, rows), total=len(rows)):
            new_paths.append(result)

    # Update DF
    # Filter out failures
    success_mask = [p is not None for p in new_paths]
    df_filtered = df[success_mask].copy()
    df_filtered["image_path"] = [p for p in new_paths if p is not None]

    # Make paths relative to project root? Or absolute?
    # Absolute is safer for now.

    # Save new parquet
    new_parquet_path = Path(parquet_path).parent / f"{Path(parquet_path).stem}_optimized.parquet"
    df_filtered.to_parquet(new_parquet_path)
    logger.info(f"Saved optimized dataset to {new_parquet_path} (Dropped {len(df) - len(df_filtered)} broken images)")

def main():
    parser = argparse.ArgumentParser(description="Optimize images for training")
    parser.add_argument("--files", nargs="+", required=True, help="Parquet files to process")
    parser.add_argument("--image_dirs", nargs="+", help="Root directories for images (one per file). Use 'None' for CWD.")
    parser.add_argument("--output_dir", default="data/optimized_images", help="Where to save resized images")
    parser.add_argument("--max_dim", type=int, default=1024, help="Max image dimension")
    parser.add_argument("--workers", type=int, default=8, help="Num workers")

    args = parser.parse_args()

    image_dirs = args.image_dirs
    if not image_dirs:
        image_dirs = [None] * len(args.files)

    if len(image_dirs) != len(args.files):
        print("Error: Number of image directories must match number of files")
        return

    for f, img_dir in zip(args.files, image_dirs):
        real_img_dir = img_dir if img_dir and img_dir != "None" else None
        process_dataset(f, args.output_dir, args.max_dim, args.workers, image_dir=real_img_dir)

if __name__ == "__main__":
    main()
