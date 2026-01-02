import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifier")

def verify_parquet(parquet_path: str):
    path = Path(parquet_path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return

    logger.info(f"Loading {path}...")
    df = pd.read_parquet(path)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    required_cols = ["image_path", "width", "height", "texts", "polygons", "labels"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")

    # Check nulls
    logger.info("Checking for nulls...")
    print(df.isnull().sum())

    # Check first row
    row = df.iloc[0]
    logger.info("First row sample:")
    logger.info(f"Image Path: {row['image_path']}")
    logger.info(f"Text count: {len(row['texts'])}")
    logger.info(f"Polygon count: {len(row['polygons'])}")

    # Check file existence
    # Note: image_path might be relative or absolute.
    # The processing script saves it as relative to where?
    # It saved: str(rel_image_dir / file_name) which is like "02.원천데이터(Jpg)/Category/..."
    # Root is data/raw/external/aihub_public_admin_doc/validation/images

    root_images = Path("data/raw/external/aihub_public_admin_doc/validation/images")
    img_path = root_images / row['image_path']
    if img_path.exists():
        logger.info(f"Verified image file exists: {img_path}")
    else:
        logger.warning(f"Image file NOT found: {img_path} (Check path mapping)")

    # Check polygon shape
    polys = row['polygons']
    if len(polys) > 0:
        p = polys[0]
        logger.info(f"Sample polygon: {p}, type: {type(p)}")
        if isinstance(p, (list, np.ndarray)) and len(p) == 4:
            logger.info("Polygon format appears correct [x1, y1, x2, y2]")
        else:
            logger.warning(f"Polygon format check failed or unexpected: {p}")

if __name__ == "__main__":
    verify_parquet("data/processed/aihub_validation.parquet")
