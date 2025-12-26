#!/usr/bin/env python3
"""Generate pseudo-labels for unlabeled images using Upstage Document Parse API.

This script processes a directory of images (or a Parquet file) and uses the
Upstage API to extracting text and layout information. The results are saved
in the standardized OCRStorageItem Parquet format.
"""

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

import aiohttp
import cv2  # NEW
import numpy as np  # NEW
import pandas as pd
from PIL import Image
from rembg import remove
from tqdm.asyncio import tqdm

from ocr.data.schemas.storage import OCRStorageItem
from ocr.inference.preprocessing_pipeline import apply_optional_perspective_correction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_URL = "https://api.upstage.ai/v1/document-ai/ocr"
DEFAULT_RATE_LIMIT = 50  # Requests per minute


# --- Enhanced Processing Logic ---
def get_inverse_matrix(matrix: np.ndarray) -> np.ndarray:
    """Calculate inverse homography matrix."""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.eye(3)


def warp_points(points: list[list[float]], matrix: np.ndarray) -> list[list[float]]:
    """Apply homography transform to a list of points (N, 2)."""
    if not points:
        return []

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, matrix)
    return transformed.reshape(-1, 2).tolist()


def enhance_image(image_path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Apply Rembg + Perspective Correction.

    Returns:
        corrected_image: The enhanced image (BGR)
        inverse_matrix: Matrix to map corrected points back to raw space
        raw_size: (width, height) of original image
    """

    # Load Raw
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"Could not load {image_path}")

    raw_h, raw_w = original_image.shape[:2]

    # 1. Background Removal
    # Convert to RGB for rembg
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    output_rgba = remove(image_rgb)

    # Extract alpha and composite over white
    alpha = output_rgba[:, :, 3] / 255.0
    foreground_rgb = output_rgba[:, :, :3]
    bg_color = [255, 255, 255]
    composite = np.zeros_like(foreground_rgb)
    for c in range(3):
        composite[:, :, c] = alpha * foreground_rgb[:, :, c] + (1 - alpha) * bg_color[c]

    # Back to BGR for OpenCV functions
    image_rembg = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    # 2. Perspective Correction
    # We use the library function but need to capture the matrix
    # Note: apply_optional_perspective_correction returns (image, matrix) when return_matrix=True
    corrected_image, matrix = apply_optional_perspective_correction(image_rembg, enable_perspective_correction=True, return_matrix=True)

    if matrix is None:
        matrix = np.eye(3)

    inverse_matrix = get_inverse_matrix(matrix)
    return corrected_image, inverse_matrix, (raw_w, raw_h)


class UpstageClient:
    def __init__(self, api_key: str, rate_limit: int = DEFAULT_RATE_LIMIT):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.delay = 60.0 / rate_limit
        self.last_request_time = 0.0

    async def extract(self, session: aiohttp.ClientSession, image_bytes: bytes, filename: str) -> dict | None:
        """Call Upstage API with image bytes."""

        # Simple rate limiting
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.delay:
            await asyncio.sleep(self.delay - time_since_last)

        self.last_request_time = time.time()

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            data = aiohttp.FormData()
            data.add_field("document", image_bytes, filename=filename)

            async with session.post(API_URL, headers=headers, data=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API Error {response.status}: {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None


async def process_images(image_paths: list[Path], api_key: str, output_path: Path, dataset_name: str, enhance: bool = False):
    client = UpstageClient(api_key)
    results = []

    async with aiohttp.ClientSession() as session:
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                inverse_matrix = np.eye(3)
                raw_w, raw_h = 0, 0

                if enhance:
                    try:
                        # Apply enhancement pipeline
                        corrected_img, inverse_matrix, (raw_w, raw_h) = enhance_image(img_path)

                        # Encode for API
                        success, buf = cv2.imencode(".jpg", corrected_img)
                        if not success:
                            raise ValueError("Encoding failed")
                        image_bytes = buf.tobytes()

                    except Exception as e:
                        logger.warning(f"Enhancement failed for {img_path}, falling back to raw: {e}")
                        with open(img_path, "rb") as f:
                            image_bytes = f.read()
                        with Image.open(img_path) as im:
                            raw_w, raw_h = im.size
                else:
                    with open(img_path, "rb") as f:
                        image_bytes = f.read()
                    with Image.open(img_path) as im:
                        raw_w, raw_h = im.size

                # Call API
                api_result = await client.extract(session, image_bytes, img_path.name)

                if not api_result:
                    continue

                polygons = []
                texts = []
                labels = []

                pages = api_result.get("pages", [])
                for page in pages:
                    words = page.get("words", [])
                    for word in words:
                        bbox = word.get("boundingBox", {}).get("vertices", [])
                        # API returns [{x,y}, ...]
                        poly = [[float(v.get("x", 0)), float(v.get("y", 0))] for v in bbox]

                        if enhance:
                            # Map back to raw space!
                            poly = warp_points(poly, inverse_matrix)

                        polygons.append(poly)
                        texts.append(word.get("text", ""))
                        labels.append("text")

                item = OCRStorageItem(
                    id=f"{dataset_name}_pseudo_{img_path.stem}",
                    split="pseudo",
                    image_path=str(img_path),
                    image_filename=img_path.name,
                    width=raw_w,
                    height=raw_h,
                    polygons=polygons,
                    texts=texts,
                    labels=labels,
                    metadata={"source": "upstage_api", "enhanced": enhance, "inverse_matrix": inverse_matrix.tolist() if enhance else None},
                )
                item.validate_lengths()
                results.append(item.model_dump())

            except Exception as e:
                logger.error(f"Processing failed for {img_path}: {e}")

    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_parquet(output_path, engine="pyarrow", index=False)
        logger.info(f"Saved {len(results)} pseudo-labels to {output_path}")
    else:
        logger.warning("No results generated.")


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels with Upstage API")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output Parquet file")
    parser.add_argument("--name", type=str, default="dataset", help="Dataset name prefix")
    parser.add_argument("--limit", type=int, default=0, help="Max images to process (0 for all)")
    parser.add_argument("--enhance", action="store_true", help="Enable Rembg + Perspective Correction")

    args = parser.parse_args()

    api_key = os.environ.get("UPSTAGE_API_KEY")
    if not api_key:
        logger.error("UPSTAGE_API_KEY not found in environment variables.")
        return

    # Gather images
    root = Path(args.image_dir)
    image_paths = list(root.glob("*.jpg")) + list(root.glob("*.png")) + list(root.glob("*.jpeg"))

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    logger.info(f"Found {len(image_paths)} images to process.")

    asyncio.run(process_images(image_paths, api_key, Path(args.output), args.name, enhance=args.enhance))


if __name__ == "__main__":
    main()
