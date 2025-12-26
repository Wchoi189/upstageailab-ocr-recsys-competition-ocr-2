#!/usr/bin/env python3
import asyncio
import logging
import os
import cv2
import aiohttp
import numpy as np
from pathlib import Path
from rembg import remove
from ocr.utils.sepia_enhancement import enhance_sepia
from ocr.inference.preprocessing_pipeline import apply_optional_perspective_correction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "https://api.upstage.ai/v1/document-ai/ocr"

def apply_full_pipeline(image_path: Path) -> np.ndarray:
    logger.info(f"Loading {image_path}...")
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"Could not load {image_path}")

    # 1. Rembg (White BG) - Replicated logic from generate_pseudo_labels.py
    logger.info("Step 1: Rembg (Background Removal)...")
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    output_rgba = remove(image_rgb)
    alpha = output_rgba[:, :, 3] / 255.0
    foreground_rgb = output_rgba[:, :, :3]
    bg_color = [255, 255, 255]
    composite = np.zeros_like(foreground_rgb)
    for c in range(3):
        composite[:, :, c] = alpha * foreground_rgb[:, :, c] + (1 - alpha) * bg_color[c]
    image_rembg = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    # 2. Perspective Correction
    logger.info("Step 2: Perspective Correction...")
    # apply_optional_perspective_correction does its own rembg check usually, but we fed it cleaner input?
    # Actually, apply_optional_perspective_correction CALLS remove_background_and_mask internally if we rely on it.
    # But generate_pseudo_labels.py calls remove() EXPLICITLY first, then passes result to apply_optional_perspective_correction.
    # Wait. generate_pseudo_labels.py:67-81 does manual rembg.
    # Then line 86 calls apply_optional_perspective_correction(image_rembg, ...).
    # If image_rembg is already white-bg, apply_optional_perspective_correction might fail to find a mask if it expects complex BG?
    # Or it works fine? The script was written that way, so I assume it works.

    corrected_image = apply_optional_perspective_correction(
        image_rembg,
        enable_perspective_correction=True,
        return_matrix=False # We don't need matrix for this visual test, just the image to send
    )

    # 3. Sepia
    logger.info("Step 3: Sepia Enhancement...")
    final_img = enhance_sepia(corrected_image)

    return final_img

async def test_full_chain(image_path: str, api_key: str):
    path = Path(image_path)

    # Run Pipeline
    try:
        processed_img = apply_full_pipeline(path)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return

    # Save for visualization
    viz_path = "validation_viz/full_chain_test_1454.jpg"
    cv2.imwrite(viz_path, processed_img)
    logger.info(f"Saved processed image to {viz_path}")

    # Prepare for API
    success, buf = cv2.imencode(".jpg", processed_img)
    if not success:
        logger.error("Encoding failed")
        return
    image_bytes = buf.tobytes()

    # Call API
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("document", image_bytes, filename=path.name)

        logger.info("Sending to Upstage API...")
        async with session.post(API_URL, headers=headers, data=data) as response:
            if response.status == 200:
                result = await response.json()
                texts = []
                polygons = []
                for page in result.get("pages", []):
                    for word in page.get("words", []):
                        texts.append(word.get("text"))
                        bbox = word.get("boundingBox", {}).get("vertices", [])
                        poly = [[int(v.get("x", 0)), int(v.get("y", 0))] for v in bbox]
                        polygons.append(poly)

                logger.info(f"Success! Found {len(texts)} text elements.")

                # Visualize
                viz_img = processed_img.copy()
                for poly, text in zip(polygons, texts):
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(viz_img, [pts], True, (0, 255, 0), 2)
                    if len(poly) > 0:
                         cv2.putText(viz_img, text[:20], (poly[0][0], poly[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                final_viz_path = "validation_viz/full_chain_viz_1454.jpg"
                cv2.imwrite(final_viz_path, viz_img)
                logger.info(f"Saved visualization to {final_viz_path}")

                if len(texts) > 0:
                    logger.info(f"First 5 texts: {texts[:5]}")
            else:
                logger.error(f"API Error {response.status}: {await response.text()}")

def main():
    api_key = os.environ.get("UPSTAGE_API_KEY")
    if not api_key:
        logger.error("UPSTAGE_API_KEY not found.")
        return

    image_path = "data/datasets/images/train/drp.en_ko.in_house.selectstar_001454.jpg"
    asyncio.run(test_full_chain(image_path, api_key))

if __name__ == "__main__":
    main()
