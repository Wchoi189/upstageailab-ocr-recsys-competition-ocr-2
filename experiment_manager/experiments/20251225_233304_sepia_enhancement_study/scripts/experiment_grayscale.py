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

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert back to BGR so it saves/encodes as a standard 3-channel image for consistency
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

async def test_grayscale(image_path: str, api_key: str):
    path = Path(image_path)
    if not path.exists():
        logger.error(f"Image not found: {path}")
        return

    logger.info(f"Loading {path}...")
    original_image = cv2.imread(str(path))

    # 1. Pipeline with Grayscale instead of Sepia
    # We will keep Rembg+Perspective to be fair, but swap Sepia for Gray
    logger.info("Applying Rembg...")
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    output_rgba = remove(image_rgb)
    alpha = output_rgba[:, :, 3] / 255.0
    foreground_rgb = output_rgba[:, :, :3]
    bg_color = [255, 255, 255]
    composite = np.zeros_like(foreground_rgb)
    for c in range(3):
        composite[:, :, c] = alpha * foreground_rgb[:, :, c] + (1 - alpha) * bg_color[c]
    image_rembg = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    logger.info("Applying Perspective...")
    corrected_image = apply_optional_perspective_correction(
        image_rembg,
        enable_perspective_correction=True,
        return_matrix=False
    )

    logger.info("Applying Grayscale...")
    gray_img = apply_grayscale(corrected_image)

    # Save visual
    viz_path = "validation_viz/grayscale_test_1454.jpg"
    cv2.imwrite(viz_path, gray_img)
    logger.info(f"Saved grayscale image to {viz_path}")

    # Encode
    success, buf = cv2.imencode(".jpg", gray_img)
    if not success:
        logger.error("Encoding failed")
        return
    image_bytes = buf.tobytes()

    # Call API
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field("document", image_bytes, filename=path.name)

        logger.info("Sending Grayscale to Upstage API...")
        async with session.post(API_URL, headers=headers, data=data) as response:
            if response.status == 200:
                result = await response.json()
                texts = []
                for page in result.get("pages", []):
                    for word in page.get("words", []):
                        texts.append(word.get("text"))

                logger.info(f"Success! Found {len(texts)} text elements with Grayscale.")
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
    asyncio.run(test_grayscale(image_path, api_key))

if __name__ == "__main__":
    main()
