#!/usr/bin/env python3
import asyncio
import logging
import os
from pathlib import Path

import aiohttp
import cv2
import numpy as np
from rembg import remove

from ocr.utils.sepia_enhancement import enhance_sepia

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "https://api.upstage.ai/v1/document-ai/ocr"


def apply_rembg_white_bg(image: np.ndarray) -> np.ndarray:
    # Convert to RGB for rembg
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_rgba = remove(image_rgb)

    # Extract alpha and composite over white
    alpha = output_rgba[:, :, 3] / 255.0
    foreground_rgb = output_rgba[:, :, :3]
    bg_color = [255, 255, 255]
    composite = np.zeros_like(foreground_rgb)
    for c in range(3):
        composite[:, :, c] = alpha * foreground_rgb[:, :, c] + (1 - alpha) * bg_color[c]

    # Back to BGR
    return cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)


async def test_rembg_sepia(image_path: str, api_key: str):
    path = Path(image_path)
    if not path.exists():
        logger.error(f"Image not found: {path}")
        return

    logger.info(f"Loading {path}...")
    img = cv2.imread(str(path))

    # 1. Apply Rembg
    logger.info("Applying Rembg (White BG)...")
    rembg_img = apply_rembg_white_bg(img)

    # 2. Apply Sepia
    logger.info("Applying Sepia enhancement...")
    final_img = enhance_sepia(rembg_img)

    # Save for visualization
    viz_path = "validation_viz/rembg_sepia_test_1454.jpg"
    cv2.imwrite(viz_path, final_img)
    logger.info(f"Saved combined image to {viz_path}")

    # Prepare for API
    success, buf = cv2.imencode(".jpg", final_img)
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
                for page in result.get("pages", []):
                    for word in page.get("words", []):
                        texts.append(word.get("text"))

                logger.info(f"Success! Found {len(texts)} text elements.")
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
    asyncio.run(test_rembg_sepia(image_path, api_key))


if __name__ == "__main__":
    main()
