#!/usr/bin/env python3
import asyncio
import logging
import os
from pathlib import Path

import aiohttp
import cv2

from ocr.utils.sepia_enhancement import enhance_sepia

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "https://api.upstage.ai/v1/document-ai/ocr"


async def test_sepia_on_image(image_path: str, api_key: str):
    path = Path(image_path)
    if not path.exists():
        logger.error(f"Image not found: {path}")
        return

    logger.info(f"Loading {path}...")
    img = cv2.imread(str(path))

    # Apply Sepia
    logger.info("Applying Sepia enhancement...")
    enhanced_img = enhance_sepia(img)

    # Save for visualization
    viz_path = "validation_viz/sepia_test_1454.jpg"
    cv2.imwrite(viz_path, enhanced_img)
    logger.info(f"Saved enhanced image to {viz_path}")

    # Prepare for API
    success, buf = cv2.imencode(".jpg", enhanced_img)
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
                    logger.warning("No text detected even with Sepia.")
            else:
                logger.error(f"API Error {response.status}: {await response.text()}")


def main():
    api_key = os.environ.get("UPSTAGE_API_KEY")
    if not api_key:
        logger.error("UPSTAGE_API_KEY not found.")
        return

    # Image 1454 from worst performers
    # Need to find where it is. It was in validation_viz, so it likely came from the previous run.
    # The previous run processed `data/zero_prediction_worst_performers`.
    # Let's verify the filename there.
    # From Step 102 (list_dir images_val_canonical), it's `drp.en_ko.in_house.selectstar_001454.jpg`?
    # No, Step 102 did not list 1454.
    # Step 114 listed `viz_drp.en_ko.in_house.selectstar_001454.jpg`.
    # Wait, Step 114 output for `validation_viz` listing:
    # {"name":"viz_drp.en_ko.in_house.selectstar_001454.jpg", "sizeBytes":"348669"}
    # Actually Step 114 list output includes: `viz_drp.en_ko.in_house.selectstar_001454.jpg`
    # Let me re-read Step 114 output carefully.
    # Step 114:
    # viz_drp.en_ko.in_house.selectstar_000699.jpg
    # ...
    # viz_drp.en_ko.in_house.selectstar_001454.jpg
    # So the source image name is `drp.en_ko.in_house.selectstar_001454.jpg`.
    # I need to find where this source image IS.
    # It was in `data/zero_prediction_worst_performers`?
    # Step 82 list_dir `data` showed `zero_prediction_worst_performers`.
    # Let's assume it's in `data/zero_prediction_worst_performers/drp.en_ko.in_house.selectstar_001454.jpg`.

    image_path = "data/datasets/images/train/drp.en_ko.in_house.selectstar_001454.jpg"
    asyncio.run(test_sepia_on_image(image_path, api_key))


if __name__ == "__main__":
    main()
