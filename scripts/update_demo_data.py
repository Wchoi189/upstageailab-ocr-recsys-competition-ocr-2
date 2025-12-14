import requests
import json
import shutil
import os

API_URL = "http://localhost:8000/ocr/predict"
IMAGE_PATH = "apps/ocr-inference-console/public/demo-hf.jpg"
DEST_IMAGE = "apps/ocr-inference-console/public/demo.jpg"
DEST_JSON = "apps/ocr-inference-console/public/demo.json"

def update_demo():
    print(f"Uploading {IMAGE_PATH} to {API_URL}...")
    with open(IMAGE_PATH, "rb") as f:
        response = requests.post(API_URL, files={"file": f})

    if response.status_code != 200:
        print(f"Error: {response.text}")
        return

    data = response.json()
    predictions = data.get("predictions", [])
    print(f"Received {len(predictions)} predictions.")

    # Convert to demo.json format (Map of ID -> Object)
    demo_data = {}
    for idx, pred in enumerate(predictions):
        demo_data[str(idx)] = {
            "points": pred["points"],
            "transcription": "", # No text from detection model
            "confidence": pred["confidence"]
        }

    # Save JSON
    with open(DEST_JSON, "w") as f:
        json.dump(demo_data, f, indent=2)
    print(f"Saved {DEST_JSON}")

    # Copy Image
    shutil.copy2(IMAGE_PATH, DEST_IMAGE)
    print(f"Copied image to {DEST_IMAGE}")

if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"{IMAGE_PATH} not found. Run download_hf_sample.py first.")
    else:
        update_demo()
