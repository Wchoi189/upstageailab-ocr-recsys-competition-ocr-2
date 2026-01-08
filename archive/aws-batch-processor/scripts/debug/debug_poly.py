
import requests
import boto3
import os
import json
from pathlib import Path

# Load Key
api_key = os.getenv("UPSTAGE_API_KEY")
if not api_key:
    with open(".env.local") as f:
        for line in f:
            if line.startswith("UPSTAGE_API_KEY="):
                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

if not api_key:
    print("No API Key")
    exit(1)

# Get image
s3 = boto3.client('s3')
s3.download_file('ocr-batch-processing', 'images/drp.en_ko.in_house.selectstar_000003.jpg', 'test_poly.jpg')

# Request
headers = {"Authorization": f"Bearer {api_key}"}
files = {"document": open("test_poly.jpg", "rb")}
resp = requests.post(
    "https://api.upstage.ai/v1/information-extraction",
    headers=headers,
    data={"model": "receipt-extraction"},
    files=files
)

try:
    data = resp.json()
    if 'fields' in data and len(data['fields']) > 0:
        print(json.dumps(data['fields'][0], indent=2))
    else:
        print("No fields found or error:", data)
except Exception as e:
    print("Error:", e, resp.text)
