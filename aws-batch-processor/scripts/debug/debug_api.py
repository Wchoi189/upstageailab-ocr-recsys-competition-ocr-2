import os
import sys
import boto3
import requests
import json
from pathlib import Path

# Config
API_URL = "https://api.upstage.ai/v1/information-extraction"
IMAGE_S3_PATH = "s3://ocr-batch-processing/images/drp.en_ko.in_house.selectstar_000003.jpg"

def debug_api():
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        print("Please set UPSTAGE_API_KEY env var")
        sys.exit(1)

    print(f"Downloading {IMAGE_S3_PATH}...")
    s3 = boto3.client('s3')
    bucket, key = IMAGE_S3_PATH[5:].split('/', 1)
    
    local_file = "debug_temp.jpg"
    s3.download_file(bucket, key, local_file)
    
    print("Sending to API...")
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(local_file, "rb") as f:
        response = requests.post(
            API_URL,
            headers=headers,
            data={"model": "receipt-extraction"},
            files={"document": f}
        )
        
    if response.status_code == 200:
        res_json = response.json()
        with open("debug_api_response.json", "w") as f:
            json.dump(res_json, f, indent=2)
        print("Saved full response to debug_api_response.json")
        
        # Quick Inspect
        print("\n--- Field Keys found ---")
        for field in res_json.get('fields', [])[:5]:
            print(f"Key: {field.get('key')} | Keys present: {list(field.keys())}")
            
    else:
        print(f"Error: {response.status_code} {response.text}")
        
    if os.path.exists(local_file):
        os.remove(local_file)

if __name__ == "__main__":
    debug_api()
