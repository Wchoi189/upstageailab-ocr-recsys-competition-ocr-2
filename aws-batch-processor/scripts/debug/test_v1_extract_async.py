import os
import time
import requests
import json
from pathlib import Path

# Setup
BASE_DIR = Path(__file__).resolve().parent.parent.parent
API_KEY = os.getenv("UPSTAGE_API_KEY")

if not API_KEY:
    # Try reading .env.local manually if not in env
    try:
        env_path = BASE_DIR / ".env.local"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("UPSTAGE_API_KEY="):
                        API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    except Exception:
        pass

if not API_KEY:
    print("Error: UPSTAGE_API_KEY not found")
    exit(1)

API_URL = "https://api.upstage.ai/v1/information-extraction/async"
TASK_URL = "https://api.upstage.ai/v1/information-extraction/jobs/{}"
IMAGE_PATH = BASE_DIR / "test_poly.jpg"

if not IMAGE_PATH.exists():
    print(f"Error: {IMAGE_PATH} not found")
    exit(1)

print(f"Submitting {IMAGE_PATH}...")

headers = {"Authorization": f"Bearer {API_KEY}"}
files = {"document": open(IMAGE_PATH, "rb")}
data = {
    "model": "universal-document-extraction-2025-03-27", 
    "location": "true", 
    "task": "prebuilt_receipt_extraction"
}

try:
    response = requests.post(API_URL, headers=headers, files=files, data=data)
    if response.status_code != 200:
        print(f"Submit Error {response.status_code}: {response.text}")
        exit(1)
        
    task_id = response.json().get("task_id")
    print(f"Task ID: {task_id}")
    
    # Poll
    for i in range(30):
        print(f"Polling attempt {i+1}...")
        time.sleep(2)
        poll_res = requests.get(TASK_URL.format(task_id), headers=headers)
        if poll_res.status_code == 200:
            status = poll_res.json()
            task_status = status.get("status")
            print(f"Status: {task_status}")
            
            if task_status == "completed":
                result = status.get("result")
                # Check for boundingBoxes
                print("\n--- Result Analysis ---")
                if "boundingBoxes" in str(result): # Simple string check first
                    print("SUCCESS: 'boundingBoxes' found in result!")
                else:
                    print("WARNING: 'boundingBoxes' NOT found in result string.")
                
                # Deeper check
                fields = result.get('fields', [])
                poly_count = 0
                for field in fields:
                    if 'boundingBoxes' in field:
                        poly_count += 1
                    if 'boundingPoly' in field: # Check for this too
                         pass # older api style
                
                print(f"Found {poly_count} fields with 'boundingBoxes'")
                
                # Save full result for inspection
                with open("test_async_result.json", "w") as f:
                    json.dump(result, f, indent=2)
                print("Saved result to test_async_result.json")
                break
            elif task_status in ["failed", "cancelled"]:
                print("Task failed")
                break
        else:
            print(f"Poll Error: {poll_res.status_code}")
            
except Exception as e:
    print(f"Exception: {e}")
