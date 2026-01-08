import os
import requests
import json
from pathlib import Path

# Setup
BASE_DIR = Path(__file__).resolve().parent.parent
API_KEY = os.getenv("UPSTAGE_API_KEY")

if not API_KEY:
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

IMAGE_PATH = BASE_DIR / "test_poly.jpg"
if not IMAGE_PATH.exists():
    print(f"Error: {IMAGE_PATH} not found")
    exit(1)

headers = {"Authorization": f"Bearer {API_KEY}"}

def test_sync_prebuilt_with_location():
    print("\n--- Testing Sync Prebuilt (Receipt) with location=true ---")
    url = "https://api.upstage.ai/v1/information-extraction"
    
    # Try sending location=true (undocumented but worth trying)
    data = {
        "model": "receipt-extraction",
        "location": "true" 
    }
    files = {"document": open(IMAGE_PATH, "rb")}
    
    try:
        response = requests.post(url, headers=headers, data=data, files=files)
        if response.status_code == 200:
            result = response.json()
            # print(json.dumps(result, indent=2)[:500] + "...")
            
            # Check for polygons
            has_poly = False
            fields = result.get('fields', [])
            for field in fields:
                if 'boundingPoly' in field or 'boundingBoxes' in field:
                    has_poly = True
                    break
            
            if has_poly:
                print("SUCCESS: Polygons found in Sync Prebuilt response!")
            else:
                print("FAILURE: No polygons found in Sync Prebuilt response.")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

def test_document_parse():
    print("\n--- Testing Document Parse (Digitization) ---")
    url = "https://api.upstage.ai/v1/document-digitization"
    
    data = {"model": "document-parse"}
    files = {"document": open(IMAGE_PATH, "rb")}
    
    try:
        response = requests.post(url, headers=headers, data=data, files=files)
        if response.status_code == 200:
            result = response.json()
            # Save raw output
            with open("test_parse_result.json", "w") as f:
                json.dump(result, f, indent=2)
            print("Saved result to test_parse_result.json")
            
            # Check for coordinates in elements
            elements = result.get('elements', [])
            if elements:
                first_el = elements[0]
                print(f"First Element: {json.dumps(first_el, indent=2)}")
                if 'coordinates' in first_el or 'boundingBox' in first_el:
                    print("SUCCESS: Coordinates found in Document Parse!")
                else:
                    print("WARNING: No explicit coordinates key found in first element (check file).")
            else:
                print("WARNING: No elements returned.")
                
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_sync_prebuilt_with_location()
    test_document_parse()
