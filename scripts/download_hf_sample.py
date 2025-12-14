import pandas as pd
import io
from PIL import Image

# Use the test split as it's smaller (18MB vs 248MB)
url = "https://huggingface.co/datasets/mychen76/invoices-and-receipts_ocr_v1/resolve/main/data/test-00000-of-00001-af2d92d1cee28514.parquet"
print(f"Reading {url}...")
try:
    df = pd.read_parquet(url)
    print("Loaded dataframe.")
    print(f"Columns: {df.columns}")

    # Inspect first row
    row = df.iloc[0]
    # 'image' column usually contains valid image data
    # In HF datasets converted to parquet, it's often a dictionary with 'bytes'

    img_data = row['image']
    print(f"Image data type: {type(img_data)}")

    if isinstance(img_data, dict) and 'bytes' in img_data:
        image_bytes = img_data['bytes']
    elif isinstance(img_data, bytes):
        image_bytes = img_data
    else:
        print("Unknown image format")
        image_bytes = None

    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes))
        save_path = "apps/ocr-inference-console/public/demo-hf.jpg"
        image.save(save_path)
        print(f"Saved {save_path}")

except Exception as e:
    print(f"Error: {e}")
