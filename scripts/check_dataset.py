
import sys
import os
import argparse
from pathlib import Path
import lmdb
import io
from PIL import Image
from tqdm import tqdm

def check_dataset(lmdb_path):
    print(f"Checking LMDB at {lmdb_path}")

    if not os.path.exists(lmdb_path):
        print(f"Error: Path {lmdb_path} does not exist.")
        return

    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn:
        num_samples_bytes = txn.get("num-samples".encode("utf-8"))
        if num_samples_bytes is None:
            print("Error: 'num-samples' key not found.")
            return

        num_samples = int(num_samples_bytes.decode("utf-8"))
        print(f"Dataset claims to have {num_samples} samples.")

        # Check first 10, middle 10, last 10
        indices_to_check = list(range(1, 11)) + \
                           list(range(num_samples // 2, num_samples // 2 + 10)) + \
                           list(range(num_samples - 9, num_samples + 1))

        # Filter indices within valid range
        indices_to_check = [i for i in indices_to_check if 1 <= i <= num_samples]

        valid_count = 0
        for i in indices_to_check:
            image_key = f"image-{i:09d}".encode("utf-8")
            label_key = f"label-{i:09d}".encode("utf-8")

            img_bytes = txn.get(image_key)
            label_bytes = txn.get(label_key)

            if img_bytes is None:
                print(f"Error: Missing image for index {i}")
                continue
            if label_bytes is None:
                print(f"Error: Missing label for index {i}")
                continue

            try:
                img_buf = io.BytesIO(img_bytes)
                img = Image.open(img_buf)
                img.load() # Verify image integrity

                label = label_bytes.decode("utf-8")

                print(f"[{i}] {img.size} Label: {label}")

                if img.size != (128, 32):
                     print(f"WARNING: Image size mismatch at {i}. Expected (128, 32), got {img.size}")

                valid_count += 1

            except Exception as e:
                print(f"Error processing index {i}: {e}")

        print(f"Verified {valid_count}/{len(indices_to_check)} samples.")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, required=True)
    args = parser.parse_args()

    check_dataset(args.lmdb_path)
