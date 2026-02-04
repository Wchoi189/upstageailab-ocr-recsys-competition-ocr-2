import lmdb
import sys
from PIL import Image
import io
from tqdm import tqdm

def inspect_lmdb(path):
    print(f"Inspecting LMDB at {path}")
    try:
        env = lmdb.open(path, readonly=True, lock=False)
    except Exception as e:
        print(f"Error opening LMDB: {e}")
        return

    with env.begin() as txn:
        num_samples_bytes = txn.get("num-samples".encode("utf-8"))
        if not num_samples_bytes:
            print("Key 'num-samples' not found.")
            return

        num_samples = int(num_samples_bytes.decode("utf-8"))
        print(f"Total samples: {num_samples}")

        sizes = []
        aspect_ratios = []

        # Check first 100 images
        limit = 100
        print(f"Checking first {limit} images...")

        for i in range(1, min(num_samples, limit) + 1):
            key = f"image-{i:09d}".encode("utf-8")
            img_bytes = txn.get(key)
            if not img_bytes:
                print(f"Missing image key: {key}")
                continue

            try:
                img = Image.open(io.BytesIO(img_bytes))
                sizes.append(img.size)
                aspect_ratios.append(img.size[0] / img.size[1])

                label_key = f"label-{i:09d}".encode("utf-8")
                label_bytes = txn.get(label_key)
                label = label_bytes.decode("utf-8") if label_bytes else "[MISSING]"

                if i <= 5:
                    print(f"Sample {i}: Size={img.size}, Label='{label}'")
            except Exception as e:
                print(f"Error decoding image {i}: {e}")

    # Stats
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"\nStats for first {len(sizes)} images:")
        print(f"Width: min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.1f}")
        print(f"Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.1f}")

        exact_32_128 = sum(1 for s in sizes if s == (128, 32))
        print(f"Exact 128x32: {exact_32_128}/{len(sizes)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_lmdb.py <lmdb_path>")
    else:
        inspect_lmdb(sys.argv[1])
