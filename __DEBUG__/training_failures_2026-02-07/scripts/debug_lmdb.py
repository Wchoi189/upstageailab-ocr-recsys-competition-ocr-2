
import sys
import os
import time
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

# sys.path hack removed - run with 'uv run python script.py'
# sys.path hack removed - use imported PROJECT_ROOT
from ocr.core.utils.path_utils import PROJECT_ROOT

from ocr.domains.recognition.data.lmdb_dataset import LMDBRecognitionDataset

# Mock tokenizer
class MockTokenizer:
    def encode(self, text):
        return [1, 2, 3] # Dummy tokens

def main():
    # Attempt to find the dataset
    lmdb_path = PROJECT_ROOT / "data/processed/recognition/aihub_lmdb_validation"
    if not lmdb_path.exists():
        print(f"Path not found: {lmdb_path}")
        # Try raw path just in case
        lmdb_path = PROJECT_ROOT / "data/aihub_lmdb_validation"

    if not lmdb_path.exists():
        print("Could not find aihub_lmdb_validation in expected paths.")
        return

    print(f"Testing LMDB at: {lmdb_path}")

    dataset = LMDBRecognitionDataset(
        lmdb_path=lmdb_path,
        tokenizer=MockTokenizer(),
        max_len=25,
        transform=transforms.Compose([
            transforms.Resize((32, 100)),
            transforms.ToTensor()
        ])
    )

    print(f"Dataset length: {len(dataset)}")

    # Test single access
    start = time.time()
    item = dataset[0]
    duration = time.time() - start
    print(f"Single item access time: {duration:.4f}s")
    print(f"Item keys: {item.keys()}")
    print(f"Label: {item['label']}")

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)

    print("Testing DataLoader iteration (1 batch)...")
    start = time.time()
    for batch in loader:
        print("Batch loaded!")
        print(f"Batch keys: {batch.keys()}")
        print(f"Images shape: {len(batch['image'])} (list of PIL images)")
        break
    duration = time.time() - start
    print(f"DataLoader batch time: {duration:.4f}s")

if __name__ == "__main__":
    main()
