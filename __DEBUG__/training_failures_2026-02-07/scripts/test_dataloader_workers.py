#!/usr/bin/env python3
"""
DataLoader num_workers Systematic Test
Tests different num_workers configurations to isolate CUDA+multiprocessing issue
"""

import sys
import time
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp


class DummyDataset(Dataset):
    """Simple dataset for testing DataLoader"""
    def __init__(self, size=100, use_cuda=False):
        self.size = size
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Return CPU tensors - will be moved to GPU by model
        data = torch.randn(3, 32, 32)
        label = torch.tensor(idx % 10)
        return data, label


def test_dataloader(num_workers, pin_memory, persistent_workers, use_cuda_data=False):
    """Test DataLoader with specific configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: num_workers={num_workers}, pin_memory={pin_memory}, "
          f"persistent_workers={persistent_workers}, cuda_data={use_cuda_data}")
    print(f"{'='*60}")

    try:
        dataset = DummyDataset(size=20, use_cuda=use_cuda_data)

        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

        print(f"✓ DataLoader created successfully")

        # Test iteration
        start = time.time()
        batch_count = 0
        for batch_idx, (data, labels) in enumerate(loader):
            batch_count += 1
            if batch_idx == 0:
                print(f"  First batch: data.shape={data.shape}, device={data.device}")

            # Simulate moving to CUDA (if available)
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                data = data.to(device)
                labels = labels.to(device)

                # Simple computation
                result = data * 2 + 1

                # Cleanup
                del result

            if batch_idx >= 4:  # Test first 5 batches
                break

        elapsed = time.time() - start
        print(f"✅ SUCCESS: Processed {batch_count} batches in {elapsed:.2f}s")

        # Cleanup
        del loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("  DataLoader num_workers CUDA Test Suite")
    print("="*60)

    # Check CUDA availability
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # Test configurations
    configs = [
        # (num_workers, pin_memory, persistent_workers, use_cuda_data)
        (0, False, False, False),  # Baseline: no workers, no CUDA
        (0, True, False, False),   # no workers, with pin_memory
        (1, False, False, False),  # 1 worker, no pin_memory
        (1, True, False, False),   # 1 worker, with pin_memory
        (1, False, True, False),   # 1 worker, persistent
        (2, False, False, False),  # 2 workers, no pin_memory
        (2, False, True, False),   # 2 workers, persistent
        (2, True, True, False),    # 2 workers, pin_memory + persistent (fails expected)
    ]

    results = {}
    for num_workers, pin_memory, persistent, use_cuda in configs:
        key = f"w{num_workers}_pm{pin_memory}_pw{persistent}_cd{use_cuda}"
        success = test_dataloader(num_workers, pin_memory, persistent, use_cuda)
        results[key] = success

        # Small delay between tests
        time.sleep(0.5)

    # Summary
    print("\n" + "="*60)
    print("  Test Results Summary")
    print("="*60)

    for config, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {config}")

    # Analysis
    print("\n" + "="*60)
    print("  Analysis")
    print("="*60)

    working_configs = [k for k, v in results.items() if v]
    failing_configs = [k for k, v in results.items() if not v]

    print(f"\nWorking configurations: {len(working_configs)}/{len(results)}")
    print(f"Failing configurations: {len(failing_configs)}/{len(results)}")

    if failing_configs:
        print("\n⚠️  The following configurations failed:")
        for config in failing_configs:
            print(f"  - {config}")

        # Pattern analysis
        if all('_w0_' in c for c in working_configs):
            print("\n💡 Pattern: Only num_workers=0 works")
            print("   Recommendation: Disable multiprocessing for CUDA training")
        elif all('_pmTrue_' in c for c in failing_configs):
            print("\n💡 Pattern: pin_memory=True causes failures with workers")
            print("   Recommendation: Set pin_memory=False")
        else:
            print("\n💡 Pattern: Mixed - requires deeper investigation")


if __name__ == "__main__":
    # Set multiprocessing start method (important for CUDA)
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method: spawn")
    except RuntimeError:
        print("Multiprocessing start method already set")

    main()
