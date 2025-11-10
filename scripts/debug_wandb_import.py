#!/usr/bin/env python3
"""Test wandb import specifically to identify the hang."""

import sys
import time
import signal

def timeout_handler(signum, frame):
    print("\n[TIMEOUT] Import timed out!")
    sys.exit(1)

# Set a timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

print("[TEST] Testing wandb import...")
print("[TEST] This may take a while or hang...")
sys.stdout.flush()

start_time = time.time()
try:
    import wandb
    elapsed = time.time() - start_time
    signal.alarm(0)  # Cancel timeout
    print(f"[OK] wandb import successful in {elapsed:.2f}s")
    print(f"[INFO] wandb version: {wandb.__version__ if hasattr(wandb, '__version__') else 'unknown'}")
    sys.stdout.flush()
except Exception as e:
    elapsed = time.time() - start_time
    signal.alarm(0)  # Cancel timeout
    print(f"[ERROR] wandb import failed after {elapsed:.2f}s: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)

# Test wandb.finish() which is called in train.py
print("\n[TEST] Testing wandb.finish()...")
sys.stdout.flush()
start_time = time.time()
try:
    wandb.finish()
    elapsed = time.time() - start_time
    print(f"[OK] wandb.finish() completed in {elapsed:.2f}s")
    sys.stdout.flush()
except Exception as e:
    elapsed = time.time() - start_time
    print(f"[ERROR] wandb.finish() failed after {elapsed:.2f}s: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)

print("\n[SUCCESS] All wandb tests passed!")
sys.stdout.flush()

