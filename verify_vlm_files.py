#!/usr/bin/env python3
"""Verify VLM files exist and create missing ones."""

import sys
from pathlib import Path

BASE = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2")
VLM_DIR = BASE / "agent_qms" / "vlm"

print(f"Base path: {BASE}")
print(f"Base exists: {BASE.exists()}")
print(f"VLM dir: {VLM_DIR}")
print(f"VLM dir exists: {VLM_DIR.exists()}")

if VLM_DIR.exists():
    files = list(VLM_DIR.rglob("*"))
    print(f"\nFound {len(files)} items in VLM directory:")
    for f in sorted(files)[:20]:
        print(f"  {f.relative_to(BASE)}")
else:
    print(f"\nVLM directory does not exist. Creating...")
    VLM_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created: {VLM_DIR}")

# Test write
test_file = VLM_DIR / "VERIFICATION_TEST.txt"
test_file.write_text("This is a test file created by verify script")
print(f"\nTest file written: {test_file}")
print(f"Test file exists: {test_file.exists()}")
print(f"Test file content: {test_file.read_text()}")
