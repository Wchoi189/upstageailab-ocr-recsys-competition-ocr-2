#!/usr/bin/env python3
"""Script to create all VLM module files.

This script ensures all files are created with proper absolute paths.
"""

import json
from pathlib import Path

BASE = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2")
VLM_DIR = BASE / "agent_qms" / "vlm"

# Create directory structure
dirs = [
    VLM_DIR / "cli",
    VLM_DIR / "core",
    VLM_DIR / "backends",
    VLM_DIR / "integrations",
    VLM_DIR / "prompts" / "markdown",
    VLM_DIR / "prompts" / "jinja2",
    VLM_DIR / "templates",
    VLM_DIR / "via",
    VLM_DIR / "docs",
    VLM_DIR / "utils",
    BASE / "tests" / "agent_qms" / "vlm",
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"Created: {d}")

print(f"\nCreated {len(dirs)} directories")
print(f"VLM directory exists: {VLM_DIR.exists()}")
print(f"VLM directory: {VLM_DIR}")
