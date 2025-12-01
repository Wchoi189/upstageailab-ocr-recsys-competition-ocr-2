#!/usr/bin/env python3
"""Create all VLM module files.

This script creates all necessary files for the VLM module.
Run this script to ensure all files are created in the correct location.
"""

import json
from pathlib import Path

# Base paths
BASE = Path(__file__).parent.resolve()
VLM_DIR = BASE / "agent_qms" / "vlm"

print(f"Base directory: {BASE}")
print(f"VLM directory: {VLM_DIR}")

# Create all directories
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
    print(f"✓ Created directory: {d.relative_to(BASE)}")

# Files to create - using a simplified approach
# We'll create the essential files first, then you can run this script
# to verify they're in the right place

files_created = 0

# Core __init__ files
init_files = [
    VLM_DIR / "__init__.py",
    VLM_DIR / "cli" / "__init__.py",
    VLM_DIR / "core" / "__init__.py",
    VLM_DIR / "backends" / "__init__.py",
    VLM_DIR / "integrations" / "__init__.py",
    VLM_DIR / "prompts" / "__init__.py",
    VLM_DIR / "utils" / "__init__.py",
]

for f in init_files:
    if not f.exists():
        f.write_text('"""Module initialization."""\n')
        files_created += 1
        print(f"✓ Created: {f.relative_to(BASE)}")

print(f"\nCreated {files_created} new files")
print(f"\nTo create all remaining files, the write tool needs to use absolute paths.")
print(f"Current VLM directory: {VLM_DIR}")
print(f"Directory exists: {VLM_DIR.exists()}")
