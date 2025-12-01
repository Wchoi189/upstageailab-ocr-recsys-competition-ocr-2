#!/usr/bin/env python3
"""Verify VLM files exist on disk.

Run this script to check if all VLM module files were created successfully.
"""

from pathlib import Path

BASE = Path(__file__).parent.resolve()
VLM_DIR = BASE / "agent_qms" / "vlm"

print(f"Checking VLM module files...")
print(f"Base directory: {BASE}")
print(f"VLM directory: {VLM_DIR}")
print(f"VLM directory exists: {VLM_DIR.exists()}\n")

if not VLM_DIR.exists():
    print("ERROR: VLM directory does not exist!")
    print(f"Expected path: {VLM_DIR}")
    exit(1)

# Expected files
expected_files = [
    # Core files
    VLM_DIR / "__init__.py",
    VLM_DIR / "core" / "__init__.py",
    VLM_DIR / "core" / "contracts.py",
    VLM_DIR / "core" / "interfaces.py",
    VLM_DIR / "core" / "preprocessor.py",
    VLM_DIR / "core" / "client.py",
    VLM_DIR / "core" / "template.py",
    # Backends
    VLM_DIR / "backends" / "__init__.py",
    VLM_DIR / "backends" / "base.py",
    VLM_DIR / "backends" / "openrouter.py",
    VLM_DIR / "backends" / "solar_pro2.py",
    VLM_DIR / "backends" / "cli_qwen.py",
    # Integrations
    VLM_DIR / "integrations" / "__init__.py",
    VLM_DIR / "integrations" / "via.py",
    VLM_DIR / "integrations" / "reports.py",
    VLM_DIR / "integrations" / "experiment_tracker.py",
    # CLI
    VLM_DIR / "cli" / "__init__.py",
    VLM_DIR / "cli" / "analyze_image_defects.py",
    # Utils
    VLM_DIR / "utils" / "__init__.py",
    VLM_DIR / "utils" / "paths.py",
    # Prompts
    VLM_DIR / "prompts" / "__init__.py",
    VLM_DIR / "prompts" / "manager.py",
    VLM_DIR / "prompts" / "few_shot_examples.json",
    # Documentation
    VLM_DIR / "README.md",
    VLM_DIR / "CHANGELOG.md",
    VLM_DIR / "requirements.txt",
    VLM_DIR / "docs" / "CODING_STANDARDS.md",
    VLM_DIR / "docs" / "workflow.mmd",
    # Templates
    VLM_DIR / "templates" / "image_analysis.md",
    # Docker
    BASE / "docker" / "Dockerfile.vlm",
    BASE / "docker" / "docker-compose.vlm.yml",
]

missing = []
existing = []

for file_path in expected_files:
    if file_path.exists():
        existing.append(file_path)
        print(f"✓ {file_path.relative_to(BASE)}")
    else:
        missing.append(file_path)
        print(f"✗ MISSING: {file_path.relative_to(BASE)}")

print(f"\nSummary:")
print(f"  Existing: {len(existing)}/{len(expected_files)}")
print(f"  Missing: {len(missing)}/{len(expected_files)}")

if missing:
    print(f"\nMissing files:")
    for f in missing:
        print(f"  - {f.relative_to(BASE)}")
    exit(1)
else:
    print("\n✓ All files exist!")
    exit(0)
